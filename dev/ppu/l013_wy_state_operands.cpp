#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_tiles.cuh"
#include "gdn_qsa/ppu/wy_state_operands.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;

// Independent physical cube equation, not the candidate's XOR expression.
unsigned hardware(unsigned row, unsigned col) {
  unsigned const line = row / 4, vec = ((row % 4) * 2 + col / 8) ^ (line % 2);
  return (line * 32 + vec * 4 + (col % 8) / 2) * 2 + col % 2;
}
struct Counts { unsigned elements=0, operands=0, owners=0, ordered=0; };

template <unsigned Rows, unsigned Cols>
unsigned addresses(Counts& count, unsigned plant=0) {
  unsigned bad=0;
  for (unsigned rt=0; rt<Rows/16; ++rt) for (unsigned ct=0; ct<Cols/16; ++ct) {
    unsigned base=operand_cube<Rows>(rt,ct);
    if (plant==2) base*=2; // half-element pitch accidentally treated as bytes
    for (unsigned lane=0; lane<32; ++lane) for (unsigned slot=0; slot<8; ++slot) {
      unsigned const native=Native::CLayout{}(lane,slot), r=native%16, c=native/16;
      unsigned at=operand_store<Rows>(rt,ct,lane,slot);
      if (plant==1) at^=8;
      if (plant==3) at=operand_cube<Rows>(rt*16+r,ct*16+c); // cube API misused
      bad += at!=unsigned(swizzle<Rows,Cols>(rt*16+r,ct*16+c));
      bad += at!=base+hardware(r,c);
      ++count.elements;
      for (bool transpose : {false,true}) {
        unsigned const frag=transpose ? Native::BLayout{}(lane,slot) : Native::ALayout{}(lane,slot);
        unsigned const rr=transpose ? frag/16 : frag%16;
        unsigned const cc=transpose ? frag%16 : frag/16;
        bad += base+hardware(rr,cc)!=unsigned(swizzle<Rows,Cols>(rt*16+rr,ct*16+cc));
        ++count.operands;
      }
    }
  }
  return bad;
}

unsigned ownership(Counts& count, unsigned plant=0) {
  unsigned bad=0;
  std::vector<unsigned> h(Dim*ValueTile), v(Chunk*ValueTile);
  std::vector<std::vector<unsigned>> wh(v.size()), kh(h.size());
  for (unsigned warp=0; warp<4-unsigned(plant==4); ++warp) {
    unsigned const c=StateOperandTiles::column(warp);
    bad += 16*c!=unsigned(StateTile::column(warp));
    for (unsigned lane=0; lane<32; ++lane) for (unsigned slot=0; slot<8; ++slot) {
      auto const rc=result_coord(lane,slot);
      for (unsigned k=0; k<4; ++k) {
        unsigned const r=StateOperandTiles::key(warp,k);
        bad += 16*r!=unsigned(StateTile::k_row(warp,k));
        ++h.at(operand_store<Dim>(r,c,lane,slot));
        unsigned const logical=(16*r+rc.row)*ValueTile+16*c+rc.col;
        for (unsigned reduction=0; reduction<4; ++reduction)
          kh.at(logical).push_back(16*(plant==5 ? 3-reduction : reduction));
        ++count.owners; ++count.ordered;
      }
      for (unsigned r=0; r<2; ++r) {
        unsigned const row=StateOperandTiles::value(warp,r);
        bad += 16*row!=unsigned(StateTile::value_row(warp,r));
        ++v.at(operand_store<Chunk>(row,c,lane,slot));
        unsigned const logical=(16*row+rc.row)*ValueTile+16*c+rc.col;
        for (unsigned reduction=0; reduction<8; ++reduction) wh.at(logical).push_back(16*reduction);
        ++count.owners; ++count.ordered;
      }
    }
  }
  for (auto n:h) bad+=n!=1;
  for (auto n:v) bad+=n!=1;
  for (auto const& order:wh) bad+=order!=std::vector<unsigned>{0,16,32,48,64,80,96,112};
  for (auto const& order:kh) bad+=order!=std::vector<unsigned>{0,16,32,48};
  return bad;
}

unsigned selectors(unsigned plant=0) {
  std::array<bool,16384> expected{};
  for (unsigned p:{0u,1u,8u,256u,1280u,2304u})
    for (unsigned s:{0u,2u,16u,80u,144u,208u,4304u})
      for (unsigned o:{0u,4u,32u,544u}) expected[p|s|o]=true;
  unsigned bad=0,cases=0,old=0,valid=0,new_count=0;
  for (unsigned mask=0; mask<expected.size(); ++mask) {
    if (plant==8 && mask==5616) continue;
    bool const admitted=plant==7 && mask==4096 ? true : valid_delivery(mask);
    bad+=admitted!=expected[mask];
    bool const selected=state_operands_selected(plant==6 ? mask & ~4096u : mask);
    bad+=selected!=bool(mask&4096u);
    old+=admitted && mask<4096; valid+=admitted; new_count+=admitted && selected; ++cases;
  }
  return bad+(cases!=16384)+(valid!=168)+(old!=144)+(new_count!=24);
}

int main() {
  Counts count;
  if (addresses<64,128>(count)||addresses<128,32>(count)||addresses<64,32>(count)||
      ownership(count)||selectors()) throw std::runtime_error("operand address mismatch");
  if (count.elements!=14336 || count.operands!=28672 || count.owners!=6144 || count.ordered!=6144)
    throw std::runtime_error("operand denominator mismatch");
  for (unsigned plant=1; plant<=8; ++plant) {
    Counts scratch;
    unsigned const bad=plant<=3 ? addresses<64,128>(scratch,plant) :
                       plant<=5 ? ownership(scratch,plant) : selectors(plant);
    if (!bad) throw std::runtime_error("operand negative escaped");
    std::printf("[WY operands negative] plant=%u bad=%u EXPECTED-RED/PASS\n",plant,bad);
  }
  std::puts("[WY operands] elements=14336 operands=28672 owners=6144 reduction_orders=6144 selectors=16384/168 old144-preserved PASS device_execution=NOT_RUN");
}
