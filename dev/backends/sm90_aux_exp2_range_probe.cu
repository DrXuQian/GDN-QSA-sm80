// Exact CUDA12.8 exponent seam and the actual warp-prefix range reducer.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "aux_exp2_range.cuh"

void check(cudaError_t rc) {
  if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
float value(uint32_t u) { float x; std::memcpy(&x,&u,4); return x; }

__global__ void standard(const float* x,float* y,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) y[i]=exp2f(x[i]);
}
template<bool Guarded>
__global__ void candidate(const float* x,float* y,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    bool safe=gdn::sm90::aux_normal_span(isfinite(x[i]),fabsf(x[i]));
    if constexpr (Guarded) y[i]=gdn::sm90::auxiliary_exp2_guarded(x[i],safe);
    else y[i]=gdn::sm90::auxiliary_exp2<true>(x[i]);
  }
}
template<bool OmitLast = false>
__global__ void prefix_flags(const float* x,int* flags) {
  int lane=threadIdx.x,base=blockIdx.x*64;
  float hi=x[base+lane+32];
  if constexpr (OmitLast) if(lane==31) hi=0.f;
  flags[blockIdx.x*32+lane]=gdn::sm90::collect_aux_normal_span(x[base+lane],hi);
}

int main() {
  std::vector<float> x;
  // Every FP32 value from -124 through -128, not random boundary sampling.
  for(uint32_t u=bits(-124.f);u<=bits(-128.f);++u) x.push_back(value(u));
  for(uint32_t i=0;i<(1u<<20);++i) {
    uint32_t u=i*747796405u+2891336453u;
    x.push_back(value(u)); // exponent/mantissa/sign strata, including specials
  }
  x.insert(x.end(),{0.f,-0.f,126.f,-126.f,127.f,-127.f,INFINITY,-INFINITY,NAN});
  float *dx,*dy;check(cudaMalloc(&dx,x.size()*4));check(cudaMalloc(&dy,x.size()*4));
  check(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
  std::vector<float> ref(x.size()),got(x.size());
  standard<<<(x.size()+255)/256,256>>>(dx,dy,x.size());check(cudaGetLastError());
  check(cudaMemcpy(ref.data(),dy,x.size()*4,cudaMemcpyDeviceToHost));
  candidate<true><<<(x.size()+255)/256,256>>>(dx,dy,x.size());check(cudaGetLastError());
  check(cudaMemcpy(got.data(),dy,x.size()*4,cudaMemcpyDeviceToHost));
  size_t bad=0;
  for(size_t i=0;i<x.size();++i) bad+=bits(ref[i])!=bits(got[i]);
  if(bad) throw std::runtime_error("guarded EX2 differs from standard exp2f");
  candidate<false><<<(x.size()+255)/256,256>>>(dx,dy,x.size());check(cudaGetLastError());
  check(cudaMemcpy(got.data(),dy,x.size()*4,cudaMemcpyDeviceToHost));
  size_t planted=0;
  for(size_t i=0;i<x.size();++i) planted+=bits(ref[i])!=bits(got[i]);
  if(!planted) throw std::runtime_error("unconditional direct EX2 negative escaped");
  check(cudaFree(dx));check(cudaFree(dy));

  std::vector<float> prefixes;
  for(int valid=1;valid<=64;++valid) for(float g:{-.1f,-8.f})
    for(int i=0;i<64;++i) prefixes.push_back(g*float(i<valid?i+1:valid));
  for(int i=0;i<64;++i) prefixes.push_back((i%2?-1.f:1.f)*i);
  for(float span:{std::nextafter(126.f,0.f),126.f,std::nextafter(126.f,INFINITY)})
    for(int i=0;i<64;++i) prefixes.push_back(i==63?-span:0.f);
  for(int poison=0;poison<64;++poison)
    for(int i=0;i<64;++i) prefixes.push_back(i==poison?NAN:0.f);
  for(int i=0;i<64;++i) prefixes.push_back(i==63?-127.f:0.f);
  constexpr int cases=197;
  if(prefixes.size()!=cases*64) throw std::runtime_error("range denominator changed");
  int* df;check(cudaMalloc(&dx,prefixes.size()*4));check(cudaMalloc(&df,cases*32*4));
  check(cudaMemcpy(dx,prefixes.data(),prefixes.size()*4,cudaMemcpyHostToDevice));
  prefix_flags<false><<<cases,32>>>(dx,df);check(cudaGetLastError());
  std::vector<int> flags(cases*32);
  check(cudaMemcpy(flags.data(),df,flags.size()*4,cudaMemcpyDeviceToHost));
  int fast=0,slow=0;
  for(int c=0;c<cases;++c) {
    bool finite=true;float lo=INFINITY,hi=-INFINITY;
    for(int i=0;i<64;++i) {float z=prefixes[c*64+i];finite &= std::isfinite(z);lo=fminf(lo,z);hi=fmaxf(hi,z);}
    bool expected=finite && hi-lo>=0.f && hi-lo<126.f;
    for(int lane=0;lane<32;++lane) if(flags[c*32+lane]!=expected)
      throw std::runtime_error("actual producer range flag disagrees");
    fast+=expected;slow+=!expected;
    if(expected) for(int i=0;i<64;++i) for(int j=0;j<64;++j)
      if(!(prefixes[c*64+i]-prefixes[c*64+j]>-126.f))
        throw std::runtime_error("unsafe direct EX2 admitted");
  }
  // Mutate the actual device producer, not a parallel host bound. The last
  // fixture contains a real -127 difference that this omission falsely admits.
  prefix_flags<true><<<cases,32>>>(dx,df);check(cudaGetLastError());
  std::vector<int> omitted(cases*32);
  check(cudaMemcpy(omitted.data(),df,omitted.size()*4,cudaMemcpyDeviceToHost));
  int omitted_bad=0;
  for(int i=0;i<cases*32;++i) omitted_bad+=omitted[i]!=flags[i];
  for(int lane=0;lane<32;++lane)
    if(flags[(cases-1)*32+lane] || !omitted[(cases-1)*32+lane])
      throw std::runtime_error("omitted-prefix device negative did not distinguish");
  check(cudaFree(dx));check(cudaFree(df));
  std::printf("EX2 seam RAW/PASS n=%zu; unguarded EXPECTED_RED bad=%zu; actual-prefix cases=%d fast=%d fallback=%d all32-lanes/PASS; omitted-prefix EXPECTED_RED bad=%d\n",x.size(),planted,cases,fast,slow,omitted_bad);
}
