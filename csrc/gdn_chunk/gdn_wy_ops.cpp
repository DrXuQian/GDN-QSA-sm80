#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "gdn_qsa/wy_contract.hpp"

extern "C" int gdn_wy_forward(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t);
extern "C" int gdn_wy_forward_delivery(
    void const*, void const*, void const*, void const*, void const*, float const*,
    void*, float*, void*, void*, void*, void*, float*, int, int, int, int, bool, cudaStream_t, unsigned);

namespace {
std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> initial,
    bool output_final_state, unsigned delivery) {
  using namespace gdn_qsa::wy;
  TORCH_CHECK(delivery <= 7, "invalid WY delivery mask");
  TORCH_CHECK(q.dim() == 4 && q.size(3) == Dim && k.sizes() == q.sizes(),
              "WY q/k must have identical [B,S,Hk,128] shapes");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == q.size(1) && v.size(3) == Dim,
              "WY v must be [B,S,Hv,128]");
  int64_t const B = q.size(0), S = q.size(1), Hk = q.size(2), Hv = v.size(2);
  TORCH_CHECK(B > 0 && S > 0 && Hk > 0 && Hv > 0 && Hv % Hk == 0, "invalid WY/GVA extents");
  TORCH_CHECK(g.sizes() == torch::IntArrayRef({B, S, Hv}) && beta.sizes() == g.sizes(),
              "WY g/beta must be [B,S,Hv]");
  for (auto const& t : {q, k, v, g, beta}) {
    TORCH_CHECK(t.is_cuda() && t.device() == q.device() && t.is_contiguous(),
                "WY inputs must be contiguous on one PPU");
  }
  for (auto const& t : {q, k, v, beta})
    TORCH_CHECK(t.scalar_type() == torch::kBFloat16, "WY q/k/v/beta must be BF16");
  TORCH_CHECK(g.scalar_type() == torch::kBFloat16 || g.scalar_type() == torch::kFloat32,
              "WY natural-log gate must be BF16 or FP32");
  TORCH_CHECK(B <= INT_MAX && S <= INT_MAX && Hv <= INT_MAX,
              "WY dimensions exceed 32-bit kernel extents");
  int64_t const nt = (S - 1) / Chunk + 1;
  TORCH_CHECK(B <= INT_MAX / Hv && B * Hv <= INT_MAX / nt && B * Hv <= INT_MAX / 4,
              "WY launch grid overflow");
  if (initial.has_value()) {
    auto const& h = *initial;
    TORCH_CHECK(h.device() == q.device() && h.is_contiguous() && h.scalar_type() == torch::kFloat32 &&
                h.sizes() == torch::IntArrayRef({B, Hv, Dim, Dim}),
                "WY initial state must be contiguous FP32 [B,Hv,128,128] on the same PPU");
  }
  c10::cuda::CUDAGuard guard(q.device());
  auto out = torch::empty_like(v);
  auto final = output_final_state ? torch::empty({B, Hv, Dim, Dim}, q.options().dtype(torch::kFloat32))
                                  : torch::empty({0}, q.options().dtype(torch::kFloat32));
  auto w = torch::empty({B * Hv * nt, Chunk, Dim}, q.options());
  auto u = torch::empty_like(w), vn = torch::empty_like(w);
  auto snapshots = torch::empty({B * Hv * nt, Dim, Dim}, q.options());
  auto gates = torch::empty({B * Hv * nt, Chunk}, q.options().dtype(torch::kFloat32));
  int const rc = gdn_wy_forward_delivery(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
      initial.has_value() ? initial->data_ptr<float>() : nullptr, out.data_ptr(),
      output_final_state ? final.data_ptr<float>() : nullptr, w.data_ptr(), u.data_ptr(),
      snapshots.data_ptr(), vn.data_ptr(), gates.data_ptr<float>(), int(B), int(S), int(Hk), int(Hv),
      g.scalar_type() == torch::kFloat32, at::cuda::getCurrentCUDAStream().stream(), delivery);
  TORCH_CHECK(rc == 0, "PPU WY kernel launch failed: status=", rc);
  return {out, final};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("g"), pybind11::arg("beta"), pybind11::arg("initial_state") = pybind11::none(),
        pybind11::arg("output_final_state") = true, pybind11::arg("delivery") = 0);
}
