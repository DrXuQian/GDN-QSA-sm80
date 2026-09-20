#!/usr/bin/env python3
import argparse
import ctypes
from dataclasses import dataclass

import torch

from gdn_qsa_sm80.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule


class Problem(ctypes.Structure):
    _fields_ = [
        ("schema_version", ctypes.c_uint32),
        ("batch", ctypes.c_int32),
        ("sequence", ctypes.c_int32),
        ("qk_heads", ctypes.c_int32),
        ("value_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("chunk", ctypes.c_int32),
        ("group_chunks", ctypes.c_int32),
    ]


@dataclass
class Metrics:
    output_rel_l1: float
    state_rel_l1: float
    output_max_abs: float
    state_max_abs: float


def relative_l1(got: torch.Tensor, want: torch.Tensor) -> float:
    numerator = (got.float() - want.float()).abs().sum()
    denominator = want.float().abs().sum().clamp_min(1.0e-12)
    return float((numerator / denominator).item())


def load_api(path: str):
    library = ctypes.CDLL(path)
    size = library.gdn_qsa_ppu_workspace_size_v1
    size.argtypes = [ctypes.POINTER(Problem)]
    size.restype = ctypes.c_uint64
    forward = library.gdn_qsa_ppu_forward_bf16_v1
    forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(Problem),
        ctypes.c_void_p,
    ]
    forward.restype = ctypes.c_int
    return size, forward


def run_once(forward, problem, workspace, q, k, v, g, beta):
    output = torch.empty_like(v)
    final_state = torch.empty(
        (problem.batch, problem.value_heads, 128, 128),
        dtype=torch.bfloat16,
        device=q.device,
    )
    stream = torch.cuda.current_stream(q.device).cuda_stream
    rc = forward(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        g.data_ptr(),
        beta.data_ptr(),
        output.data_ptr(),
        final_state.data_ptr(),
        workspace.data_ptr(),
        workspace.numel(),
        ctypes.byref(problem),
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"PPU GDN returned status {rc}")
    torch.cuda.synchronize(q.device)
    return output, final_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    size, forward = load_api(args.library)

    # B2 catches batch/head flattening, Hq1:Hv2 catches GVA ownership, S65
    # catches both a one-token tail and a three-group/two-round affine scan.
    problem = Problem(1, 2, 65, 1, 2, 128, 16, 2)
    workspace_bytes = int(size(ctypes.byref(problem)))
    if workspace_bytes <= 0:
        raise AssertionError("valid problem returned zero workspace")
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    generator = torch.Generator(device=device).manual_seed(0x6A09E667)
    q = (torch.randn((2, 65, 1, 128), generator=generator, device=device) / 16).to(torch.bfloat16)
    k = (torch.randn((2, 65, 1, 128), generator=generator, device=device) / 16).to(torch.bfloat16)
    v = (torch.randn((2, 65, 2, 128), generator=generator, device=device) / 8).to(torch.bfloat16)
    g = (-torch.rand((2, 65, 2), generator=generator, device=device) / 16).to(torch.bfloat16)
    beta = torch.sigmoid(
        torch.randn((2, 65, 2), generator=generator, device=device)
    ).to(torch.bfloat16)

    expanded_q = q.repeat_interleave(2, dim=2)
    expanded_k = k.repeat_interleave(2, dim=2)
    want_output, want_state = torch_recurrent_gated_delta_rule(
        expanded_q,
        expanded_k,
        v,
        g,
        beta,
        output_final_state=True,
    )
    got_output, got_state = run_once(
        forward, problem, workspace, q, k, v, g, beta
    )
    replay_output, replay_state = run_once(
        forward, problem, workspace, q, k, v, g, beta
    )

    metrics = Metrics(
        relative_l1(got_output, want_output),
        relative_l1(got_state, want_state),
        float((got_output.float() - want_output.float()).abs().max().item()),
        float((got_state.float() - want_state.float()).abs().max().item()),
    )
    output_replay_bad = int((got_output.view(torch.int16) != replay_output.view(torch.int16)).sum().item())
    state_replay_bad = int((got_state.view(torch.int16) != replay_state.view(torch.int16)).sum().item())
    finite = bool(torch.isfinite(got_output.float()).all() and torch.isfinite(got_state.float()).all())

    bad_schema = Problem(2, 1, 65, 1, 2, 128, 16, 2)
    bad_heads = Problem(1, 1, 65, 3, 4, 128, 16, 2)
    negative_schema = int(size(ctypes.byref(bad_schema))) == 0
    negative_heads = int(size(ctypes.byref(bad_heads))) == 0
    insufficient = forward(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
        got_output.data_ptr(), got_state.data_ptr(), workspace.data_ptr(),
        workspace_bytes - 1, ctypes.byref(problem),
        torch.cuda.current_stream(device).cuda_stream,
    ) == 2

    passed = (
        finite
        and metrics.output_rel_l1 < 0.03
        and metrics.state_rel_l1 < 0.03
        and output_replay_bad == 0
        and state_replay_bad == 0
        and negative_schema
        and negative_heads
        and insufficient
    )
    print(
        "[PPU GDN backend] "
        f"{'PASS' if passed else 'FAIL'} shape=B2,S65,Hq1,Hv2,D128,C16,GC2 "
        f"workspace={workspace_bytes} output_rel_l1={metrics.output_rel_l1:.8f} "
        f"state_rel_l1={metrics.state_rel_l1:.8f} "
        f"output_max_abs={metrics.output_max_abs:.8f} "
        f"state_max_abs={metrics.state_max_abs:.8f} "
        f"replay_bad={output_replay_bad}/{state_replay_bad} "
        f"schema_negative={'PASS' if negative_schema else 'FAIL'} "
        f"head_negative={'PASS' if negative_heads else 'FAIL'} "
        f"workspace_negative={'PASS' if insufficient else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
