"""CPU fixture and explicit public-ABI adapters, outside the measured region.

No reference math lives here. The independent recurrent oracle and its fixed
2% gate remain in the existing reference/test modules.
"""
import torch


def make_inputs(workload, gate, fixture):
    cpu = fixture(*workload.shape, gate)
    if workload.fp32_gate:
        cpu = (*cpu[:3], cpu[3].float(), cpu[4])
    if workload.vary_gate:
        values = gate * (.25 + .75 * torch.rand(cpu[3].shape, device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(101)))
        cpu = (*cpu[:3], values.to(cpu[3].dtype), cpu[4])
    initial = None
    if workload.initial:
        initial = .005 * torch.randn(workload.batch, workload.v_heads, 128, 128,
            generator=torch.Generator(device="cpu").manual_seed(17), device="cpu")
    return cpu, initial


def expand_reference_heads(cpu, workload):
    q, k = (x.repeat_interleave(workload.head_ratio, 2) for x in cpu[:2])
    validate_expanded_heads((q, k), workload)
    return q, k


def validate_expanded_heads(pair, workload):
    expected = (workload.batch, workload.length, workload.v_heads, 128)
    if any(tuple(x.shape) != expected for x in pair):
        raise ValueError("GVA head mapping does not cover the public value heads")


def flatten_tokens(tensor, workload, heads):
    if tuple(tensor.shape) != (workload.batch, workload.length, heads, 128):
        raise ValueError("unexpected fixed-batch token layout")
    return tensor.reshape(workload.batch * workload.length, heads, 128)


def flatten_gate(tensor, workload):
    if tuple(tensor.shape) != (workload.batch, workload.length, workload.v_heads):
        raise ValueError("unexpected fixed-batch gate layout")
    return tensor.reshape(workload.batch * workload.length, workload.v_heads)


def flashinfer_initial(initial, workload):
    if initial is None:
        return None
    if tuple(initial.shape) != (workload.batch, workload.v_heads, 128, 128):
        raise ValueError("unexpected KV initial-state layout")
    return initial.transpose(-2, -1).contiguous()


def flashinfer_outputs(pair, workload):
    output, final = pair
    if tuple(output.shape) != (workload.batch * workload.length, workload.v_heads, 128):
        raise ValueError("unexpected packed output layout")
    if tuple(final.shape) != (workload.batch, workload.v_heads, 128, 128):
        raise ValueError("unexpected VK final-state layout")
    return (output.reshape(workload.batch, workload.length, workload.v_heads, 128),
            final.transpose(-2, -1).contiguous())
