"""Frozen workload authority for the H800 scalar-GDN comparison.

This is a measurement inventory, not a kernel selector. K/V=128 and C=64
are the current algorithm's scope; the other axes must not be conflated.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    length: int
    q_heads: int
    v_heads: int
    fp32_gate: bool = False
    vary_gate: bool = False
    initial: bool = False

    def __post_init__(self):
        if min(self.shape) <= 0 or self.v_heads % self.q_heads:
            raise ValueError("positive B/T/heads and integral GVA required")

    @property
    def shape(self):
        return self.batch, self.length, self.q_heads, self.v_heads

    @property
    def head_ratio(self):
        return self.v_heads // self.q_heads

    @property
    def offsets(self):
        return tuple(i * self.length for i in range(self.batch + 1))

    def receipt(self):
        return dict(asdict(self), key_dim=128, value_dim=128, chunk=64,
                    sequence_offsets=self.offsets, head_ratio=self.head_ratio)


WORKLOADS = (
    Workload("seq512", 1, 512, 16, 32),
    Workload("seq1024", 1, 1024, 16, 32),
    Workload("seq2048", 1, 2048, 16, 32),
    Workload("seq4096", 1, 4096, 16, 32),
    Workload("seq8192", 1, 8192, 16, 32),
    Workload("tail2051", 1, 2051, 16, 32),
    Workload("batch2", 2, 2048, 16, 32),
    Workload("batch4", 4, 2048, 16, 32),
    Workload("heads64-gva4", 1, 2048, 16, 64),
    Workload("heads64-gva2", 1, 2048, 32, 64),
    Workload("heads32-gva1", 1, 2048, 32, 32),
    Workload("heads16", 1, 2048, 8, 16),
    Workload("vary-fp32", 1, 2048, 16, 32, fp32_gate=True, vary_gate=True),
    Workload("initial-vary", 1, 2048, 16, 32, fp32_gate=True,
             vary_gate=True, initial=True),
)
GATES = (-.1, -1.)
FAMILIES = ("flashinfer", "flashqla")
BY_NAME = {row.name: row for row in WORKLOADS}


def validate_inventory(rows):
    """Independent registered denominator; deleting a row cannot shrink it."""
    expected = {
        "seq512", "seq1024", "seq2048", "seq4096", "seq8192", "tail2051",
        "batch2", "batch4", "heads64-gva4", "heads64-gva2", "heads32-gva1",
        "heads16", "vary-fp32", "initial-vary",
    }
    if len(rows) != 14 or {row.name for row in rows} != expected:
        raise ValueError("registered workload denominator is exactly 14")


def validate_offsets(workload, offsets):
    if tuple(offsets) != workload.offsets:
        raise ValueError("sequence boundaries changed; never merge batch streams")


validate_inventory(WORKLOADS)
