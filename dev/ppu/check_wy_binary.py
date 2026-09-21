#!/usr/bin/env python3
"""Fail-closed compile/resource gate; NOT a device numerical verdict."""
import argparse
from pathlib import Path
import re
import subprocess


def audit(isa, resources, symbols):
    funcs = re.findall(r"Func \d+ (\S+) RESOURCE INFO:\n(.*?)(?=Func \d+ \S+ RESOURCE INFO:|\Z)",
                       resources, flags=re.S)
    if len(funcs) != 9:
        raise AssertionError(f"WY image denominator must be 6 controls + 3 tiled, got {len(funcs)}")
    rows = []
    mma_counts = {}
    for role, packed in ((role, packed) for role in ("prepare", "state", "output") for packed in (False, True)):
        matches = [(name, body) for name, body in funcs
                   if f"gdn_wy_{role}ILb{int(packed)}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one WY {role}/{packed} image: {len(matches)}")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack:
            raise AssertionError(f"WY {role} spills: {stack} bytes")
        sections = isa.split("Disassembly of section ")
        own = [x for x in sections if x.splitlines() and name in x.splitlines()[0]]
        if len(own) != 1:
            raise AssertionError(f"missing exact ISA section: {role}")
        for opcode in ("v.mma.f32.bf16.m16n16k16", "tsm.ld.swzl", "vmem.ld.tsm"):
            if opcode not in own[0]:
                raise AssertionError(f"{role} lost {opcode}")
        if role == "prepare" and "v.mma.f32.tf32.m16n16k8" not in own[0]:
            raise AssertionError("FP32 block solve lost its TF32 products")
        if packed:
            if "vmem.st.b32x4" not in own[0]:
                raise AssertionError(f"{role} lost 16-byte vector global stores")
            if "vmem.st.b16" in own[0]:
                raise AssertionError(f"{role} retained scalar BF16 global stores")
            if role == "state" and "\tvmem.ld.b16\t" in own[0]:
                raise AssertionError("state retained scattered scalar BF16 U loads")
        counts = tuple(own[0].count(op) for op in (
            "v.mma.f32.bf16.m16n16k16", "v.mma.f32.tf32.m16n16k8"))
        if not packed:
            mma_counts[role] = counts
        elif counts != mma_counts[role]:
            raise AssertionError(f"{role} delivery change altered static MMA body counts")
        rows.append(dict(role=role, delivery="packed" if packed else "scalar", registers=regs, stack=stack))
    # New tilings deliberately change static loop bodies; do not compare
    # static MMA counts to a differently rolled scalar loop. Exact expected
    # bodies plus l009's per-output reduction order cover the new decomposition.
    for role, bf16, tf32 in (("prepare", 40, 12), ("state", 32, 0), ("output", 40, 0)):
        matches = [(name, body) for name, body in funcs if f"gdn_wy_tiled_{role}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one tiled {role} image: {len(matches)}")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack:
            raise AssertionError(f"tiled {role} spills: {stack} bytes")
        own = [x for x in isa.split("Disassembly of section ")
               if x.splitlines() and name in x.splitlines()[0]]
        if len(own) != 1:
            raise AssertionError(f"missing exact tiled ISA section: {role}")
        body_isa = own[0]
        counts = tuple(body_isa.count(op) for op in (
            "v.mma.f32.bf16.m16n16k16", "v.mma.f32.tf32.m16n16k8"))
        if counts != (bf16, tf32):
            raise AssertionError(f"tiled {role} MMA body differs: {counts} != {(bf16, tf32)}")
        for opcode in ("tsm.ld.swzl", "vmem.ld.tsm", "vmem.st.b32x4"):
            if opcode not in body_isa:
                raise AssertionError(f"tiled {role} lost {opcode}")
        if "vmem.st.b16" in body_isa:
            raise AssertionError(f"tiled {role} scalar BF16 store regression")
        if role == "state" and ("v.shuffle" in body_isa or "\tvmem.ld.b16\t" in body_isa):
            raise AssertionError("tiled state retained register shuffle or scalar U load")
        rows.append(dict(role=role, delivery="tiled", registers=regs, stack=stack))
    for name in ("gdn_wy_forward", "gdn_wy_forward_delivery"):
        if not re.search(rf"\b{name}$", symbols, re.M):
            raise AssertionError(f"WY launcher missing from linked library: {name}")
    for name in ("configure_tiled", "launch_tiled_prepare", "launch_tiled_state", "launch_tiled_output"):
        if not re.search(rf"\b_ZN7gdn_qsa2wy\d+{name}E\S*$", symbols, re.M):
            raise AssertionError(f"tiled cross-TU launcher missing from linked library: {name}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    isa = (args.build / "gdn_wy_ppu.isa").read_text()
    resources = (args.build / "gdn_wy_ppu.resources").read_text()
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(args.build / "libgdn_wy_ppu.so")], text=True)
    for row in audit(isa, resources, symbols):
        print(f"[WY binary] {row} compile-only/PASS")
    if args.self_test:
        for label, texts in (
            ("spill", (isa, resources.replace("STACK SIZE:0", "STACK SIZE:32", 1), symbols)),
            ("missing-tf32", (isa.replace("v.mma.f32.tf32.m16n16k8", "REMOVED"), resources, symbols)),
            ("missing-link", (isa, resources, symbols.replace("gdn_wy_forward", "REMOVED"))),
            ("scalar-store-regression", (isa.replace("vmem.st.b32x4", "vmem.st.b16"), resources, symbols)),
            ("missing-tiled-image", (isa.replace("gdn_wy_tiled_state", "MISSING_tiled_state"), resources, symbols)),
            ("missing-tiled-host-link", (isa, resources, symbols.replace("launch_tiled_state", "MISSING_tiled_state"))),
            ("tiled-shuffle-regression", (isa.replace("tsm.ld.swzl.b32x4.s0.t1.trans1", "v.shuffle.idx.b32"), resources, symbols)),
        ):
            try:
                audit(*texts)
            except AssertionError:
                print(f"[WY binary negative] {label} EXPECTED-RED/PASS")
            else:
                raise AssertionError(f"escaped negative: {label}")
    print("[WY binary] PASS device_execution=NOT_RUN")


if __name__ == "__main__":
    main()
