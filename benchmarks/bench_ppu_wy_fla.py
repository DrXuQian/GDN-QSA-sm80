#!/usr/bin/env python3
"""Same-input original/WY/FLA comparison. No routing promotion or reset retuning."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_ppu_gdn_fla import admission, checked_pair, fla_call, load_fla, verdict
from gdn_qsa_sm80 import gdn_chunk_wy


def order(sample):
    roles = ("original", "wy", "fla")
    rows = (roles, roles[::-1], (roles[1], roles[2], roles[0]),
            (roles[0], roles[2], roles[1]), (roles[2], roles[0], roles[1]),
            (roles[1], roles[0], roles[2]))
    return rows[sample % len(rows)]


def comparison_summary(arms):
    """Keep descriptive medians separate from the unchanged envelope rule."""
    result = dict(ratio_scope="DESCRIPTIVE_MEDIANS_NOT_ADMISSION")
    medians = {role: statistics.median(arm["samples_us"]) for role, arm in arms.items()}
    for control in ("original", "fla"):
        label = verdict(arms["wy"]["samples_us"], arms[control]["samples_us"])
        result[f"wy_vs_{control}"] = label.replace("OURS", "WY").replace("FLA", control.upper())
    result["wy_over_fla"] = medians["wy"] / medians["fla"]
    result["speedup_over_original"] = medians["original"] / medians["wy"]
    return result


@torch.inference_mode()
def compare(fn, gate, args, device):
    cpu = admission.fixture(1, 2048, 16, 32, gate)
    inputs = tuple(x.to(device) for x in cpu)
    want = admission.reference(cpu)
    calls = dict(original=lambda: admission.gdn_chunk(*inputs),
                 wy=lambda: gdn_chunk_wy(*inputs), fla=fla_call(fn, inputs, "native"))
    record = dict(g=gate, shape="B1/S2048/Hk16/Hv32/D128", input_sha=admission.digest(cpu), arms={})
    for role, call in calls.items():
        first = call()
        torch.cuda.synchronize()
        errors = checked_pair(first, want)
        fingerprint = admission.digest(first)
        for _ in range(7):
            if admission.digest(call()) != fingerprint:
                raise AssertionError(f"{role} replay changed")
        record["arms"][role] = dict(errors=errors, fingerprint=fingerprint,
                                   state_dtype=str(first[1].dtype), samples_us=[])
        del first
        for _ in range(args.warmup):
            call()
        print(f"[WY compare admission] g={gate} role={role} errors={errors} "
              "repeat=8/8 NUMERIC/PASS", flush=True)
    for sample in range(args.samples):
        for role in order(sample):
            torch.cuda.synchronize()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.launches):
                result = calls[role]()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000 / args.launches
            if not math.isfinite(us) or us <= 0:
                raise AssertionError("invalid device event timing")
            checked_pair(result, want)
            if admission.digest(result) != record["arms"][role]["fingerprint"]:
                raise AssertionError(f"{role} timed replay changed")
            record["arms"][role]["samples_us"].append(us)
            del result
    if admission.digest(inputs) != record["input_sha"]:
        raise AssertionError("a compared call changed input")
    for role, arm in record["arms"].items():
        times = arm["samples_us"]
        arm["median_us"] = statistics.median(times)
        print(f"[WY compare] g={gate} role={role} median_us={arm['median_us']:.3f} "
              f"range=[{min(times):.3f},{max(times):.3f}] samples_us={times}", flush=True)
    record.update(comparison_summary(record["arms"]))
    for control in ("original", "fla"):
        label = record[f"wy_vs_{control}"]
        print(f"[WY verdict] g={gate} control={control} verdict={label} rule=disjoint-observed-envelopes")
    print(f"[WY ratios] g={gate} WY/FLA={record['wy_over_fla']:.4f} "
          f"original/WY={record['speedup_over_original']:.4f} scope={record['ratio_scope']} "
          f"WY_vs_FLA={record['wy_vs_fla']} routing=UNCHANGED")
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", required=True, type=Path)
    p.add_argument("--wy-extension", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--launches", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    if args.samples < 3 or args.launches < 1 or args.samples * args.launches < 50 or args.warmup < 5:
        p.error("need >=5 warmups, >=3 samples and >=50 timed launches per arm")
    for key, path in (("GDN_QSA_PPU_EXTENSION", args.extension), ("GDN_QSA_WY_EXTENSION", args.wy_extension)):
        if not path.is_file():
            p.error(f"missing {key}: {path}")
        os.environ[key] = str(path.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if "PPU" not in props.name.upper():
        raise RuntimeError(f"not a PPU: {props.name}")
    fn, identity = load_fla()
    result = dict(protocol="full-public-api-event-span", includes_allocation_and_launch_gaps=True,
                  includes_original_host_dispatch_sync=True, initial_state="zero", final_state=True,
                  qk_norm=False, scale="1/sqrt(128)", dtype="bf16", device=str(props),
                  torch=torch.__version__, fla=identity, samples=args.samples, launches=args.launches,
                  warmup=args.warmup, limit=admission.MAX_RELATIVE_ERROR,
                  binary_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                                 for x in (args.extension, args.wy_extension)}, cases=[])
    for gate in (-.1, -1.):
        result["cases"].append(compare(fn, gate, args, torch.device("cuda", args.device)))
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[WY compare] PASS scope=NUMERICS+MEASUREMENT_COMPLETED_NOT_SPEED_ADMISSION "
          f"results={args.results} routing=UNCHANGED")


if __name__ == "__main__":
    main()
