#!/usr/bin/env python3
"""Real four-body stash/role/retirement gate; static, not a speed claim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def bodies(path):
    result = {}
    for section in path.read_text().split("Function :")[1:]:
        symbol = section.splitlines()[0].strip()
        if "FlatKernelTmaWarpSpecializedKdaFwd" not in symbol:
            continue
        name = subprocess.check_output(["c++filt", symbol], text=True)
        gates = [g for g in ("float", "cutlass::bfloat16_t")
                 if f"(kda::sm90::kernel::Tag)11, {g}>" in name]
        initial = [i for i in ("false", "true")
                   if f"(kda::sm90::kernel::Tag)8, cute::C<{i}>" in name]
        assert len(gates) == len(initial) == 1
        key = (gates[0], initial[0])
        assert key not in result
        result[key] = re.findall(r"/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*", section)
    assert len(result) == 4
    return result


def counts(rows):
    return Counter(re.match(r"(?:@!?\w+\s+)?([A-Z][\w.]*)", row)[1] for row in rows)


def check(candidate, parent, mode):
    assert len(candidate) == len(parent) == 4 and candidate.keys() == parent.keys()
    result = {}
    fixed = ("HGMMA", "HMMA", "WARPGROUP", "LDSM", "STSM", "BAR", "SYNCS", "UTMA")
    for key, rows in candidate.items():
        a, b = counts(rows), counts(parent[key])
        assert {o: n for o, n in a.items() if o.startswith(fixed)} == {
            o: n for o, n in b.items() if o.startswith(fixed)}, (key, "math/protocol changed")
        assert all("gsb0, 0x0" in row for row in rows if "WARPGROUP.DEPBAR" in row)
        # Two non-initial versus four initial-state body clones use O1.
        expected_vectors = 16 if key[1] == "false" else 32
        assert a["STS.128"] - b["STS.128"] == expected_vectors
        assert a["LDS.128"] - b["LDS.128"] == expected_vectors
        regs = sorted(row for row in rows if row.startswith("USETMAXREG."))
        expected = ["USETMAXREG.DEALLOC.CTAPOOL 0x18"]
        expected += (["USETMAXREG.DEALLOC.CTAPOOL 0x68",
                      "USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xc0"] if mode == 1 else
                     ["USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xa8",
                      "USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xa0"])
        assert regs == sorted(expected), (key, regs)
        result[str(key)] = dict(sites=len(rows), extra_vector_stores=expected_vectors,
                               extra_vector_loads=expected_vectors, register_transitions=regs)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("parent", type=Path)
    p.add_argument("--mode", type=int, choices=(1, 2), required=True)
    p.add_argument("--device-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    log = args.device_log.read_text()
    assert "C7512" not in log and "C7510" not in log and "fatal" not in log
    candidate, parent = bodies(args.candidate), bodies(args.parent)
    result = check(candidate, parent, args.mode)
    def negative(plant):
        try:
            check(plant, parent, args.mode)
        except AssertionError:
            return
        raise AssertionError("native negative escaped")
    negative(parent)
    negative(dict(list(candidate.items())[1:]))
    key = next(iter(candidate))
    wrong_wait = [row.replace("gsb0, 0x0", "gsb0, 0x1") for row in candidate[key]]
    negative(candidate | {key: wrong_wait})
    wrong_reg = [row.replace("DEALLOC.CTAPOOL 0x18", "DEALLOC.CTAPOOL 0x10")
                 for row in candidate[key]]
    negative(candidate | {key: wrong_reg})
    data = dict(status="PASS", scope="STATIC_NOT_SPEED", mode=args.mode, bodies=result,
                negatives=4, sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (args.candidate, args.parent, args.device_log)})
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
