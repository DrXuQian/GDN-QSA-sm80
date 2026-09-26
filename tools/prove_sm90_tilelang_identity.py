#!/usr/bin/env python3
"""Real first-JIT/cache-hit proof using the same admitted FlashQLA forward."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--ppu-source", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir()
    env = dict(os.environ, TILELANG_CACHE_DIR=str(a.out / "cache"),
               TILELANG_TMP_DIR=str(a.out / "scratch"), CUDA_HOME="/usr/local/cuda-12.8",
               PATH="/usr/local/cuda-12.8/bin:" + os.environ["PATH"])
    images = []
    for phase in ("fresh", "cached"):
        out = a.out / phase
        with (a.out / f"{phase}.log").open("x") as log:
            subprocess.run([sys.executable, str(ROOT / "benchmarks/profile_sm90_libraries.py"),
                "--family", "flashqla", "--reference-root", str(a.work / "FlashQLA"),
                "--source-archive", str(a.work / "flashqla-source.tar.gz"),
                "--cuda-extension", str(a.control), "--candidate-extension", str(a.candidate),
                "--ppu-source-extension", str(a.ppu_source), "--candidate-raw-bit",
                "--workload", "heads64-gva4", "--gate", "-0.1", "--preflight-only",
                "--out", str(out)], env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=300)
        receipt = json.loads((out / "receipt.json").read_text())
        assert receipt["status"] == "PREFLIGHT_PASS" and not receipt["calls"]
        compiled = (a.out / f"{phase}.log").read_text().count("TileLang begins to compile kernel")
        assert (compiled > 0) if phase == "fresh" else (compiled == 0)
        images.append(sorted(i["sha256"] for i in receipt["reference_binaries"]))
        print(f"[JIT identity] {phase} compiled={compiled} bound_images={len(images[-1])}", flush=True)
    if not images[0] or images[0] != images[1]:
        raise RuntimeError("fresh/cached actual CUDA image inventories differ")
    result = dict(status="PASS", images=images[0], fresh_cache_identity="EXACT",
                  numerical="5_ARMS_CPU+8_REPEATS", timing="NOT_RUN")
    (a.out / "proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
