#!/usr/bin/env python3
"""One warmed public-API forward inside an ACU start/stop range.

This is a counter capture, NOT another latency benchmark. Reuse the comparison
fixture, oracle, tolerance and FLA dispatch/compatibility code without forking
their numerical contract.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch

import bench_ppu_gdn_fla as bench


def loaded_library_paths():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if path.is_file():
                paths.add(path.resolve())
    return paths


class PPUProfiler:
    """Use the already-loaded PPU runtime, never introduce a second SDK/CUDART."""
    def __init__(self):
        wrappers = [p for p in loaded_library_paths() if p.name == "libhggc_wrapper.so"]
        if len(wrappers) != 1:
            raise RuntimeError(f"need one loaded PPU profiler runtime, found {wrappers}")
        self.path = wrappers[0]
        self.lib = ctypes.CDLL(str(self.path))
        for name in ("hggcProfilerStart", "hggcProfilerStop"):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = [], ctypes.c_int

    def invoke(self, name):
        rc = getattr(self.lib, name)()
        if rc:
            raise RuntimeError(f"{name} failed: PPU runtime status={rc}")

    def start(self):
        self.invoke("hggcProfilerStart")

    def stop(self):
        self.invoke("hggcProfilerStop")


def capture_one(call, synchronize, start, stop):
    """Warmup/checks are deliberately the caller's responsibility, outside here."""
    synchronize()
    start()
    try:
        result = call()
        synchronize()
        return result
    finally:
        stop()


def save_fla_sources(destination):
    """Snapshot imported FLA Python sources, not caches, datasets or site-packages."""
    import fla
    package = Path(fla.__file__).resolve().parent
    paths = {Path(module.__file__).resolve() for name, module in tuple(sys.modules.items())
             if (name == "fla" or name.startswith("fla."))
             and getattr(module, "__file__", None)
             and Path(module.__file__).suffix == ".py"}
    from triton.backends.nvidia import compiler
    compiler_source = Path(compiler.__file__).resolve()
    paths.add(compiler_source)
    manifest = []
    for source in sorted(paths):
        if source.is_relative_to(package):
            relative = Path("fla") / source.relative_to(package)
        elif source == compiler_source:
            relative = Path("triton") / "nvidia_compiler.py"
        else:
            # A vendor may alias an external helper into the fla namespace.
            # Preserve it without colliding with the Triton compiler snapshot.
            relative = Path("external-fla") / hashlib.sha256(str(source).encode()).hexdigest()[:16] / source.name
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append(dict(original=str(source), saved=str(relative),
                             sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    return manifest


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("ours", "fla"))
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--gate", type=float, choices=(-0.1, -1.0), default=-0.1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    if not args.extension.is_file() or args.warmup < 1:
        parser.error("require an existing PPU extension and at least one warmup")
    os.environ["GDN_QSA_PPU_EXTENSION"] = str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)  # physical device is fixed by CUDA_VISIBLE_DEVICES
    props = torch.cuda.get_device_properties(0)
    if "PPU" not in props.name.upper():
        raise RuntimeError(f"not a PPU device: {props.name}")

    cpu = bench.admission.fixture(1, 2048, 16, 32, args.gate)
    want = bench.admission.reference(cpu)
    inputs = tuple(x.cuda() for x in cpu)
    input_hash = bench.admission.digest(cpu)
    identity = {}
    if args.role == "ours":
        call = lambda: bench.admission.gdn_chunk(*inputs, output_final_state=True)
    else:
        fn, identity = bench.load_fla()
        call = bench.fla_call(fn, inputs, "native")

    print(f"[PPU GDN ACU config] role={args.role} g={args.gate} "
          f"shape=B1,S2048,Hk16,Hv32,D128 input_sha={input_hash} "
          "initial_state=zero final_state=1 GVA=native forward_only=1", flush=True)
    first = call()  # import/JIT/autotune outside profiler range
    torch.cuda.synchronize()
    errors = bench.checked_pair(first, want)
    fingerprint = bench.admission.digest(first)
    del first
    for _ in range(args.warmup):
        warm = call()
    torch.cuda.synchronize()
    if bench.admission.digest(warm) != fingerprint:
        raise AssertionError("warmup output/state is not bit-stable")
    del warm
    if bench.admission.digest(inputs) != input_hash:
        raise AssertionError("preflight modified fixture inputs")
    print(f"[PPU GDN ACU preflight] role={args.role} output_state_error={errors} "
          f"output_sha={fingerprint} NUMERIC/PASS replay=RAW-BIT/STABLE", flush=True)

    profiler = PPUProfiler()
    print(f"[PPU GDN ACU range] begin role={args.role} public_api_calls=1", flush=True)
    result = capture_one(call, torch.cuda.synchronize, profiler.start, profiler.stop)
    print(f"[PPU GDN ACU range] end role={args.role} public_api_calls=1", flush=True)
    post_errors = bench.checked_pair(result, want)
    if bench.admission.digest(result) != fingerprint:
        raise AssertionError("profiled output/state differs from unprofiled preflight")
    if bench.admission.digest(inputs) != input_hash:
        raise AssertionError("profiled call modified fixture inputs")

    source_manifest = save_fla_sources(args.sources) if args.role == "fla" else []
    # Actual mapped library paths/hashes expose a stale dependency even when the
    # extension filename itself looks current. Do not archive process env vars.
    loaded = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in loaded_library_paths()
              if path.name.startswith(("libhggc", "libcuda", "libgdn", "_gdn_chunk"))}
    receipt = dict(status="PASS", role=args.role, gate=args.gate,
                   shape=dict(B=1, S=2048, Hk=16, Hv=32, K=128, V=128),
                   input_sha=input_hash, reference_sha=bench.admission.digest(want),
                   fixture_seed=0x6A09E667, gate_bf16=float(cpu[3].flatten()[0]),
                   output_sha=fingerprint, errors=errors, post_errors=post_errors,
                   max_relative_error_limit=bench.admission.MAX_RELATIVE_ERROR,
                   public_api_calls=1, warmup=args.warmup,
                   initial_state="zero", output_final_state=True, fla_heads="native",
                   protocol="ACU-full-public-api-single-forward",
                   timing_scope="PROFILED_DIAGNOSTIC_NOT_BENCHMARK",
                   cache_control="all (profiler flush; not benchmark cache state)",
                   torch=torch.__version__, torch_cuda=torch.version.cuda,
                   torch_build_config=torch.__config__.show(),
                   python=sys.version, python_executable=sys.executable,
                   device=dict(name=props.name, cu=props.multi_processor_count,
                               total_memory=props.total_memory,
                               properties=str(props),
                               uuid=str(getattr(props, "uuid", "UNAVAILABLE")),
                               visible=os.environ.get("CUDA_VISIBLE_DEVICES")),
                   extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
                   loaded_libraries=loaded, fla=identity, fla_sources=source_manifest,
                   profiler_api=dict(start="hggcProfilerStart", stop="hggcProfilerStop",
                                     library=str(profiler.path),
                                     sha256=hashlib.sha256(profiler.path.read_bytes()).hexdigest()))
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[PPU GDN ACU] PASS: role={args.role} one API call; receipt={args.receipt}", flush=True)


if __name__ == "__main__":
    main()
