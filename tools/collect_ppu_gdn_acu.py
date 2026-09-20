#!/usr/bin/env python3
"""Collect ours/FLA weak-decay ACU reports and a bounded, uploadable tar.gz.

No remote commands, installs, device clock changes, model data or full caches.
Failed captures still get a clearly INCOMPLETE diagnostic archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, log, env, *, optional=False, console=True, timeout=None):
    """Save the precise argv and tool output; optional probes never masquerade as PASS."""
    command = [str(x) for x in command]
    log.with_suffix(log.suffix + ".command").write_text(shlex.join(command) + "\n")
    print(f"[GDN ACU bundle] {shlex.join(command)}", flush=True)
    with log.open("w") as stream:
        try:
            if timeout is not None:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout, check=False)
                rc = result.returncode
            else:
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as proc:
                    for line in proc.stdout:
                        stream.write(line)
                        stream.flush()
                        if console:
                            print(line, end="", flush=True)
                    rc = proc.wait()
        except (OSError, subprocess.TimeoutExpired) as exc:
            stream.write(f"UNAVAILABLE: {type(exc).__name__}: {exc}\n")
            if not optional:
                raise
            return dict(status="UNAVAILABLE", reason=str(exc))
        if rc:
            stream.write(f"\n{'UNAVAILABLE' if optional else 'FAIL'}: command returncode={rc}\n")
            if not optional:
                raise RuntimeError(f"command failed (rc={rc}); see {log}")
            return dict(status="UNAVAILABLE", returncode=rc)
    return dict(status="COLLECTED", returncode=0)


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy contents, never archive a link back into the operator's filesystem.
    shutil.copyfile(source, destination)


def report_file(base):
    candidates = [p for p in (base, Path(str(base) + ".acurep"))
                  if p.is_file() and p.stat().st_size > 0]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one nonempty ACU report, found {candidates}")
    return candidates[0]


def acu_command(acu, report, extension, role, gate, bundle):
    # Verified against PPU SDK 2.1.1 acu --help. Capture all kernels in ONE API
    # call, not just the recurrence; no launch-count shortcut or kernel filter.
    return [str(acu), "-f", "-o", str(report), "--set", "full",
            "--profile-from-start", "no", "--replay-mode", "kernel",
            "--cache-control", "all", "--clock-control", "none",
            "--kill", "no", "--check-exit-code", "yes",
            sys.executable, "-u", str(ROOT / "benchmarks/profile_ppu_gdn_fla.py"),
            "--extension", str(extension), "--role", role, "--gate", str(gate),
            "--receipt", str(bundle / f"{role}.json"),
            "--sources", str(bundle / "sources/reference")]


def validate_pair(ours, fla):
    for record, role in ((ours, "ours"), (fla, "fla")):
        if record.get("status") != "PASS" or record.get("role") != role or record.get("public_api_calls") != 1:
            raise ValueError(f"{role}: missing successful single-call capture receipt")
        for key in ("input_sha", "reference_sha", "extension_sha256"):
            if not record.get(key):
                raise ValueError(f"{role}: empty {key}")
    for key in ("gate", "shape", "input_sha", "reference_sha", "device", "torch", "torch_cuda",
                "initial_state", "output_final_state", "fla_heads", "max_relative_error_limit",
                "protocol", "cache_control", "extension_sha256"):
        if key not in ours or ours[key] != fla.get(key):
            raise ValueError(f"profile arms differ or lack identity: {key}")


def validate_loaded_binary(receipt, extension, library):
    loaded = receipt.get("loaded_libraries", {})
    for binary in (extension, library):
        matches = [digest for path, digest in loaded.items() if Path(path).name == binary.name]
        if matches != [sha(binary)]:
            raise ValueError(f"loaded binary differs from archive or is absent: {binary.name}")


def pack(bundle, status):
    (bundle / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in bundle.rglob("*")):
        raise ValueError("bundle must not contain symlinks")
    checksums = bundle / "SHA256SUMS"
    checksums.write_text("".join(f"{sha(p)}  {p.relative_to(bundle).as_posix()}\n"
                                 for p in files if p != checksums))
    archive = bundle.parent / f"{bundle.parent.name}.tar.gz"
    with archive.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
            tar.add(path, arcname=f"{bundle.parent.name}/{path.relative_to(bundle)}", recursive=False)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{sha(archive)}  {archive.name}\n")
    print(f"[GDN ACU bundle] {status['status']}; UPLOAD={archive} bytes={archive.stat().st_size}", flush=True)
    return archive


def find_acu(sdk):
    if os.environ.get("ACU"):
        return Path(os.environ["ACU"]).resolve()
    candidates = [Path("/sim/eec/shared/junfu.qx/asight/bin/acu"), sdk / "asight/bin/acu"]
    on_path = shutil.which("acu")
    if on_path:
        candidates.append(Path(on_path))
    return next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), candidates[0])


def collect(args, bundle, env):
    status = dict(status="INCOMPLETE", errors=[], probes={})
    try:
        for name, command in (
            ("git-head", ["git", "rev-parse", "HEAD"]),
            ("git-status", ["git", "status", "--porcelain=v1"]),
            ("git-diff", ["git", "diff", "HEAD", "--binary"]),
            ("submodules", ["git", "submodule", "status", "--recursive"]),
        ):
            run(command, bundle / f"{name}.txt", env, console=False)
        for directory in ("csrc/gdn_chunk", "include/gdn_qsa/ppu", "gdn_qsa_sm80",
                          "benchmarks", "tests", "tools", "scripts", "dev/ppu"):
            for path in (ROOT / directory).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".hpp", ".cpp", ".h", ".sh"):
                    copy_file(path, bundle / "sources/ours" / path.relative_to(ROOT))
        for name in ("CMakeLists.txt", ".gitmodules", "docs/PPU_BACKEND.md"):
            copy_file(ROOT / name, bundle / "sources/ours" / name)
        copy_file(ROOT / "docs/PPU_GDN_ACU.md", bundle / "README.md")
        if env.get("FLA_ROOT") and not (Path(env["FLA_ROOT"]) / "fla/__init__.py").is_file():
            raise RuntimeError(f"FLA_ROOT is not an FLA checkout: {env['FLA_ROOT']}")
        if not args.acu.is_file() or not os.access(args.acu, os.X_OK):
            raise RuntimeError(f"ACU unavailable: {args.acu}; set ACU=/path/to/acu")
        for name, command in (("acu-version", [args.acu, "--version"]),
                              ("acu-help", [args.acu, "--help"]),
                              ("hgcc-version", [args.sdk / "bin/hgcc", "--version"]),
                              ("host", ["uname", "-a"])):
            status["probes"][name] = run(command, bundle / f"{name}.txt", env,
                                          optional=True, console=False, timeout=20)
        smi = shutil.which("ppu-smi", path=env["PATH"])
        if smi:
            status["probes"]["device-before"] = run([smi, "-i", args.device, "-q"],
                bundle / "device-before.txt", env, optional=True, console=False, timeout=20)
        else:
            status["probes"]["ppu-smi"] = dict(status="UNAVAILABLE", reason="not on PATH")

        extension = args.extension
        if extension is None:
            build = bundle.parent / "build"
            build_env = env | {"BUILD_DIR": str(build), "PPU_SDK": str(args.sdk)}
            run(["bash", ROOT / "scripts/build_ppu.sh"], bundle / "build.log", build_env)
            candidates = list(build.glob("_gdn_chunk_ppu*.so"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one built binding; found {candidates}")
            extension = candidates[0].resolve()
            status["binary_source_binding"] = "built-in-this-run (git head/diff and sources included)"
        else:
            status["binary_source_binding"] = "operator-supplied; current source not asserted as binary origin"
        library = extension.parent / "libgdn_qsa_ppu.so"
        for binary in (extension, library):
            if not binary.is_file():
                raise RuntimeError(f"PPU binary missing: {binary}")
            copy_file(binary, bundle / "binaries" / binary.name)
        status["binaries"] = {str(p): sha(p) for p in (extension, library)}
        for option, filename in (("--dump-resource-usage=all", "resources.txt"), ("--dump-isa", "isa.txt")):
            status["probes"][filename] = run(
                [args.sdk / "bin/hgobjdump", "--arch=ppu1.0", option, library],
                bundle / filename, env, optional=True, console=False, timeout=60)

        for role in ("ours", "fla"):
            try:
                base = bundle / f"{role}-g{args.gate}.report"
                command = acu_command(args.acu, base, extension, role, args.gate, bundle)
                run(command, bundle / f"{role}-acu.log", env)
                report = report_file(base)
                # Native reports remain the authority. Text exports make the tar
                # inspectable without the GUI; no CSV/copy-paste required.
                for page in ("details", "raw"):
                    status["probes"][f"{role}-{page}"] = run(
                        [args.acu, "--import", report, "--page", page],
                        bundle / f"{role}-{page}.txt", env,
                        optional=True, console=False, timeout=120)
            except Exception as exc:
                status["errors"].append(f"{role}: {type(exc).__name__}: {exc}")
                print(f"[GDN ACU bundle] FAIL: {status['errors'][-1]}", flush=True)
        if smi:
            status["probes"]["device-after"] = run([smi, "-i", args.device, "-q"],
                bundle / "device-after.txt", env, optional=True, console=False, timeout=20)
        if status["errors"]:
            return status
        ours, fla = (json.loads((bundle / f"{role}.json").read_text()) for role in ("ours", "fla"))
        validate_pair(ours, fla)
        validate_loaded_binary(ours, extension, library)
        if ours["extension_sha256"] != status["binaries"][str(extension)]:
            raise ValueError("profiled extension differs from archived binary")
        for path, expected in status["binaries"].items():
            if sha(path) != expected:
                raise ValueError(f"binary changed during capture: {path}")
        status["status"] = "PASS"
    except Exception as exc:
        status["errors"].append(f"{type(exc).__name__}: {exc}")
        (bundle / "failure.txt").write_text(traceback.format_exc())
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=float, choices=(-0.1, -1.0), default=-0.1)
    parser.add_argument("--extension", type=Path, default=os.environ.get("EXTENSION"))
    args = parser.parse_args()
    args.sdk = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK")).resolve()
    args.acu = find_acu(args.sdk)
    args.device = os.environ.get("DEVICE", "0")
    if not args.device.isdigit():
        parser.error("DEVICE must be one physical PPU's nonnegative integer index")
    if args.extension:
        args.extension = args.extension.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(os.environ.get("OUT", f"/workspace/gdn-qsa-acu-{stamp}-{os.getpid()}"))
    # Own a new directory; no stale reports, destructive cleanup, or mktemp.
    out.mkdir(parents=True, exist_ok=False)
    bundle = out.resolve() / "bundle"
    bundle.mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.device, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               PPU_SDK=str(args.sdk),
               PATH=f"{args.sdk}/bin:" + os.environ.get("PATH", ""),
               LD_LIBRARY_PATH=f"{args.sdk}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
               PYTHONPATH=":".join(filter(None, [str(ROOT), os.environ.get("FLA_ROOT"),
                                                os.environ.get("PYTHONPATH")])))
    status = collect(args, bundle, env)
    status.update(created_utc=stamp, gate=args.gate, physical_device=args.device,
                  sdk=str(args.sdk), acu=str(args.acu),
                  scope="counters only; profiled time is not a new performance verdict")
    pack(bundle, status)
    if status["errors"]:
        print("\n".join(status["errors"]), file=sys.stderr)
    return 0 if status["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
