#!/usr/bin/env python3
"""Collect original/FLA or explicit WY/FLA ACU reports in an uploadable tar.gz.

No remote commands, installs, device clock changes, model data or full caches.
Failed captures still get a clearly INCOMPLETE diagnostic archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
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


def child_command(extension, role, gate, bundle, phase, implementation="original", delivery="scalar"):
    receipt = f"{role}-preflight.json" if phase == "preflight" else f"{role}.json"
    return [sys.executable, "-u", str(ROOT / "benchmarks/profile_ppu_gdn_fla.py"),
            "--extension", str(extension), "--role", role, "--phase", phase, "--gate", str(gate),
            "--implementation", implementation,
            "--wy-delivery", delivery,
            "--receipt", str(bundle / receipt),
            "--sources", str(bundle / "sources/reference")]


def acu_command(acu, report, extension, role, gate, bundle, implementation="original", delivery="scalar"):
    # Same direct CLI pattern as quactlize/tools/run_dense_marlin_m8_acu_box.sh:
    # preflight is a separate process; ACU owns profiling from process start.
    return [str(acu), "-f", "-o", str(report), "--set", "full",
            "--check-exit-code", "yes",
            *child_command(extension, role, gate, bundle, "subject", implementation, delivery)]


def read_wy_run(directory):
    """Bind a reused WY pair to the completed comparison, not today's checkout."""
    directory = directory.resolve()
    candidates = sorted((directory / "build").glob("_gdn_wy_ppu*.so"))
    if len(candidates) != 1:
        raise ValueError(f"expected one reused WY binding, found {candidates}")
    extension = candidates[0].resolve()
    library = extension.parent / "libgdn_wy_ppu.so"
    comparison = json.loads((directory / "comparison.json").read_text())
    if (comparison.get("protocol") != "full-public-api-event-span"
            or not comparison.get("cases")):
        raise ValueError("WY run lacks the original complete-API comparison")
    source_sha = (directory / "sha.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("WY run has no complete source SHA")
    recorded = []
    for line in (directory / "binaries.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        recorded.append((Path(filename.lstrip("*")).name, digest))
    binary_hashes = {}
    for path in (extension, library):
        matches = [digest for name, digest in recorded if name == path.name]
        if matches != [sha(path)]:
            raise ValueError(f"reused binary differs from comparison manifest: {path.name}")
        binary_hashes[str(path)] = matches[0]
    matches = [digest for name, digest in comparison.get("binary_sha256", {}).items()
               if Path(name).name == extension.name]
    if matches != [sha(extension)]:
        raise ValueError("WY binding is not the one recorded by comparison.json")
    return extension, comparison, dict(source_sha=source_sha, binary_sha256=binary_hashes,
        source_diff_sha256=sha(directory / "source.diff"),
        source_diff_empty=(directory / "source.diff").stat().st_size == 0,
        comparison_sha256=sha(directory / "comparison.json"),
        binary_manifest_sha256=sha(directory / "binaries.sha256"))


def validate_comparison(comparison, subject, fla):
    """Do not rebind an old median to changed inputs, FLA code or arithmetic."""
    cases = [case for case in comparison["cases"] if case.get("g") == subject["gate"]]
    if len(cases) != 1 or cases[0].get("input_sha") != subject["input_sha"]:
        raise ValueError("capture fixture differs from the selected WY comparison case")
    for key, value in (("initial_state", "zero"), ("final_state", True),
                       ("qk_norm", False), ("scale", "1/sqrt(128)"), ("dtype", "bf16")):
        if comparison.get(key) != value:
            raise ValueError(f"unsupported preceding comparison contract: {key}")
    for key in ("torch", "fla"):
        if key not in comparison or comparison[key] != fla.get(key):
            raise ValueError(f"capture differs from the preceding comparison: {key}")
    if comparison.get("limit") != subject["max_relative_error_limit"]:
        raise ValueError("comparison and capture use different numerical limits")
    # The older comparison stores properties but no device UUID. Preserve that
    # limitation: matching properties do NOT establish cross-run physical identity.
    if comparison.get("device") != subject["device"].get("properties"):
        raise ValueError("capture device properties differ from the comparison")
    delivery = subject.get("wy_delivery", "scalar")
    role_name = "wy" if delivery == "scalar" else f"wy-{delivery}"
    if delivery != "scalar" and not comparison.get("delivery_ab"):
        raise ValueError("selected delivery was not admitted in the preceding comparison")
    for role, record in ((role_name, subject), ("fla", fla)):
        arm = cases[0].get("arms", {}).get(role, {})
        if arm.get("fingerprint") != record["output_sha"]:
            raise ValueError(f"{role}: output differs from the compared binary/input")
        if arm.get("state_dtype") != record.get("state_dtype") or not record.get("state_dtype"):
            raise ValueError(f"{role}: final-state precision differs from the comparison")


def validate_preflight(preflight, subject):
    if preflight.get("wy_delivery", "scalar") != subject.get("wy_delivery", "scalar"):
        raise ValueError("subject differs from independent preflight: wy_delivery")
    if (preflight.get("status") != "PASS" or preflight.get("phase") != "preflight"
            or preflight.get("warmup", 0) < 1
            or preflight.get("public_api_calls") != preflight["warmup"] + 1):
        raise ValueError("missing successful independent preflight")
    if subject.get("phase") != "subject" or subject.get("public_api_calls") != 1 or subject.get("warmup") != 0:
        raise ValueError("subject must contain exactly one API call and no warmup")
    for key in ("role", "gate", "shape", "input_sha", "reference_sha", "output_sha", "device",
                "torch", "torch_cuda", "fla", "extension_sha256", "library_sha256", "implementation", "protocol",
                "output_dtype", "state_dtype"):
        if key not in preflight or preflight[key] != subject.get(key):
            raise ValueError(f"subject differs from independent preflight: {key}")


def validate_pair(ours, fla, implementation="original"):
    if ours.get("wy_delivery", "scalar") != fla.get("wy_delivery", "scalar"):
        raise ValueError("capture pair delivery labels differ")
    if implementation not in ("original", "wy"):
        raise ValueError(f"unknown implementation: {implementation}")
    for record, role in ((ours, "wy" if implementation == "wy" else "ours"), (fla, "fla")):
        if (record.get("status") != "PASS" or record.get("role") != role
                or record.get("implementation") != implementation
                or record.get("phase") != "subject" or record.get("warmup") != 0
                or record.get("public_api_calls") != 1):
            raise ValueError(f"{role}: missing successful single-call capture receipt")
        for key in ("input_sha", "reference_sha", "extension_sha256", "library_sha256"):
            if not record.get(key):
                raise ValueError(f"{role}: empty {key}")
    for key in ("gate", "shape", "input_sha", "reference_sha", "device", "torch", "torch_cuda",
                "initial_state", "output_final_state", "fla_heads", "max_relative_error_limit",
                "protocol", "cache_control", "extension_sha256", "library_sha256"):
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
    # The site installation can predate the compiler/runtime by a release.
    # Prefer the profiler shipped with the explicitly selected SDK. Never
    # retry a failed profiler with another version inside the same experiment.
    candidates = [sdk / "asight/bin/acu"]
    on_path = shutil.which("acu")
    if on_path:
        candidates.append(Path(on_path))
    candidates.append(Path("/sim/eec/shared/junfu.qx/asight/bin/acu"))
    return next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), candidates[0])


def collect(args, bundle, env):
    implementation = "wy" if args.wy_run else "original"
    delivery = getattr(args, "wy_delivery", "scalar")
    roles = ("wy" if implementation == "wy" else "ours", "fla")
    status = dict(status="INCOMPLETE", errors=[], probes={}, implementation=implementation, wy_delivery=delivery)
    try:
        for name, command in (
            ("git-head", ["git", "rev-parse", "HEAD"]),
            ("git-status", ["git", "status", "--porcelain=v1"]),
            ("git-diff", ["git", "diff", "HEAD", "--binary"]),
            ("submodules", ["git", "submodule", "status", "--recursive"]),
        ):
            run(command, bundle / f"{name}.txt", env, console=False)
        for directory in ("csrc/gdn_chunk", "include/gdn_qsa", "gdn_qsa_sm80", "cmake",
                          "benchmarks", "tests", "tools", "scripts", "dev/ppu"):
            for path in (ROOT / directory).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".hpp", ".cpp", ".h", ".sh", ".cmake"):
                    copy_file(path, bundle / "sources/ours" / path.relative_to(ROOT))
        for name in ("CMakeLists.txt", ".gitmodules", "docs/PPU_BACKEND.md"):
            copy_file(ROOT / name, bundle / "sources/ours" / name)
        copy_file(ROOT / "docs/PPU_GDN_ACU.md", bundle / "README.md")
        if env.get("FLA_ROOT") and not (Path(env["FLA_ROOT"]) / "fla/__init__.py").is_file():
            raise RuntimeError(f"FLA_ROOT is not an FLA checkout: {env['FLA_ROOT']}")
        if not args.acu.is_file() or not os.access(args.acu, os.X_OK):
            raise RuntimeError(f"ACU unavailable: {args.acu}; set ACU=/path/to/acu")
        status["acu_identity"] = dict(path=str(args.acu), sha256=sha(args.acu),
            sdk_bundled=args.acu.resolve() == (args.sdk / "asight/bin/acu").resolve())
        print(f"[GDN ACU tool] sdk={args.sdk} acu={args.acu} "
              f"sdk_bundled={int(status['acu_identity']['sdk_bundled'])}", flush=True)
        status["probes"]["acu-version"] = run([args.acu, "--version"], bundle / "acu-version.txt", env)
        for name, command in (("acu-help", [args.acu, "--help"]),
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
        comparison = None
        if args.wy_run:
            extension, comparison, origin = read_wy_run(args.wy_run)
            status["binary_source_binding"] = "reused-comparison; current sources are capture helpers, not binary origin"
            status["comparison_origin"] = origin
            status["prior_device_identity"] = "properties-only; previous comparison did not record a UUID"
            for name in ("sha.txt", "source.diff", "binaries.sha256", "comparison.json", "comparison.log", "wy-correctness.log"):
                if (args.wy_run / name).is_file():
                    copy_file(args.wy_run / name, bundle / "preceding-comparison" / name)
        elif extension is None:
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
        library = extension.parent / ("libgdn_wy_ppu.so" if implementation == "wy" else "libgdn_qsa_ppu.so")
        arch_contract = extension.parent / "gdn_hgcc_arch.txt"
        if arch_contract.is_file():
            copy_file(arch_contract, bundle / arch_contract.name)
        for binary in (extension, library):
            if not binary.is_file():
                raise RuntimeError(f"PPU binary missing: {binary}")
            copy_file(binary, bundle / "binaries" / binary.name)
        status["binaries"] = (origin["binary_sha256"] if args.wy_run else
                              {str(p): sha(p) for p in (extension, library)})
        for path, expected in status["binaries"].items():
            if sha(path) != expected or sha(bundle / "binaries" / Path(path).name) != expected:
                raise ValueError(f"binary changed while preparing capture: {path}")
        for option, filename in (("--dump-resource-usage=all", "resources.txt"), ("--dump-isa", "isa.txt")):
            status["probes"][filename] = run(
                [args.sdk / "bin/hgobjdump", "--arch=ppu1.0", option, library],
                bundle / filename, env, optional=True, console=False, timeout=60)

        for role in roles:
            try:
                run(child_command(extension, role, args.gate, bundle, "preflight", implementation, delivery),
                    bundle / f"{role}-preflight.log", env)
            except Exception as exc:
                status["errors"].append(f"{role} preflight: {type(exc).__name__}: {exc}")
        if status["errors"]:
            return status
        if comparison is not None:
            preflights = [json.loads((bundle / f"{role}-preflight.json").read_text()) for role in roles]
            validate_comparison(comparison, *preflights)

        for role in roles:
            try:
                base = bundle / f"{role}-g{args.gate}.report"
                command = acu_command(args.acu, base, extension, role, args.gate, bundle, implementation, delivery)
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
        ours, fla = (json.loads((bundle / f"{role}.json").read_text()) for role in roles)
        for subject in (ours, fla):
            preflight = json.loads((bundle / f"{subject['role']}-preflight.json").read_text())
            validate_preflight(preflight, subject)
        validate_pair(ours, fla, implementation)
        if comparison is not None:
            validate_comparison(comparison, ours, fla)
        validate_loaded_binary(ours, extension, library)
        if (ours["extension_sha256"] != status["binaries"][str(extension)]
                or ours["library_sha256"] != status["binaries"][str(library)]):
            raise ValueError("profiled binding/library differs from archived binary")
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
    parser.add_argument("--wy-run", type=Path,
                        help="reuse this completed WY comparison directory; never compile")
    parser.add_argument("--wy-delivery", choices=("scalar", "prepare", "state", "output", "all"), default="scalar")
    args = parser.parse_args()
    if args.wy_run and args.extension:
        parser.error("--wy-run and --extension/EXTENSION are mutually exclusive")
    if args.wy_delivery != "scalar" and not args.wy_run:
        parser.error("--wy-delivery requires --wy-run with admitted comparison samples")
    if args.wy_run:
        args.wy_run = args.wy_run.resolve()
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
