"""Host-only contract/negative tests; no device, ACU or installed FLA required."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import profile_ppu_gdn_fla as profile

spec = importlib.util.spec_from_file_location("collect_acu", ROOT / "tools/collect_ppu_gdn_acu.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


class ACUContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small synthetic artifacts are retained, entirely under /workspace.
        cls.root = Path("/workspace") / f"gdn-acu-contract-{uuid4().hex}"
        cls.root.mkdir()

    def directory(self):
        directory = self.root / self._testMethodName
        directory.mkdir()
        return directory

    def records(self):
        ours = dict(status="PASS", role="ours", phase="subject", warmup=0, public_api_calls=1, gate=-0.1,
                    shape=dict(B=1, S=2048), input_sha="input", reference_sha="ref",
                    device=dict(name="PPU", uuid="fixture", cu=72), torch="vendor",
                    torch_cuda="13.0", initial_state="zero", output_final_state=True,
                    fla_heads="native", max_relative_error_limit=0.02, protocol="single-forward",
                    cache_control="all", extension_sha256="binary", output_sha="output", fla={})
        fla = copy.deepcopy(ours)
        fla["role"] = "fla"
        return ours, fla

    def test_direct_subject_calls_once_without_profiler_hooks(self):
        order = []
        def step(name):
            return lambda: order.append(name)
        profile.capture_one(step("call"), step("sync"))
        self.assertEqual(order, ["sync", "call", "sync"])

    def test_subject_call_failure_is_not_swallowed(self):
        order = []
        def bad():
            raise RuntimeError("planted launch failure")
        with self.assertRaisesRegex(RuntimeError, "launch failure"):
            profile.capture_one(bad, lambda: order.append("sync"))
        self.assertEqual(order, ["sync"])

    def test_no_profiler_library_or_api_dependency_remains(self):
        source = (ROOT / "benchmarks/profile_ppu_gdn_fla.py").read_text()
        for forbidden in ("PPUProfiler", "ProfilerStart", "ProfilerStop", "ctypes", "cuda.profiler"):
            self.assertNotIn(forbidden, source)

    def test_only_preflight_warms_up_and_comparison_runs_on_cpu(self):
        import torch
        want = (torch.tensor([1.0]), torch.tensor([2.0]))
        copies, calls = [], []
        class DeviceResult:
            def __init__(self, cpu):
                self.value = cpu
            def detach(self):
                return self
            def cpu(self):
                copies.append("D2H")
                return self.value
            def float(self):
                raise AssertionError("must not launch device verification casts")
        def call():
            calls.append("forward")
            return tuple(DeviceResult(x) for x in want)
        subject = profile.run_phase(call, lambda: None, want, "subject", 5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(copies, ["D2H", "D2H"])
        self.assertEqual((subject["public_api_calls"], subject["warmup"]), (1, 0))
        calls.clear()
        preflight = profile.run_phase(call, lambda: None, want, "preflight", 5)
        self.assertEqual(len(calls), 6)
        self.assertEqual(preflight["output_sha"], subject["output_sha"])
        with self.assertRaises(ValueError):
            profile.run_phase(call, lambda: None, want, "unknown", 5)

    def test_preflight_and_subject_must_match(self):
        subject, _ = self.records()
        preflight = subject | dict(phase="preflight", warmup=5, public_api_calls=6)
        collect.validate_preflight(preflight, subject)
        for key, value in (("output_sha", "changed"), ("device", {}), ("input_sha", "changed"),
                           ("warmup", 5), ("phase", "preflight"), ("public_api_calls", 6)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                collect.validate_preflight(preflight, subject | {key: value})
        with self.assertRaises(ValueError):
            collect.validate_preflight({}, subject)

    def test_all_kernels_profiled_not_warmup_or_truncated_subset(self):
        cmd = collect.acu_command(Path("/acu"), Path("/report"), Path("/binding.so"),
                                  "ours", -0.1, Path("/bundle"))
        self.assertEqual(cmd[:6], ["/acu", "-f", "-o", "/report", "--set", "full"])
        for flag, value in (("--set", "full"), ("--phase", "subject"),
                            ("--check-exit-code", "yes"), ("--gate", "-0.1")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        for forbidden in ("--profile-from-start", "--launch-count", "--kernel-name",
                          "--disable-profiler-start-stop", "--csv"):
            self.assertNotIn(forbidden, cmd)
        preflight = collect.child_command(Path("/binding.so"), "ours", -0.1, Path("/bundle"), "preflight")
        self.assertEqual(preflight[preflight.index("--phase") + 1], "preflight")
        self.assertNotIn("/acu", preflight)

    def test_changed_input_device_or_missing_receipt_cannot_pass(self):
        ours, fla = self.records()
        collect.validate_pair(ours, fla)
        for key, value in (("input_sha", "different"), ("device", dict(cu=32)),
                           ("gate", -1.0), ("public_api_calls", 2), ("extension_sha256", "stale"),
                           ("output_final_state", False), ("status", "FAIL"), ("role", "ours")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                collect.validate_pair(ours, fla | {key: value})
        with self.assertRaises(ValueError):
            collect.validate_pair(ours, {})
        with self.assertRaises(ValueError):
            collect.validate_pair(ours | dict(input_sha=""), fla | dict(input_sha=""))

    def test_report_extension_variants_missing_empty_ambiguous(self):
        directory = self.directory()
        for suffix in ("", ".acurep"):
            base = directory / ("exact.report" if not suffix else "appended.report")
            report = Path(str(base) + suffix)
            report.write_bytes(b"synthetic report")
            self.assertEqual(collect.report_file(base), report)
        for base in (directory / "absent", directory / "empty", directory / "duplicate"):
            if base.name == "empty":
                base.touch()
            if base.name == "duplicate":
                base.write_bytes(b"one")
                Path(str(base) + ".acurep").write_bytes(b"two")
            with self.subTest(base=base), self.assertRaises(RuntimeError):
                collect.report_file(base)

    def test_loaded_device_library_stale_is_red(self):
        directory = self.directory()
        binding, library = directory / "_gdn_chunk_ppu.so", directory / "libgdn_qsa_ppu.so"
        binding.write_bytes(b"binding")
        library.write_bytes(b"device")
        loaded = {str(path): collect.sha(path) for path in (binding, library)}
        collect.validate_loaded_binary(dict(loaded_libraries=loaded), binding, library)
        for plant in (loaded | {str(library): "stale"}, {str(binding): collect.sha(binding)}, {}):
            with self.assertRaises(ValueError):
                collect.validate_loaded_binary(dict(loaded_libraries=plant), binding, library)

    def test_command_failure_not_mistaken_for_optional_probe(self):
        directory = self.directory()
        with self.assertRaisesRegex(RuntimeError, "rc=7"):
            collect.run([sys.executable, "-c", "raise SystemExit(7)"], directory / "required.log",
                        os.environ.copy(), console=False)
        optional = collect.run([sys.executable, "-c", "raise SystemExit(7)"], directory / "optional.log",
                               os.environ.copy(), console=False, optional=True)
        self.assertEqual(optional["status"], "UNAVAILABLE")

    def test_tar_contents_and_checksums_include_failed_capture(self):
        directory = self.directory()
        bundle = directory / "bundle"
        bundle.mkdir()
        (bundle / "ours.report.acurep").write_bytes(b"synthetic native report")
        (bundle / "fla-acu.log").write_text("planted FLA failure\n")
        archive = collect.pack(bundle, dict(status="INCOMPLETE", errors=["FLA failed"]))
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            self.assertTrue(all(not n.startswith("/") and ".." not in Path(n).parts for n in names))
            status = json.load(tar.extractfile(f"{directory.name}/STATUS.json"))
            self.assertEqual(status["status"], "INCOMPLETE")
            sums = tar.extractfile(f"{directory.name}/SHA256SUMS").read().decode().splitlines()
            for row in sums:
                expected, name = row.split("  ", 1)
                self.assertEqual(expected, collect.sha(bundle / name))
            self.assertIn(f"{directory.name}/ours.report.acurep", names)

    def test_archive_cannot_follow_source_links(self):
        directory = self.directory()
        bundle = directory / "bundle"
        bundle.mkdir()
        outside = directory / "not-for-upload"
        outside.write_text("synthetic private file")
        (bundle / "planted-link").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            collect.pack(bundle, dict(status="INCOMPLETE"))


if __name__ == "__main__":
    unittest.main()
