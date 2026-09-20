"""Host-only contract/negative tests; no device, ACU or installed FLA required."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch
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
        ours = dict(status="PASS", role="ours", public_api_calls=1, gate=-0.1,
                    shape=dict(B=1, S=2048), input_sha="input", reference_sha="ref",
                    device=dict(name="PPU", uuid="fixture", cu=72), torch="vendor",
                    torch_cuda="13.0", initial_state="zero", output_final_state=True,
                    fla_heads="native", max_relative_error_limit=0.02, protocol="single-forward",
                    cache_control="all", extension_sha256="binary")
        fla = copy.deepcopy(ours)
        fla["role"] = "fla"
        return ours, fla

    def test_one_call_and_synchronization_are_inside_range_only(self):
        order = ["preflight", "warmup", "correctness"]
        def step(name):
            return lambda: order.append(name)
        profile.capture_one(step("call"), step("sync"), step("start"), step("stop"))
        order.append("postcheck")
        self.assertEqual(order, ["preflight", "warmup", "correctness", "sync",
                                 "start", "call", "sync", "stop", "postcheck"])

    def test_call_failure_stops_range_but_is_not_swallowed(self):
        order = []
        def bad():
            raise RuntimeError("planted launch failure")
        with self.assertRaisesRegex(RuntimeError, "launch failure"):
            profile.capture_one(bad, lambda: order.append("sync"),
                                lambda: order.append("start"), lambda: order.append("stop"))
        self.assertEqual(order, ["sync", "start", "stop"])

    def test_profiler_start_failure_never_runs_subject(self):
        def fail():
            raise RuntimeError("profiler unavailable")
        def must_not_run():
            self.fail("subject/stop should not run without profiler start")
        with self.assertRaisesRegex(RuntimeError, "profiler unavailable"):
            profile.capture_one(must_not_run, lambda: None, fail, must_not_run)

    def test_profiler_runtime_missing_ambiguous_and_error_are_red(self):
        for paths in (set(), {Path("/one/libhggc_wrapper.so"), Path("/two/libhggc_wrapper.so")}):
            with patch.object(profile, "loaded_library_paths", return_value=paths):
                with self.assertRaisesRegex(RuntimeError, "need one loaded"):
                    profile.PPUProfiler()
        profiler = object.__new__(profile.PPUProfiler)
        class BrokenRuntime:
            def hggcProfilerStart(self):
                return 7
        profiler.lib = BrokenRuntime()
        with self.assertRaisesRegex(RuntimeError, "status=7"):
            profiler.start()

    def test_all_kernels_profiled_not_warmup_or_truncated_subset(self):
        cmd = collect.acu_command(Path("/acu"), Path("/report"), Path("/binding.so"),
                                  "ours", -0.1, Path("/bundle"))
        for flag, value in (("--profile-from-start", "no"), ("--set", "full"),
                            ("--kill", "no"), ("--check-exit-code", "yes"), ("--gate", "-0.1")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        for forbidden in ("--launch-count", "--kernel-name", "--disable-profiler-start-stop", "--csv"):
            self.assertNotIn(forbidden, cmd)

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
