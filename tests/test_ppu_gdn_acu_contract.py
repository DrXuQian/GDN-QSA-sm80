"""Host-only contract/negative tests; no device, ACU or installed FLA required."""
import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
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

    def test_portable_streaming_sha_matches_full_digest(self):
        directory = self.directory()
        path = directory / "non-ascii-证据.bin"
        data = bytes(range(256)) * 5000  # spans the streaming chunk boundary
        path.write_bytes(data)
        self.assertEqual(collect.sha(path), hashlib.sha256(data).hexdigest())

    def records(self):
        ours = dict(status="PASS", role="ours", phase="subject", warmup=0, public_api_calls=1, gate=-0.1,
                    shape=dict(B=1, S=2048), input_sha="input", reference_sha="ref",
                    device=dict(name="PPU", uuid="fixture", cu=72), torch="vendor",
                    torch_cuda="13.0", initial_state="zero", output_final_state=True,
                    fla_heads="native", max_relative_error_limit=0.02, protocol="single-forward",
                    cache_control="all", extension_sha256="binary", library_sha256="device-binary",
                    implementation="original", output_sha="output", fla={},
                    output_dtype="torch.bfloat16", state_dtype="torch.float32")
        fla = copy.deepcopy(ours)
        fla["role"] = "fla"
        return ours, fla

    def test_sdk_acu_outranks_shared_site_and_path(self):
        directory = self.directory()
        sdk = directory / "sdk"
        matched = sdk / "asight/bin/acu"
        matched.parent.mkdir(parents=True)
        matched.write_text("synthetic executable identity only\n")
        matched.chmod(0o755)
        site = Path("/sim/eec/shared/junfu.qx/asight/bin/acu")
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(collect.shutil, "which", return_value=str(directory / "older-acu")), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(collect.os, "access", return_value=True):
            self.assertEqual(collect.find_acu(sdk), matched)
            self.assertNotEqual(collect.find_acu(sdk), site)

    def test_explicit_acu_is_not_silently_replaced(self):
        directory = self.directory()
        absent = directory / "missing-explicit-acu"
        with patch.dict(os.environ, {"ACU": str(absent)}, clear=True):
            self.assertEqual(collect.find_acu(directory / "sdk"), absent.resolve())

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

    def test_native_exception_keeps_failed_receipt_and_is_not_retried(self):
        directory = self.directory()
        receipt = directory / "wy.json"
        calls = []
        thrown = IndexError("map::at")
        def bad():
            calls.append("forward")
            raise thrown
        with patch.object(profile, "loaded_library_hashes", return_value={"/sdk/libparser.so": "hash"}):
            with self.assertRaises(IndexError) as error:
                profile.run_with_failure_receipt(bad, receipt, dict(role="wy", phase="subject"))
        self.assertIs(error.exception, thrown)
        self.assertEqual(calls, ["forward"])
        saved = json.loads(receipt.read_text())
        self.assertEqual(saved["status"], "FAIL")
        self.assertEqual(saved["error"], "map::at")
        self.assertEqual(saved["loaded_libraries"], {"/sdk/libparser.so": "hash"})
        _, fla = self.records()
        with self.assertRaises(ValueError):
            collect.validate_pair(saved, fla, "wy")

    def test_profiler_libraries_join_identity_without_loading_anything(self):
        directory = self.directory()
        paths = {directory / name for name in ("libhgBinaryAnalysis.so.13", "libhggc_injection.so",
                 "libperfworks.so", "_gdn_wy_ppu.so", "unrelated.so")}
        for path in paths:
            path.write_bytes(b"synthetic DSO")
        with patch.object(profile, "loaded_library_paths", return_value=paths):
            actual = profile.loaded_library_hashes()
        self.assertEqual(len(actual), 4)
        self.assertNotIn(str(directory / "unrelated.so"), actual)

    def test_wy_is_not_an_alias_for_original(self):
        import gdn_qsa_sm80
        inputs = (object(),) * 5
        extension = self.root / "_gdn_wy_ppu.so"
        with patch.object(gdn_qsa_sm80, "gdn_chunk_wy", return_value="wy-result") as wy, \
                patch.object(profile.bench.admission, "gdn_chunk", side_effect=AssertionError("wrong API")), \
                patch.dict(os.environ, {}, clear=True):
            call, identity = profile.subject_call("wy", "wy", extension, inputs)
            self.assertEqual(call(), "wy-result")
            wy.assert_called_once_with(*inputs, output_final_state=True)
            self.assertEqual(os.environ["GDN_QSA_WY_EXTENSION"], str(extension.resolve()))
            self.assertNotIn("GDN_QSA_PPU_EXTENSION", os.environ)
            self.assertEqual(identity, {})
            for role, implementation in (("ours", "wy"), ("wy", "original"), ("fla", "unknown")):
                with self.assertRaises(ValueError):
                    profile.subject_call(role, implementation, extension, inputs)

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
        wy = collect.acu_command(Path("/acu"), Path("/report"), Path("/_gdn_wy_ppu.so"),
                                 "wy", -0.1, Path("/bundle"), "wy")
        self.assertEqual(wy[wy.index("--implementation") + 1], "wy")
        self.assertEqual(wy[wy.index("--role") + 1], "wy")

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

    def test_delivery_reaches_subject_and_receipts_cannot_cross_variants(self):
        for delivery in collect.DELIVERIES:
            if delivery == "scalar":
                continue
            cmd = collect.acu_command(Path("/acu"), Path("/report"), Path("/_gdn_wy_ppu.so"),
                                      "wy", -.1, Path("/bundle"), "wy", delivery)
            self.assertEqual(cmd[cmd.index("--wy-delivery") + 1], delivery)
            with patch("gdn_qsa_sm80.gdn_chunk_wy", return_value=("out", "state")) as forward:
                call, _ = profile.subject_call("wy", "wy", Path("/_gdn_wy_ppu.so"), (1, 2, 3, 4, 5), delivery)
                self.assertEqual(call(), ("out", "state"))
                self.assertEqual(forward.call_args.kwargs["delivery"], delivery)
        ours, fla = self.records()
        ours.update(role="wy", implementation="wy", wy_delivery="all")
        fla.update(implementation="wy", wy_delivery="all")
        collect.validate_pair(ours, fla, "wy")
        with self.assertRaisesRegex(ValueError, "delivery"):
            collect.validate_pair(ours, fla | dict(wy_delivery="scalar"), "wy")
        pre = ours | dict(phase="preflight", warmup=5, public_api_calls=6)
        collect.validate_preflight(pre, ours)
        with self.assertRaisesRegex(ValueError, "wy_delivery"):
            collect.validate_preflight(pre | dict(wy_delivery="state"), ours)

    def test_wy_receipt_cannot_use_original_role_or_other_device_library(self):
        ours, fla = self.records()
        wy = ours | dict(role="wy", implementation="wy")
        fla["implementation"] = "wy"
        collect.validate_pair(wy, fla, "wy")
        for plant in (wy | dict(role="ours"), wy | dict(implementation="original"),
                      wy | dict(library_sha256="stale")):
            with self.assertRaises(ValueError):
                collect.validate_pair(plant, fla, "wy")
        with self.assertRaises(ValueError):
            collect.validate_pair(wy, fla)  # default still means ORIGINAL

    def make_wy_run(self, directory):
        build = directory / "build"
        build.mkdir()
        binding, library = build / "_gdn_wy_ppu.cpython312.so", build / "libgdn_wy_ppu.so"
        binding.write_bytes(b"WY binding")
        library.write_bytes(b"WY device library")
        (directory / "sha.txt").write_text("d" * 40 + "\n")
        (directory / "source.diff").write_text("")
        (directory / "binaries.sha256").write_text("".join(
            f"{collect.sha(path)}  {path}\n" for path in (binding, library)))
        comparison = dict(protocol="full-public-api-event-span", cases=[dict(g=-0.1)],
                          binary_sha256={str(binding): collect.sha(binding)})
        (directory / "comparison.json").write_text(json.dumps(comparison))
        return binding, library

    def test_reused_binding_and_device_library_are_both_bound_to_old_run(self):
        directory = self.directory()
        binding, library = self.make_wy_run(directory)
        chosen, _, origin = collect.read_wy_run(directory)
        self.assertEqual(chosen, binding.resolve())
        self.assertEqual(origin["source_sha"], "d" * 40)
        for binary in (binding, library):
            original = binary.read_bytes()
            binary.write_bytes(b"planted replacement")
            with self.subTest(binary=binary), self.assertRaisesRegex(ValueError, "manifest"):
                collect.read_wy_run(directory)
            binary.write_bytes(original)
        record = json.loads((directory / "comparison.json").read_text())
        record["binary_sha256"][str(binding)] = "planted-other-run"
        (directory / "comparison.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "comparison.json"):
            collect.read_wy_run(directory)

    def test_reuse_missing_or_ambiguous_binding_is_red(self):
        directory = self.directory()
        with self.assertRaisesRegex(ValueError, "one reused WY"):
            collect.read_wy_run(directory)
        binding, _ = self.make_wy_run(directory)
        (binding.parent / "_gdn_wy_ppu.old.so").write_bytes(b"ambiguous")
        with self.assertRaisesRegex(ValueError, "one reused WY"):
            collect.read_wy_run(directory)

    def test_comparison_rebinding_rejects_changed_fla_input_or_output(self):
        ours, fla = self.records()
        ours.update(role="wy", implementation="wy")
        fla["implementation"] = "wy"
        ours["device"]["properties"] = "synthetic properties, no UUID in old run"
        comparison = dict(cases=[dict(g=-0.1, input_sha="input", arms={
            "wy": dict(fingerprint="output", state_dtype="torch.float32"),
            "fla": dict(fingerprint="output", state_dtype="torch.float32")})],
            initial_state="zero", final_state=True, qk_norm=False, scale="1/sqrt(128)",
            dtype="bf16", torch="vendor", fla={}, limit=0.02,
            device=ours["device"]["properties"])
        collect.validate_comparison(comparison, ours, fla)
        packed = ours | dict(wy_delivery="all")
        with self.assertRaisesRegex(ValueError, "not admitted"):
            collect.validate_comparison(comparison, packed, fla)
        comparison["delivery_ab"] = True
        with self.assertRaisesRegex(ValueError, "output differs"):
            collect.validate_comparison(comparison, packed, fla)
        comparison["cases"][0]["arms"]["wy-all"] = dict(fingerprint="output", state_dtype="torch.float32")
        collect.validate_comparison(comparison, packed, fla)
        # Identical numerical fingerprints must not let a legacy capture
        # impersonate the new compute-tile kernel family.
        tiled = ours | dict(wy_delivery="tiled-all")
        with self.assertRaisesRegex(ValueError, "output differs"):
            collect.validate_comparison(comparison, tiled, fla)
        comparison["cases"][0]["arms"]["wy-tiled-all"] = dict(fingerprint="output", state_dtype="torch.float32")
        collect.validate_comparison(comparison, tiled, fla)
        # Same numerical answer is not evidence that mask48 was measured:
        # neither all-tiled nor state-only is the new mixed-stage combination.
        pair = ours | dict(wy_delivery="tiled-state-output")
        comparison["cases"][0]["arms"]["wy-tiled-state"] = dict(fingerprint="output", state_dtype="torch.float32")
        with self.assertRaisesRegex(ValueError, "output differs"):
            collect.validate_comparison(comparison, pair, fla)
        comparison["cases"][0]["arms"]["wy-tiled-state-output"] = dict(fingerprint="output", state_dtype="torch.float32")
        collect.validate_comparison(comparison, pair, fla)
        for suffix in ("address", "gates", "both"):
            name = f"tiled-state-output-{suffix}"
            subject = ours | dict(wy_delivery=name)
            with self.assertRaisesRegex(ValueError, "output differs"):
                collect.validate_comparison(comparison, subject, fla)
            comparison["cases"][0]["arms"][f"wy-{name}"] = dict(fingerprint="output", state_dtype="torch.float32")
            collect.validate_comparison(comparison, subject, fla)
        for suffix in ("prepare", "output", "both"):
            name = f"stage-address-{suffix}"
            subject = ours | dict(wy_delivery=name)
            with self.assertRaisesRegex(ValueError, "output differs"):
                collect.validate_comparison(comparison, subject, fla)
            comparison["cases"][0]["arms"][f"wy-{name}"] = dict(fingerprint="output", state_dtype="torch.float32")
            collect.validate_comparison(comparison, subject, fla)
        for suffix in ("shared", "warp"):
            name = f"prepare-rows-{suffix}"
            subject = ours | dict(wy_delivery=name)
            with self.assertRaisesRegex(ValueError, "output differs"):
                collect.validate_comparison(comparison, subject, fla)
            comparison["cases"][0]["arms"][f"wy-{name}"] = dict(fingerprint="output", state_dtype="torch.float32")
            collect.validate_comparison(comparison, subject, fla)
        for role, key, value in (("wy", "input_sha", "other"), ("wy", "output_sha", "other"),
                                  ("fla", "fla", dict(entry_sha256="changed")),
                                  ("wy", "state_dtype", "torch.bfloat16"),
                                  ("fla", "output_sha", "other")):
            with self.subTest(role=role, key=key), self.assertRaises(ValueError):
                collect.validate_comparison(comparison, ours | ({key: value} if role == "wy" else {}),
                                             fla | ({key: value} if role == "fla" else {}))

    def test_complete_reused_wy_capture_never_builds_or_selects_original(self):
        directory = self.directory()
        prior = directory / "preceding"
        prior.mkdir()
        binding, library = self.make_wy_run(prior)
        ours, fla = self.records()
        ours.update(role="wy", implementation="wy", extension_sha256=collect.sha(binding),
                    library_sha256=collect.sha(library),
                    loaded_libraries={str(p.resolve()): collect.sha(p) for p in (binding, library)})
        ours["device"]["properties"] = "synthetic PPU properties"
        fla = ours | dict(role="fla", loaded_libraries={})
        comparison_path = prior / "comparison.json"
        comparison = json.loads(comparison_path.read_text())
        comparison.update(initial_state="zero", final_state=True, qk_norm=False,
                          scale="1/sqrt(128)", dtype="bf16", torch="vendor", fla={}, limit=0.02,
                          device=ours["device"]["properties"])
        comparison["cases"] = [dict(g=-0.1, input_sha="input", arms={
            role: dict(fingerprint="output", state_dtype="torch.float32") for role in ("wy", "fla")})]
        comparison_path.write_text(json.dumps(comparison))
        bundle = directory / "bundle"
        bundle.mkdir()
        args = SimpleNamespace(wy_run=prior, extension=None, sdk=directory,
                               acu=Path(sys.executable), gate=-0.1, device="0")
        commands = []
        def fake_run(command, log, env, **kwargs):
            command = [str(x) for x in command]
            commands.append(command)
            self.assertNotIn("bash", command)
            self.assertNotIn("cmake", command)
            log.write_text("SYNTHETIC TOOL OUTPUT; not device evidence\n")
            if "--role" in command:
                role = command[command.index("--role") + 1]
                phase = command[command.index("--phase") + 1]
                self.assertIn(role, ("wy", "fla"))
                self.assertEqual(command[command.index("--implementation") + 1], "wy")
                receipt = dict(ours if role == "wy" else fla)
                if phase == "preflight":
                    receipt.update(phase=phase, warmup=5, public_api_calls=6)
                Path(command[command.index("--receipt") + 1]).write_text(json.dumps(receipt))
                if "--set" in command:
                    Path(command[command.index("-o") + 1] + ".acurep").write_bytes(b"synthetic report")
            return dict(status="COLLECTED", returncode=0)
        with patch.object(collect, "run", side_effect=fake_run), \
                patch.object(collect.shutil, "which", return_value=None):
            status = collect.collect(args, bundle, {"PATH": ""})
        self.assertEqual(status["status"], "PASS", status)
        self.assertEqual(status["implementation"], "wy")
        self.assertEqual(status["comparison_origin"]["source_sha"], "d" * 40)
        self.assertTrue((bundle / "binaries" / library.name).is_file())
        self.assertTrue((bundle / "preceding-comparison/comparison.json").is_file())
        stages = [cmd[cmd.index("--phase") + 1] for cmd in commands if "--phase" in cmd]
        self.assertEqual(stages, ["preflight", "preflight", "subject", "subject"])

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
