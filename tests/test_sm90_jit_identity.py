"""Fresh/cached live-module identity, including incomplete-denominator plants."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from sm90_jit_identity import tilelang_images


class MemoryPath:
    files = {}
    def __init__(self, name="evidence"):
        self.name = name
    def __truediv__(self, name):
        return MemoryPath(self.name + "/" + name)
    def __str__(self):
        return self.name
    def mkdir(self, **kwargs):
        pass
    def read_bytes(self):
        return self.files[self.name]


class Module:
    def __init__(self, kind, children=(), machine=190):
        self.kind, self.imports, self.machine = kind, list(children), machine
    def get_write_formats(self):
        return ["cubin"]
    def write_to_file(self, path, fmt):
        assert self.kind == "cuda" and fmt == "cubin"
        MemoryPath.files[path] = b"\x7fELF" + bytes(14) + self.machine.to_bytes(2, "little") + bytes(44)


def kernel(executable):
    return SimpleNamespace(adapter=SimpleNamespace(executable=executable))


class Identity(unittest.TestCase):
    def test_fresh_and_cached_same_live_image(self):
        device = Module("cuda")
        fresh = kernel(SimpleNamespace(mod=Module("llvm", [device]), _jitted_mod=None))
        cached = kernel(Module("library", [device]))
        a = tilelang_images({"fresh": fresh}, MemoryPath())
        b = tilelang_images({"cached": cached}, MemoryPath())
        self.assertEqual(a[0]["sha256"], b[0]["sha256"])
        self.assertEqual(a[0]["executable_kind"], "llvm")

    def test_uses_jitted_module_not_unexecuted_host_source(self):
        active = Module("library", [Module("cuda")])
        subject = kernel(SimpleNamespace(mod=Module("c"), _jitted_mod=active))
        self.assertEqual(len(tilelang_images({"actual": subject}, MemoryPath())), 1)

    def test_host_only_and_partial_denominator_are_red(self):
        good = kernel(Module("library", [Module("cuda")]))
        bad = kernel(Module("llvm"))
        for kernels in ({"bad": bad}, {"good": good, "bad": bad}):
            with self.assertRaisesRegex(RuntimeError, "no live CUDA"):
                tilelang_images(kernels, MemoryPath())

    def test_wrong_machine_and_missing_cache_are_red(self):
        with self.assertRaisesRegex(RuntimeError, "not a CUDA ELF"):
            tilelang_images({"wrong": kernel(Module("cuda", machine=62))}, MemoryPath())
        with self.assertRaisesRegex(RuntimeError, "no live TileLang"):
            tilelang_images({}, MemoryPath())


if __name__ == "__main__":
    unittest.main()
