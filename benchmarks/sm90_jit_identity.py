"""Bind every live TileLang executable's actual CUDA image, fresh or cached.

A fresh TVM Executable may run an in-memory LLVM/CUDA import tree, with no
mapped executable.so. Inspect the live object used by the adapter; never
substitute a disk-cache glob or recompile it to manufacture an identity.
"""
import hashlib
from pathlib import Path


def cuda_image(packed):
    """CUDA SaveToBytes ends in uint64 length + its exact cuModuleLoadData image."""
    starts = []
    start = packed.find(b"\x7fELF")
    while start >= 0:
        if start >= 8 and len(packed) >= start + 64:
            size = int.from_bytes(packed[start-8:start], "little")
            if (size == len(packed)-start and
                    int.from_bytes(packed[start+18:start+20], "little") == 190):
                starts.append(start)
        start = packed.find(b"\x7fELF", start+4)
    if len(starts) != 1:
        raise RuntimeError("live TileLang image is not a CUDA ELF with an exact serialized extent")
    return packed[starts[0]:]


def tilelang_images(kernels, out, serialize=None):
    if not kernels:
        raise RuntimeError("no live TileLang kernels to bind")
    if serialize is None:
        from tvm.runtime import _ffi_api
        # CPU serialization only: unlike WriteToFile this preserves dtype30
        # (TMA descriptor) in binary metadata rather than failing JSON export.
        serialize = lambda m: _ffi_api.ModulePackImportsToTensor(m).numpy().tobytes()
    output = out / "jit-binaries"
    output.mkdir(exist_ok=True)
    result = []
    for key, kernel in kernels.items():
        executable = kernel.adapter.executable
        # Executable.__call__ uses _jitted_mod when present; otherwise its mod
        # is directly runnable. A disk-cache hit is already a runtime Module.
        active = getattr(executable, "_jitted_mod", None)
        if active is None:
            active = getattr(executable, "mod", executable)
        pending, seen, count = [active], set(), 0
        key_hash = hashlib.sha256(repr(key).encode()).hexdigest()
        while pending:
            module = pending.pop()
            if id(module) in seen:
                continue
            seen.add(id(module))
            pending.extend(module.imports)
            if module.kind != "cuda":
                continue
            path = output / f"tilelang-{key_hash}-{count}.cubin"
            if module.imports:
                raise RuntimeError("unexpected imports in CUDA leaf module")
            packed = serialize(module)
            data = cuda_image(packed)
            path.write_bytes(data)
            result.append(dict(cache_key_sha256=key_hash, path=str(path), bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest(),
                               serialized_sha256=hashlib.sha256(packed).hexdigest(),
                               kind="LIVE_TVM_CUDA_MODULE", executable_kind=active.kind,
                               identity_collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            count += 1
        if count == 0:
            raise RuntimeError(f"TileLang executable has no live CUDA image: {key_hash}")
    return result
