"""Bind every live TileLang executable's actual CUDA image, fresh or cached.

A fresh TVM Executable may run an in-memory LLVM/CUDA import tree, with no
mapped executable.so. Inspect the live object used by the adapter; never
substitute a disk-cache glob or recompile it to manufacture an identity.
"""
import hashlib
from pathlib import Path


def tilelang_images(kernels, out):
    if not kernels:
        raise RuntimeError("no live TileLang kernels to bind")
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
            # This pinned TVM's CUDA module does not override GetWriteFormats
            # (it reports []). WriteToFile itself asserts fmt == stored fmt_,
            # and saves the same data_ passed to cuModuleLoadData. Do not rely
            # on the empty capability-list method or compile a replacement.
            module.write_to_file(str(path), "cubin")
            data = path.read_bytes()
            if len(data) < 20 or data[:4] != b"\x7fELF" or int.from_bytes(data[18:20], "little") != 190:
                raise RuntimeError("live TileLang image is not a CUDA ELF")
            result.append(dict(cache_key_sha256=key_hash, path=str(path), bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest(),
                               kind="LIVE_TVM_CUDA_MODULE", executable_kind=active.kind,
                               identity_collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            count += 1
        if count == 0:
            raise RuntimeError(f"TileLang executable has no live CUDA image: {key_hash}")
    return result
