# PPU original-structure port

updated-at: 2026-09-20 12:12:57 UTC
working-on: original-structure PPU port committed; ready for box numerical admission and S2048 timing
blocked-on: no PPU; local SDK runtime requires newer GLIBC/GLIBCXX, so local runtime import is unavailable
last-commit: 1af3d5c64ff91ffb0b0e9afd5acd38d822f6fbdd (original-structure implementation)

Local checkpoint: six original device TUs and original gdn_ops.cu compile/link
with hgcc ppu_10. All 305 original control expressions and host dispatch body
match; four structure negatives are red. Delivery/fragment proof and full
six-product Neumann vs forward-substitution pass with four mapping negatives.
The superseded scalar-gather implementation and its dead C ABI were removed.
Runner now checks the original public API on reset / Blelloch / Hillis-Steele /
serial, then measures B1,S2048,Hk16,Hv32,D128 for strong and weak decay.

Default C16 kernels: zero stack. Original experimental C32 recurrence: 144-byte
stack, not on the production auto-dispatch path; must remain explicitly
unadmitted as a PPU performance configuration.

Fresh full local run PASS: /workspace/gdn-qsa-original-port/full-local.log.
2/2 compiled host tests; 55 algebra cases; four mapping and four preservation
negatives; six device TUs plus original host wrapper linked; 15 kernel images.
Device correctness, performance, and runtime import remain NOT RUN; no device
results are represented by these host/source/codegen gates.
