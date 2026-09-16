# exp20 — what does turning OFF `-ffast-math` cost on device?

**Opened / closed:** 2026-09-16 · **Branch:** `feat/GAP9_w_NE16` · **Findings:** [`Findings.md`](./Findings.md)

---

## 1. Question

The PULP build sets `-ffast-math` globally (`cmake/pulp/pulp.cmake:12`). Under it clang fuses mul+add
into `fmadd.s` and reassociates, so the device's fp32 result is not the C source order and a host
reference cannot match it bit-for-bit. `exp13`/`exp14` established the fix — `DEEPLOY_STRICT_FP32`
plus `DEEPLOY_STRICT_FP32_FILES` — and used it to get a full round to `0 / 21600` errors.

What was never measured is the **price**. This experiment measures it.

## 2. Design

Two sub-experiments, identical except for the compile flags:

| | `ffast_default` | `strict_fp32` |
|---|---|---|
| flags | the default build | `-D DEEPLOY_STRICT_FP32=ON DEEPLOY_STRICT_FP32_FILES="BatchNorm.c;Gemm.c"` |
| effect | `-ffast-math` everywhere | `-fno-fast-math -ffp-contract=off` on the generated `TrainingNetwork.c`/`OptimizerNetwork.c` **and** on `BatchNorm.c` / `Gemm.c` |

Both: **one update step, `n_accum = 1`** (so 2 forwards + 1 optimizer pass — the smallest unit that
exercises the whole graph), `--profileTiling` for per-kernel cycles, 8 cores, `--l1 128000 --l2
2000000`, `BN_FROZEN_STATS=ON`, QZO recipe (eps 0.01, lr 1e-5, q 1, seed 42).

**Fixture reuse.** Both arms run `speechnet_qzo19_train` / `_update` — exp19's fixture, on the
`artifacts_reference` weights with pooled@99.99 scales. A fresh export would have cost ~3 h and
changed nothing: the flags act at compile time, not on the graph.

**Separate build directories** (`PYTEST_XDIST_WORKER=ffast` / `strict`), so the second arm cannot
reuse the first's objects — the entire point is that they are compiled differently.

## 3. Why `BatchNorm.c` and `Gemm.c`

Not arbitrary: `exp14`'s micro-probes counted the fusions clang actually emits — 13 in `BatchNorm.c`,
154 in `Gemm.c` — and found that only `-fno-fast-math` **and** `-ffp-contract=off` together give
fused = 0. Those two are the fp32 kernels the QZO graph leans on; everything else in it is int8.

## 4. What this measures, and what it does not

**Does:** the cycle cost of strict fp32 on the code paths that actually needed it for bit-exactness.

**Does not:** the cost of disabling `-ffast-math` *globally*. That flag is still on for every other
translation unit. A global switch would cost more — but it is not what the bit-exactness fix needs,
so it is not the number a deployment decision turns on.
