# exp6 — ZO single-step latency: findings

**Date:** 2026-08-20 · **Branches:** TrainDeeploy / Onnx4Deeploy `feat/BP+ZO` · **Platform:** Siracusa (GAP9, 8 cores), GVSoC.
**Run:** single ZO step, `n_accum=1`, `--profileTiling`. **Result: PASSED bit-exact — 0/2 errors**
(`loss+ = 0.016488`, `loss− = 0.069274`, both `diff=0.000000`).
Logs: `logs/runner.log` (full), `logs/profiletiling.log` (extracted `[node][SB]…` + phase/BENCH lines).

> Note on clocks: all numbers are **cycle counts** (frequency-independent). For wall-clock, GAP9 cluster ≈ 370 MHz
> (the GVSoC config clock of 50 MHz does not affect cycle counts). One ZO step ≈ 35.87 M cyc ≈ **97 ms @ 370 MHz**.

---

## 1. Headline — where a ZO step spends its cycles

One ZO step = **two perturbed forwards** (`zo_train`, run for +ε then −ε) + **one in-place update** (`zo_update`).

| Phase | Cycles | Share | Source |
|---|--:|--:|---|
| +ε forward (`zo_train`) | 17,854,345 | 49.8% | `[PHASE] +eps forward DONE` |
| −ε forward (`zo_train`) | 17,857,763 | 49.8% | `[PHASE] -eps forward DONE` |
| update (`zo_update`) | 157,521 | 0.44% | `[PHASE] update DONE` / BENCH `opt_cycles` |
| **Total step** | **35,869,629** | 100% | BENCH `train_cycles` 35,712,108 + `opt_cycles` 157,521 |

The two forwards are **near-identical** (Δ = 3,418 cyc, 0.02%) — as expected, they are the same graph with the
perturbation sign flipped. `--profileTiling` re-runs each node in-place, so its per-node "Total"s captured **both**
forward passes (every forward node shows `tiles=2`); the profiled sum **35,817,330** covers **99.6%** of the step.

### Decomposition by *role* (profiled node totals, whole step)

| Role | Nodes | Cycles | Share of step |
|---|--:|--:|--:|
| **Forward compute** (Conv, BN, ReLU, MaxPool, transposes, Gemm) | 36 | 35,359,338 | **98.6%** |
| **Forward perturbation** (`Perturb…weight/bias`, both passes) | 22 | 307,822 | **0.86%** |
| **Update perturbation** (`Perturb…_updated`, the θ-step) | 22 | 150,170 | **0.42%** |
| **All perturbation** (fwd + update) | 44 | **457,992** | **1.28%** |

**The single most important number for the quantization plan: perturbation is only ~1.3% of a ZO step today.**
The float32 forward compute utterly dominates (98.6%).

---

## 2. What dominates the forward (the 98.6%)

| Node | Cycles | Share of step | What it is |
|---|--:|--:|---|
| `node_1 … blocks_0_0_Conv` | 14,808,671 | **41.3%** | 1st conv on the raw 14×700 EMG window (huge spatial extent) |
| `node_5 … blocks_1_0_Conv` | 12,243,161 | **34.2%** | 2nd conv (largest MAC count, 5.05 M ops/pass) |
| `node_4 … blocks_0_3_MaxPool` | 1,639,112 | 4.6% | block-0 pooling |
| `…blocks_0_0_Conv…pre_transpose` | 1,610,883 | 4.5% | NCHW→NHWC layout copy feeding conv-0 |
| `node_2 … blocks_0_1_BatchNormalization` | 1,271,755 | 3.6% | folded/frozen BN |
| `node_3 … blocks_0_2_Relu` | 360,097 | 1.0% | |
| `node_13 … blocks_3_0_Conv` | 350,253 | 1.0% | |
| others (blocks 2–4 conv/BN/relu/maxpool, transposes, fc) | ~3.4 M | ~9.5% | |

**Two conv layers alone are 75.5% of the entire ZO step.** These are the early, large-spatial float32
convolutions; they are the reason a ZO step costs ~36 M cycles. This is exactly the compute that a **quantized int8
datapath on the N-EUREKA NPU** is built to accelerate.

---

## 3. The perturbation process in detail (the focus for NPU/RTL work)

Perturbation touches all **15,489 trainable parameters** (10 conv weights/biases + BN γ/β + fc), and it happens
**three times per ZO step**: perturb `θ+εz` (forward +ε), perturb `θ−εz` (forward −ε), and apply `θ−lr·g_proj·z`
(update) — i.e. ~46.5 k element-perturbations/step. Kernel: `ApplyRademacherPerturbation` in
`TargetLibraries/PULPOpen/src/RandomNoise.c` (stateless LCG+xorshift32 RNG → packed-bit ±1, `dst[i]=src[i]±ε`,
8-core block-partitioned, out-of-place through L1).

### 3a. Cost structure

| Metric | Value | Reading |
|---|--:|---|
| Update-perturb kernel | 128,615 cyc | the ±εz arithmetic + RNG for all 22 tensors |
| Update-perturb DMA (Pre+Post) | 21,555 cyc (14.4%) | read weight → L1, write back → L2 |
| Fwd-perturb kernel (2 passes) | 264,191 cyc | |
| Fwd-perturb DMA (2 passes) | 43,631 cyc (14.2%) | |
| cyc/element (large tensors, 2–7 k elem) | **6.8–7.2** | memory-bound elementwise float |
| cyc/element (fc, 288 elem) | 10.8 | fixed overhead starting to show |
| cyc/element (tiny, ≤32 elem) | 40+ | fully overhead-dominated |
| **Per-node fixed floor** (≤16-elem tensors) | **~1,500–2,000 cyc** | RNG seed + tiling/DMA setup + launch, *independent of size* |

**Two structural inefficiencies stand out:**

1. **Per-node launch overhead dominates the many small tensors.** 22 tensors × 3 passes = 66 kernel launches;
   with a ~1,500-cyc floor, the update graph alone spends **~33 k of its 150 k cycles (≈22%) on pure per-node
   overhead** (biases and BN γ/β are ≤32 elements yet cost ~1,500–2,000 cyc each — as much as a 32× larger tensor).
2. **The kernel is compiled UNOPTIMIZED.** `RandomNoise.c:9` sets **`#pragma clang optimize off`** for the whole
   file — the perturbation (and all noise) kernels run with optimization disabled. This is the single lowest-effort
   speedup available (just removing the pragma / re-enabling `-O3` + auto-vectorization), and it makes the current
   ~7 cyc/elem an *upper bound*, not a floor.
3. **Memory-bound + out-of-place.** Each perturb reads the full weight and writes a full perturbed copy through L1
   (14% DMA). The forward then reads that copy. On a streaming NPU datapath the perturbation could instead be
   **fused into the weight load** (perturb-on-the-fly as weights stream L2→NPU), eliminating both the extra copy and
   the DMA round-trip.

---

## 4. Why this matters for the quantization step (NPU + RTL preparation)

The whole point of exp6 is to set a baseline **before** quantization. The key insight is a **coming inversion**:

- **Today (float32, cluster):** forward = 98.6%, perturbation = 1.3%. Perturbation is negligible; optimizing it
  would be pointless.
- **After int8 + N-EUREKA NPU:** the two dominant convs (75.5%) and the other convs move onto the NPU and get
  int8 acceleration (typically ~10–50× on N-EUREKA). If the ~35 M forward cycles collapse to ~1–3 M, then the
  perturbation cost (~0.46 M, which stays on the cluster cores) jumps from **1.3% to ~15–30% of the step** — it
  becomes a **first-order bottleneck**.

Two additional effects push the perturbation cost **up** in the quantized path:
- **Quantized perturbation is heavier per element.** The int8 `RQSPerturbRademacher` kernel (see
  `Deeploy/Quantized_ZO.md`) does not just add ±ε — it computes a per-channel requant `noise_q = (±M + rounding) >> S`
  and an int8-saturating add. More ALU work per element than the float `src ± ε`.
- **The forward shrinks but the perturbation element count does not** — the same 15,489 params must be perturbed
  3× regardless of how fast the forward becomes.

**Therefore the concrete optimization targets this data justifies, in priority order:**

1. **Re-enable compiler optimization** on `RandomNoise*.c` (remove `#pragma clang optimize off`) — free, immediate.
2. **Kill per-node launch overhead** — fuse the 22 per-tensor perturbations into one (or a few) batched kernel
   launches; the ~33 k-cycle floor in the update graph is almost pure overhead on tiny BN/bias tensors.
3. **Fuse perturbation into the NPU weight stream** — generate `±εz` (or the quantized `±M>>S`) inline as weights
   are streamed L2→NPU, removing the separate perturbed-weight copy and its DMA round-trip. This is where an
   **RTL-level Rademacher/perturbation unit** pays off: a hardware LCG+xorshift + packed-bit ±1 generator feeding the
   NPU weight port would make the forward perturbation *nearly free*, leaving only the once-per-step update on the
   cluster.
4. **Keep the RNG bit-exactness contract** (LCG `×1664525+1013904223`, xorshift32 `<<13,>>17,<<5`, packed-bit
   LSB-first, per-core `+core_id` chunking) identical between any new SW/HW perturbation path and the Onnx4Deeploy
   reference — otherwise the device-vs-reference loss check breaks (see `Deeploy/Quantized_ZO.md` §8).

---

## 5. Reference numbers (for later comparison)

| Quantity | Value |
|---|--:|
| ZO step total | 35,869,629 cyc (~97 ms @ 370 MHz) |
| forward (both passes) | 35,712,108 cyc |
| update (θ-step) | 157,521 cyc |
| all perturbation | 457,992 cyc (1.28%) |
| — forward perturbation (2×) | 307,822 cyc |
| — update perturbation | 150,170 cyc |
| trainable params | 15,489 |
| top-2 convs (blocks_0_0 + blocks_1_0) | 27,051,832 cyc (75.5%) |
| perturb efficiency (large tensors) | ~7 cyc/elem (unoptimized, memory-bound) |
| perturb per-node fixed floor | ~1,500–2,000 cyc |

*Baseline for the quantized-ZO port. The expectation to test post-quantization: forward collapses onto the NPU,
perturbation becomes the dominant remaining SW cost, and items 1–3 in §4 become the optimization roadmap.*

---

## 6. Figures (ZO analogs of the BP exp1 latency plots — `results/`)

Generated by `results/lat_fig{1,2,6}_*.py` (parse `logs/profiletiling.log`). ZO has **no backward/optimizer**,
so the categories are **forward / perturbation (θ±εz) / update (zo_update)** instead of forward/backward/optimizer,
and the BP "Conv vs ConvGrad" plot becomes **"Conv vs Perturb"** (the extra cost beyond the forward).

- **`lat_fig1_compute_transfer.png`** — one ZO step is **95.8% compute / 4.2% DMA** (heavily compute-bound,
  like BP). BP analog: `exp1_BP_argmax_off/results/lat_fig1_compute_transfer.png`.
- **`lat_fig2_per_operator.png`** — per-operator: **Conv 82.2%**, transpose 6.1%, MaxPool 5.1%, BatchNorm 3.9%,
  ReLU 1.4%, then **Perturb θ±εz 0.86%** and **Perturb update 0.42%**. Perturbation is at the bottom of the chart.
- **`lat_fig6_conv_vs_perturb.png`** (ZO analog of BP `lat_fig6_conv_vs_convgrad.png`) — Conv (forward) vs
  Perturb per block. Perturb/Conv rises **0.1% (B0) → 0.4% → 2.5% → 19.3% → 69.3% (B4)**: perturbation cost
  tracks **weight** size while conv cost tracks **activation** size, so perturbation is negligible in the
  activation-heavy early blocks and becomes comparable to conv in the tiny-activation last block — yet stays
  ~1.3% of the whole step because B0+B1 dominate the cycle budget.

- **`lat_fig2b_per_operator_BP_vs_ZO.png`** — **fused BP-vs-ZO** per-operator (vertical grouped bars, single
  step, n_accum=1). Three regions: **FORWARD** (shared — every forward op is exactly **2×** in ZO, the two ±ε
  passes: Conv 14.72M→29.43M, etc.); **BP-only BACKWARD+optimizer** (ConvGrad 10.36M, MaxPoolGrad 4.25M,
  BNGrad 1.32M, … = 16.4M); **ZO-only PERTURBATION** (Perturb θ±εz 0.31M + update 0.15M = 0.46M). The picture:
  ZO **doubles the forward** but **eliminates the entire 16.4M backward** for a **0.46M** perturbation, netting
  BP 34.1M vs ZO 35.8M (**1.05×** — essentially tied). Source: BP `exp1_BP_argmax_off`, ZO this run.

*Reproduce:* `cd results && python3 lat_fig1_compute_transfer.py && python3 lat_fig2_per_operator.py &&
python3 lat_fig6_conv_vs_perturb.py && python3 lat_fig2b_per_operator_BP_vs_ZO.py` (needs matplotlib; run in
the `traindeeploy` container). The fused plot reads BP from `../../exp1_BP_argmax_off/logs/profiletiling.log`.
