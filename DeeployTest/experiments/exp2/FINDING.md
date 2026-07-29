# exp2 · FINDING — latency & memory of one on-device SpeechNet FT step

**Date:** 2026-07-29
**Branch:** `feat/BNFRozen_OptionB`
**Fixture:** `Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen_b1_fold3`
(full-model FT, **frozen BN stats**, batch 1, n-accum 4)
**Config:** Siracusa / GVSoC, 8 cores, L1=128 KB, L2=2 MB, `--defaultMemLevel L2`,
`--memAllocStrategy MiniMalloc`, `--searchStrategy random-max`, `BN_FROZEN_STATS=ON`.
**Source data:** `profiling_run/probe_*.log` (5 `--profileTiling` passes) merged by
`profiling_run/analyze_and_perfetto.py`; memory from `memory_alloc_{training,optimizer}.html`.
**Cross-check:** device `BENCH train_cycles=140,390,186 opt_cycles=56,427` — the merged per-op sum
(140,169,041) reconciles to within **0.16 %**.

---

## 2a results — measured numbers

### Cycles per training step (= 4 fwd/bwd/accum micro-batches + 1 SGD update)

| Phase | Compute (Kernel) | DMA (L2↔L1) | Total | Share |
|-------|-----------------:|------------:|------:|------:|
| **Forward**   | 67,838,084 | 2,844,332 | **70,682,416** | 50.4 % |
| **Backward**  | 64,462,668 | 4,921,628 | **69,384,296** | 49.5 % |
| **Optimizer** (SGD net, BENCH) | — | — | **56,427** | 0.04 % |
| Gradient-accumulation (22 `InPlaceAccumulator`, in train net) | 17,558 | 30,408 | 47,966 | 0.03 % |
| **TOTAL / step** | **132,343,452** | **7,825,589** | **≈ 140.4 M** | 100 % |

- **Compute-bound: 94.4 % of cycles are Kernel (compute), only 5.6 % are DMA.** More cores / better
  vectorisation help far more than DMA reduction on this workload.
- **Forward ≈ Backward** (~70.7 M vs ~69.4 M). The optimizer (SGD) and gradient accumulation are
  **negligible** (~0.07 % combined) — frozen BN + tiny trainable head/conv weights.
- One training step ≈ **140 M cycles**. At an assumed ~370 MHz Siracusa cluster that is ≈ **0.38 s/step**;
  with n-accum 4 that is one optimizer update every ~0.38 s.

### Top ops by per-step cycles (cluster block = Kernel + L2↔L1 DMA)

| Rank | Cycles | Kernel | DMA | Phase | Op |
|-----:|-------:|-------:|----:|-------|----|
| 1 | 29,618,156 | 29,222,820 | 395,336 | fwd | **block0 Conv** (`node_1..0_0_Conv`) |
| 2 | 24,464,796 | 24,365,560 | 99,236 | fwd | **block1 Conv** (`node_5..1_0_Conv`) |
| 3 | 13,609,948 | 13,545,424 | 64,524 | bwd | block1 **ConvGradX** |
| 4 | 12,591,396 | 11,578,340 | 1,013,056 | bwd | block0 **MaxPoolGrad** |
| 5 | 10,560,112 | 10,429,876 | 130,236 | bwd | block1 **ConvGradW** |
| 6 | 4,891,156 | 4,409,296 | 481,860 | bwd | block0 **BatchNormalizationGrad** |
| 7 | 4,373,008 | 4,190,996 | 182,012 | bwd | block0 **ConvGradW** |
| 8 | 3,817,868 | 3,792,484 | 25,384 | bwd | block2 **ConvGradX** |
| 9 | 3,706,428 | 3,680,988 | 25,440 | fwd | block2 Conv |
| 10 | 3,283,784 | 3,069,852 | 213,932 | fwd | block0 **MaxPool** |

### By op family (per step, cluster blocks)

| Family | Compute | L1 DMA | Total | Share |
|--------|--------:|-------:|------:|------:|
| Conv (forward)            | 64,139,468 | 1,849,116 | 65,988,584 | 47.1 % |
| ConvGradX (input grad)    | 19,858,412 |   121,272 | 19,979,684 | 14.3 % |
| ConvGradW (weight grad)   | 19,057,452 |   448,120 | 19,505,572 | 13.9 % |
| MaxPoolGrad               | 12,356,160 | 1,210,920 | 13,567,080 |  9.7 % |
| Relu (forward)            |  3,925,876 | 1,891,048 |  5,816,924 |  4.1 % |
| BatchNormalizationGrad    |  4,619,020 |   648,232 |  5,267,252 |  3.8 % |
| MaxPool (forward)         |  3,358,236 |   278,152 |  3,636,388 |  2.6 % |
| BatchNormInternal (fwd)   |  2,366,268 |   438,868 |  2,805,136 |  2.0 % |
| ConvGradB                 |  1,749,828 |   217,236 |  1,967,064 |  1.4 % |
| ReluGrad                  |    798,592 |   650,496 |  1,449,088 |  1.0 % |
| sgd / accumulator / Gemm / GAP / loss | ~110 K | ~70 K | ~0.18 M | 0.1 % |

**Takeaway:** **Conv fwd + ConvGrad(X/W) = ~75 %** of all cycles, and they are concentrated in the two
**early, large-spatial blocks (block0, block1)**. Add MaxPoolGrad (9.7 %) and BN-grad (3.8 %) and you have
~88 % of the step. The FC head, loss, GAP, gradient accumulation and the SGD update are rounding error.

### Memory (peak reservation + largest buffers)

| Level | Budget | Training peak | Util | Optimizer peak | Util |
|-------|-------:|--------------:|-----:|---------------:|-----:|
| **L1** | 128,000 B | **127,808 B** | **99.9 %** | 57,344 B | 44.8 % |
| **L2** | 2,000,000 B | **1,793,800 B** | **89.7 %** | 123,912 B | 6.2 % |
| L3 | 64 MB | 0 (unused) | — | 0 | — |

Largest training buffers (from `memory_alloc_training.html`):
- **L1** (binding): block0 `Conv_0_grad_tensor` **117,768 B** (one tile ≈ fills L1); then block1/block2
  conv-grad tiles 71–82 KB. L1 is packed to the last ~192 bytes → tiles are forced small.
- **L2**: block0 large-spatial activations **314,048 B** each. Several (BN/MaxPool forward outputs) have
  **lifetime ≈ 88–93 schedule steps** — i.e. **forward activations stashed across the whole graph for the
  backward pass**. That long-lived 314 KB class is what pushes L2 to ~90 %.

---

## 2b — improvement opportunities (grounded in the measured data)

Ordered by expected payoff. These are observations, not implemented changes.

**A. Attack the early large-spatial Conv/ConvGrad (biggest lever — ~75 % of cycles).**
The two dominant ops (block0/block1 Conv fwd 29.6 M + 24.5 M, plus their ConvGradX/W ~44 M) are
compute-bound (>98 % Kernel). Levers:
- **im2col elimination / direct conv.** These convs use an im2col + transpose path (visible as the
  `*_transpose_L2` sections and the `_tensor_split` scratch buffers). im2col inflates both compute and the
  314 KB L2 scratch. A direct/depthwise-aware conv kernel, or reusing the im2col buffer between fwd and the
  ConvGradW that needs the same lowered activation, would cut the single largest cost.
- **Better vectorisation/core utilisation on the big kernels.** With 94 % of the step in compute and only
  8 cores engaged, the early convs are the place where wider SIMD / im2col-free tiling pays back linearly.
  Worth measuring cycles vs `--cores` and vs tile shape specifically on block0/block1.

**B. Kernel fusion — especially the frozen-BN affine.**
BN forward (`BatchNormInternal`, 2.8 M) is, with **frozen stats**, a pure per-channel `y = a*x + b`
elementwise pass done as its own tiled op after Conv. Folding that affine **into the preceding Conv's bias/
scale** (or fusing Conv→BN→ReLU into one kernel) removes an entire elementwise sweep over the largest
tensors *and* removes its 314 KB-class intermediate from L2 and a DMA round-trip. Same idea for
`BatchNormalizationGrad` (5.3 M) on the backward side. This is the cleanest structural win for the
frozen-BN recipe specifically.

**C. Backward MaxPoolGrad (9.7 %) + activation stash-vs-recompute.**
`MaxPoolGrad` is 13.6 M cycles and also carries the most DMA of any op (1.0 M, because it scatters into a
large-spatial grad map). It needs the forward argmax/activation, which is exactly the long-lived 314 KB L2
buffer. Options: (i) **stash only the argmax indices** instead of the full activation (much smaller
lifetime buffer), or (ii) **recompute** the small forward segment during backward to trade the ~90 %-of-
schedule L2 stash for a little extra compute — attractive because we are *memory*-tight (L1 99.9 %), not
DMA-tight.

**D. Memory-level placement / tiling to relieve L1 (99.9 %) and L2 (90 %).**
L1 being at 99.9 % with a single 118 KB conv-grad tile is forcing small tiles (more kernel invocations,
more DMA rounds — MaxPoolGrad's 1 M DMA is a symptom). Levers: split that grad tile across more, smaller
tiles; or promote selected long-lived L2 activations to L3 (`--promoteToL2`/L3 placement) to free L2
headroom; or shrink the im2col scratch (ties back to A). Because the network is compute-bound, the memory
win matters mainly by *unlocking larger/better-shaped tiles* for the big convs, not for DMA per se.

**E. Cheap/no-op targets — don't bother.**
Gradient accumulation (48 K), the SGD update (56 K), GAP, loss and the FC head are all <0.1 % — no reason
to optimise them. Likewise DMA reduction in aggregate (5.6 %) is low-value except where it unblocks tiling
(item D). The frozen-BN choice is already paying off: BN has no running-stat update work in the step.

**Where the cycles go (one line):** ~75 % early large-spatial Conv fwd+grad (im2col-heavy, compute-bound),
~10 % MaxPoolGrad, ~4 % BN-grad — fwd≈bwd, optimizer≈0.
**Where the memory goes (one line):** L1 saturated by one ~118 KB conv-grad tile; L2 at ~90 % dominated by
314 KB block-0 activations stashed nearly end-to-end for the backward pass.
