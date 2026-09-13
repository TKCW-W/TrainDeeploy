# exp1 latency breakdown — BP, single training step (`--profileTiling`)

SpeechNet, S01/fold3, GAP9/Siracusa, **single‑buffer** tiling, 8 cores. Per‑tile trace with
`Pre‑Kernel` (DMA L2→L1) · `Kernel` (compute, 8 cores) · `Post‑Kernel` (DMA L1→L2).
Command & raw trace: `logs/profiletiling.log` · parser: `results/parse_profiletiling.py` ·
figures: `results/lat_fig1_*.png`, `results/lat_fig2_*.png`.

## Command
```bash
python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/deliverable_bp_train_recompute \
  --n-steps 1 --n-accum 1 --cores 8 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max \
  -D BN_FROZEN_STATS=ON --profileTiling
```
(n_accum 1 = one fwd+bwd; per‑op cycles are n_accum‑invariant, so this is exp1's per‑step compute.)

## Validation
Profiled sum = **34,095,519 cycles**; on‑device BENCH `train_cycles = 34,084,097` → **0.03 % apart**. The
trace accounts for essentially the whole step.

## 1. Compute vs transfer  (fig 1a)
| Phase | cycles | share |
|---|--:|--:|
| **Compute** (kernel, 8 cores) | 32,353,106 | **94.9 %** |
| DMA‑in (Pre‑Kernel, L2→L1) | 989,231 | 2.9 % |
| DMA‑out (Post‑Kernel, L1→L2) | 753,182 | 2.2 % |
| **Transfer total** | 1,742,413 | **5.1 %** |

**The step is 95 % compute‑bound; data movement is only 5 %.** Even though we run *single‑buffer* (DMA and
compute do **not** overlap), the DMA tail is small — so double buffering would recover at most ~5 % wall‑clock,
and it wouldn't fit L1 (already 99.8 % full). Compute is the thing to optimize, not transfer.

## 2. Forward vs Backward vs Optimizer  (fig 1b)
| Phase | cycles | share | compute % |
|---|--:|--:|--:|
| **Forward** | 17,680,731 | 51.9 % | 96 % |
| **Backward** (grad + accum) | 16,360,573 | 48.0 % | 94 % |
| **Optimizer** (SGD) | 54,215 | 0.2 % | 47 % |

### Is backprop 2× the forward?  → **No — backward ≈ 0.93× forward (they're roughly equal).**
The textbook "backward = 2× forward" assumes each layer's backward computes **both** the input gradient dX and
the weight gradient dW (≈ two forward‑convs). Here it doesn't hold, for two concrete reasons:
1. **The first layer skips dX.** Block 0's `ConvGrad` computes only dW (its input is the data — no gradient
   propagates to it). Block 0 is the most expensive conv, so dropping its dX removes the single biggest "2×"
   term. Net: `ConvGrad` = 10.36 M vs forward `Conv` = 14.72 M → **0.70×**, not 2×.
2. **Frozen BN.** BN backward is just the affine gradient (no running‑stat math) — `BNGrad` 1.32 M is cheap.

The one place the backward is *more* expensive than its forward is **MaxPool**: forward `MaxPool` = 0.91 M but
`MaxPoolGrad` = **4.25 M (≈4.7×)** — because this is the **recompute‑from‑X** kernel (it re‑derives the pooling
argmax in the backward). That's the price of the argmax‑off path, and it's the second‑largest single cost in the
whole step. (The argmax‑mask variant, exp3, replaces this with a cheap masked scatter.)

### Why no 2× — per‑block Conv vs ConvGrad  (fig 3)
The "backward = 2× forward" rule is *per layer* (each layer's backward = dX + dW ≈ two forward‑convs). It's a
network‑wide **total** we're comparing, and it's broken because the cost is **front‑loaded** onto the first two
blocks:

| block | fwd Conv | ConvGrad | ratio | reason |
|---|--:|--:|--:|---|
| **B0** | **7.41 M** | 1.46 M | **0.20×** | first layer → `ConvGrad` computes **dW only, no dX** |
| **B1** | **6.12 M** | 6.14 M | **1.00×** | dX+dW, but ≈1× on this shape (tiling efficiency) |
| B2 | 0.92 M | 1.76 M | 1.91× | dX+dW ≈ textbook 2× |
| B3 | 0.18 M | 0.62 M | 3.54× | dX+dW, channel‑heavy → super‑2× |
| B4 | 0.09 M | 0.35 M | 3.91× | dX+dW, channel‑heavy → super‑2× |
| **total** | 14.72 M | 10.33 M | **0.70×** | |

- The rule **does** hold for the deep blocks (B2 ≈ 2×; B3/B4 exceed it because the channel‑heavy backward convs
  tile less efficiently) — but B2–B4 together are <1.2 M, negligible.
- **B0 + B1 = 13.5 M = 92 % of all forward conv**, and they run at **0.20× and 1.0×**. B0 is the decisive one:
  it carries the largest forward cost yet its backward is nearly free because the **first layer skips dX** (no
  earlier layer to receive an input gradient), leaving only a tiny 1→8‑channel `dW`.
- Net: convs 0.70×, whole step 0.93× — **the first‑layer dX‑skip is the single biggest reason backprop is not 2×
  here.**

## 3. Per‑operator ranking  (fig 2)
| op | phase | cycles | share | compute % |
|---|---|--:|--:|--:|
| Conv | fwd | 14.72 M | 43.2 % | 99 % |
| ConvGrad | bwd | 10.36 M | 30.4 % | 98 % |
| **MaxPoolGrad** | bwd | **4.25 M** | **12.5 %** | 89 % |
| BNGrad | bwd | 1.32 M | 3.9 % | 88 % |
| transpose (fwd layout) | fwd | 1.09 M | 3.2 % | 74 % |
| MaxPool | fwd | 0.91 M | 2.7 % | 92 % |
| BatchNorm | fwd | 0.71 M | 2.1 % | 84 % |
| ReluGrad | bwd | 0.36 M | 1.1 % | 55 % |
| ReLU | fwd | 0.24 M | 0.7 % | 55 % |
| SGD (optimizer) | opt | 0.05 M | 0.2 % | 47 % |
| GradAccum | bwd | 0.05 M | 0.1 % | 37 % |

**Takeaways:** Conv (fwd) + ConvGrad (bwd) = **74 %** of the step — the convolutions dominate and are ~99 %
compute. The two clear optimization targets are the **im2col/Conv kernels** and the **MaxPool recompute
backward** (12.5 %). Small pointwise ops (ReLU, transpose, SGD) are more transfer‑bound (~55–74 % compute) but
negligible in absolute terms.

### Ratio = FLOP‑factor × efficiency‑factor  (exact, fig 4)
The 2× rule is only the FLOP factor; cycles also carry an efficiency factor `(bwd cyc/MAC)/(fwd cyc/MAC)`:

| blk | fwd cyc/MAC | bwd cyc/MAC | FLOP× | eff× | = ratio |
|---|--:|--:|--:|--:|--:|
| B0 | 23.61 | 4.65 | 1× (dX skip) | 0.20 | 0.20× |
| B1 | 2.45 | 1.23 | 2× | 0.50 | 1.00× |
| B2 | 1.53 | 1.46 | 2× | 0.95 | 1.91× |
| B3 | 0.70 | 1.24 | 2× | 1.77 | 3.54× |
| B4 | 0.18 | 0.35 | 2× | 1.96 | 3.91× |

Forward efficiency swings **130×** (23.6→0.18 cyc/MAC); backward stays flat (0.35–4.65). **B0/B1 forward is
inefficient** (few in‑channels, huge spatial → tiny reduction depth, im2col/output‑write bound) → ratio < 2×.
**B3/B4 forward is efficient** (small spatial, deep [7,1] channel reduction) but their backward tiles poorly →
ratio > 2×. **B2** is balanced → textbook 1.9×.

## Effective FLOP/s and FPU utilization
```
conv MACs/step (fwd+bwd) = 12,174,848 → FLOPs = 24.3 MFLOP
wall = 34,084,097 cyc / 370 MHz = 92.1 ms
FLOP/s = 24.3 M / 0.0921 s = 0.264 GFLOP/s
fp32 peak = 8 cores × 2 FLOP/cyc × 370 MHz = 5.92 GFLOP/s → utilization = 4.5 %
```
"95 % compute‑bound" ≠ efficient: effective FPU utilization is only **4.5 %** (B0 alone is 23.6 cyc/MAC). The
real latency lever is **kernel efficiency** (im2col/GEMM, the B0 shape), not DMA or double buffering.

## Benchmark counters (this run, n_accum 1)
```
BENCH train_cycles = 34,084,097   (1 fwd+bwd)   opt_cycles = 61,480   weight_sram = 61,956 B
profiled sum = 34,095,519  → 0.03 % agreement
```

## One fine‑tuning round (exp4 config: 40 epochs, 54 windows, n_accum 4 → 540 device steps)
```
per fwd+bwd (n_accum 1)       = 34.08 M cyc
per device step = 4×fb + opt  = 136.40 M cyc
round = 540 steps             = 73.65 G cycles
  @ 370 MHz → 199.1 s = 3.32 min
  @ 240 MHz → 306.9 s = 5.11 min
```
Matches exp4's ~75.8 G / 3.4 min within ~3 %. **Report single‑step BENCH × step count** — the on‑device round
counter is uint32 and overflows ~17× on a full BP round.
