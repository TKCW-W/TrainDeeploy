# Latency derivation — every number, from first principles (exp1 BP, single step)

Defense‑level walk‑through of the compute/transfer split, the forward‑vs‑backward ratio, cycles‑per‑MAC,
effective FLOP/s, and the round time. All numbers come from one profiled step:
`logs/profiletiling.log` (raw trace), reproduced by `results/kernel_split.py`, `results/flops_and_round.py`.

---

## 0. Terminology (define before deriving)

- **Cycle** — one clock tick of the compute cluster. Wall‑time = cycles ÷ frequency. GAP9 cluster @ **370 MHz**
  (⇒ 1 cycle = 2.70 ns). Everything below is in cycles; seconds only at the very end.
- **MAC = Multiply‑Accumulate** — one fused operation `acc ← acc + a·b` (one multiply **and** one add). The
  atomic unit of work in a convolution / matmul. On each PULP core the FPU issues **1 fp32 FMA per cycle**
  (throughput). With **8 cores**, the cluster's peak is **8 MAC/cycle**.
- **FLOP** — one floating‑point operation. **1 MAC = 2 FLOP** (a multiply + an add). So peak =
  8 MAC/cyc × 2 = **16 FLOP/cycle** = 16 × 370 MHz = **5.92 GFLOP/s** (fp32).
- **cycles‑per‑MAC (cyc/MAC)** — measured cycles ÷ MACs performed. It is the *inverse of efficiency*.
  Theoretical floor = 1 ÷ 8 MAC/cyc = **0.125 cyc/MAC** (all 8 FPUs doing a useful MAC every cycle).
  Anything above 0.125 is overhead: im2col, address arithmetic, loop control, loads/stores, non‑MAC
  instructions, pipeline stalls, imperfect tiling. `utilization = 0.125 / (cyc/MAC)`.
- **Kernel / DMA phases** (from `--profileTiling`, printed per tile):
  `Pre‑Kernel` = DMA **in** (L2→L1), `Kernel` = the 8 cores computing on L1 data, `Post‑Kernel` = DMA **out**
  (L1→L2). `Total = Pre + Kernel + Post`. A tiled op prints one line set per tile; we **sum over tiles** to get
  the op's cycles. Convs are ~98–99 % `Kernel`, so for them Total ≈ Kernel.
- **This run = 1 forward + 1 backward, `--n-accum 1`.** Per‑op cycles are n_accum‑invariant (identical
  schedule/shapes each micro‑batch), so this equals exp1's per‑step compute.

Example raw line (block‑0 forward Conv, one tile):
```
[node_1_blocks_0_blocks_0_0_Conv_Conv_L2][SB][8 ops][Tile 0] Kernel : 462... cycles
```
`[SB]` = single‑buffer. We sum `Pre/Kernel/Post` across that node's tiles → the node's Total.

---

## 1. Convolution geometry (needed for MAC counts)

Model = 5 blocks, each `Conv → BatchNorm → ReLU → MaxPool`. Conv params (from `ft_cfg.json`), input
`[1, 1, 14, 700]` (1 channel, H=14 electrodes, W=700 time samples), `same` padding (conv preserves H,W;
MaxPool divides W):

| blk | kernel (kh×kw) | in_ch→out_ch | pool | Conv output (ch, H, W) |
|---|---|---|---|---|
| B0 | 1×4 | 1→8 | 1×8 | 8, 14, 700 |
| B1 | 1×16 | 8→16 | 1×4 | 16, 14, 87 |
| B2 | 1×8 | 16→16 | 1×4 | 16, 14, 21 |
| B3 | 7×1 | 16→32 | 1×1 | 32, 14, 5 |
| B4 | 7×1 | 32→32 | 1×1 | 32, 14, 5 |

W propagation: 700 →(pool8) 87 →(pool4) 21 →(pool4) 5 →(pool1) 5 →(pool1) 5. **Cross‑check:** the measured L2
tensor sizes match these shapes exactly, e.g. B0 output = 8·14·700·4 B = 313,600 B = 306.25 KB (trace showed
306.69 KB); B0 pooled = 8·14·87·4 = 38,976 B = 38 KB (trace 38.06 KB). ✓

---

## 2. Forward MAC count (formula + arithmetic)

Each output element is a sum of `in_ch·kh·kw` products; there are `out_ch·H_out·W_out` output elements:
```
MAC_fwd = out_ch · H_out · W_out · (in_ch · kh · kw)
```
| blk | out_ch·H·W | ·(in_ch·kh·kw) | **MAC_fwd** |
|---|---|---|--:|
| B0 | 8·14·700 = 78,400 | ·(1·1·4)=4 | **313,600** |
| B1 | 16·14·87 = 19,488 | ·(8·1·16)=128 | **2,494,464** |
| B2 | 16·14·21 = 4,704 | ·(16·1·8)=128 | **602,112** |
| B3 | 32·14·5 = 2,240 | ·(16·7·1)=112 | **250,880** |
| B4 | 32·14·5 = 2,240 | ·(32·7·1)=224 | **501,760** |

---

## 3. Backward MAC count — and the exact "2×" in FLOPs

`ConvGrad` produces three tensors; the trace confirms three separate kernels **ConvGradX / ConvGradW /
ConvGradB**:

- **dW** (weight gradient) `∂L/∂W[o,i,kh,kw] = Σ_{h,w} X[i,·]·dY[o,·]`. One MAC per (weight × output position):
  `MAC_dW = (out_ch·in_ch·kh·kw)·(H_out·W_out) = MAC_fwd` — **identical to forward.**
- **dX** (input gradient) `∂L/∂X = ` full/transposed conv of dY with W:
  `MAC_dX = (in_ch·H_in·W_in)·(out_ch·kh·kw)`. With `same` padding H_in=H_out, W_in=W_out ⇒ `MAC_dX = MAC_fwd` —
  **also identical to forward.**
- **dB** (bias gradient) = a reduction of dY over space — additions, not MACs → negligible.

⇒ **In MACs, `ConvGrad = MAC_dX + MAC_dW = 2 · MAC_fwd` exactly** (for any interior layer). **This is where the
textbook "backward = 2× forward" comes from, and it is provably exact in FLOPs, layer‑independent.**

**Exception — the first layer (B0) skips dX.** There is no layer before B0 to receive an input gradient, so the
exporter emits **no ConvGradX** for B0 (the trace has only `ConvGradW`+`ConvGradB` for node_1). ⇒
`MAC_bwd(B0) = MAC_fwd` (1×, not 2×).

---

## 4. Measured cycles per kernel (summed over tiles)

| blk | Conv fwd | ConvGradX (dX) | ConvGradW (dW) | ConvGradB (dB) |
|---|--:|--:|--:|--:|
| B0 | 7,405,374 | — (skipped) | 1,093,044 | 365,471 |
| B1 | 6,121,186 | 3,402,803 | 2,642,291 | 92,359 |
| B2 | 922,827 | 954,448 | 782,108 | 24,740 |
| B3 | 175,772 | 434,761 | 181,157 | 6,955 |
| B4 | 90,059 | 203,149 | 146,955 | 2,101 |

(`ConvGrad` total per block = dX+dW+dB, e.g. B1 = 3,402,803+2,642,291+92,359 = **6,137,453**.)

---

## 5. cycles‑per‑MAC (efficiency) — how each number is formed

`cyc/MAC = kernel cycles ÷ that kernel's MACs`, where `MAC_dX = MAC_dW = MAC_fwd`. Worked examples:

- **Forward B0**: 7,405,374 ÷ 313,600 = **23.61 cyc/MAC**. (313,600·23 = 7,212,800; remainder 192,574;
  192,574/313,600 = 0.61 → 23.61.) Utilization = 0.125/23.61 = **0.53 %** — pathological.
- **Forward B4**: 90,059 ÷ 501,760 = **0.18 cyc/MAC**. Utilization = 0.125/0.18 = **70 %** — near‑peak.
- **dW B0**: 1,093,044 ÷ 313,600 = **3.49 cyc/MAC**.
- **dX B1**: 3,402,803 ÷ 2,494,464 = **1.36 cyc/MAC**.

Full table (cyc/MAC):

| blk | Conv fwd | dX | dW |
|---|--:|--:|--:|
| B0 | 23.61 | — | 3.49 |
| B1 | 2.45 | 1.36 | 1.06 |
| B2 | 1.53 | 1.59 | 1.30 |
| B3 | 0.70 | 1.73 | 0.72 |
| B4 | 0.18 | 0.40 | 0.29 |

**Observation:** the **forward** cyc/MAC spans **23.61 → 0.18 (≈130×)**; the gradient kernels are far more
uniform (dX 0.40–1.73, dW 0.29–3.49). Physical cause: forward B0 has reduction depth `in_ch·kh·kw = 4` and a
huge output (8·14·700) — the arithmetic is shallow and the kernel is dominated by loads/stores/loop overhead;
forward B4 has reduction depth 224 (32·7) and a tiny output (32·14·5) — the FPUs stay saturated.

---

## 6. The professor's question — Conv kernel vs ConvGrad kernel: do we see 2×?

The "2×" requires **both**: (a) dX **and** dW computed, and (b) all three kernels running at the **same
cyc/MAC** so `cyc_dX ≈ cyc_dW ≈ cyc_fwd`. Per‑kernel multiples vs the forward Conv (fig 5):

| blk | dX / fwd | dW / fwd | (dX+dW+dB)/fwd = **ConvGrad/Conv** | verdict |
|---|--:|--:|--:|---|
| B0 | — | 0.15 | **0.20×** | dX skipped **and** forward pathologically slow |
| B1 | 0.56 | 0.43 | **1.00×** | forward slow (2.45 c/MAC) → each grad kernel < 1× |
| **B2** | **1.03** | **0.85** | **1.91×** | **dX≈fwd, dW≈fwd ⇒ textbook 2×** ✓ |
| B3 | 2.47 | 1.03 | **3.54×** | forward fast (0.70) → dX inefficient → dX ≈ 2.5× |
| B4 | 2.26 | 1.63 | **3.91×** | forward near‑peak (0.18) → both grad kernels > 1× |

**Answer: we observe ≈2× only at B2.** There, `dX/fwd = 1.03` and `dW/fwd = 0.85`, i.e. each gradient kernel
costs about one forward conv — exactly the textbook picture — so `ConvGrad ≈ 2× Conv`. Everywhere else the
equal‑efficiency assumption (b) fails:
- **B0/B1**: the forward conv is *inefficient* (23.6 / 2.45 cyc/MAC — few channels, large spatial), so the
  gradient kernels — which run at ~1–3.5 cyc/MAC — look *cheaper* than the forward (each < 1×). Plus B0 skips
  dX entirely. ⇒ ratio **< 2×**.
- **B3/B4**: the forward conv is *efficient* (0.70 / 0.18 cyc/MAC — small spatial, deep channel reduction), so
  the same‑MAC gradient kernels, running at their usual ~0.3–1.7 cyc/MAC, look *expensive* (> 1× each). ⇒ ratio
  **> 2×** (dX alone is 2.5× the forward at B3).

**One‑line defense:** *In MACs, ConvGrad = 2× Conv exactly (dX and dW each equal the forward). In cycles the 2×
only survives if the three kernels have equal cycles‑per‑MAC; on this hardware the forward efficiency varies
130× across layers, so the measured Conv→ConvGrad ratio ranges 0.20×–3.9× and equals 2× only at the one block
(B2) where forward and gradient efficiencies coincide.*

---

## 7. Network totals and the ratio decomposition

Summing the per‑block Conv/ConvGrad totals (§4):

`Σ Conv_fwd = 14,715,218` cyc; `Σ ConvGrad = 10,332,342` cyc ⇒ **conv backward/forward = 0.70×**.

Per‑block, the ratio factorizes as **ratio = FLOP_factor × efficiency_factor**, with
`FLOP_factor = MAC_bwd/MAC_fwd` and `efficiency_factor = (bwd cyc/MAC)/(fwd cyc/MAC)`:

| blk | FLOP_factor | eff_factor (bwd c/MAC ÷ fwd c/MAC) | product = ratio |
|---|--:|--:|--:|
| B0 | 1 (dX skip) | 4.65/23.61 = **0.20** | 0.20× |
| B1 | 2 | 1.23/2.45 = **0.50** | 1.00× |
| B2 | 2 | 1.46/1.53 = **0.95** | 1.91× |
| B3 | 2 | 1.24/0.70 = **1.77** | 3.54× |
| B4 | 2 | 0.35/0.18 = **1.96** | 3.91× |

(bwd cyc/MAC = ConvGrad_total ÷ MAC_bwd, e.g. B0: 1,458,515/313,600 = 4.65; B1: 6,137,453/4,988,928 = 1.23.)
Here **eff_factor = 0.20** means the B0 backward spends only 0.20× as many cycles per MAC as its (pathological)
forward — i.e. 5× more efficient per useful op — which, times the FLOP_factor of 1, gives the 0.20× ratio.

**Whole step (all ops, not just conv):** forward 17.68 M, backward 16.36 M, optimizer 0.05 M ⇒ **backward/forward
= 0.93×.** The non‑conv backward adds `MaxPoolGrad` 4.25 M (the recompute‑from‑X penalty, 4.7× its forward
MaxPool of 0.91 M) + BNGrad 1.32 M + ReluGrad 0.36 M.

---

## 8. Compute vs transfer (whole step)

Sum `Kernel` vs `Pre+Post` over **all** nodes:
```
Kernel (compute)  = 32,353,106 cyc = 94.9 %
DMA-in  (Pre)     =    989,231 cyc =  2.9 %
DMA-out (Post)    =    753,182 cyc =  2.2 %   ⇒ transfer 5.1 %
Total (profiled)  = 34,095,519 cyc
```
Cross‑check vs on‑device counter: `BENCH train_cycles = 34,084,097` → **0.03 % apart** (trace is faithful).
Note: single‑buffer ⇒ DMA and compute do **not** overlap, yet transfer is only 5.1 % — so double buffering
could save ≤5 % and wouldn't fit L1 (99.8 % full). **Bottleneck = compute, and compute is inefficient.**

---

## 9. Effective FLOP/s and utilization (derivation)

`FLOP/s = (useful FLOPs done) ÷ (wall‑time)`.
```
MAC_conv (fwd+bwd) = Σ_blk (MAC_fwd + MAC_bwd)
  fwd  = 313,600+2,494,464+602,112+250,880+501,760          = 4,162,816
  bwd  = 313,600 + 2·(2,494,464+602,112+250,880+501,760)     = 8,012,032
  total MAC = 12,174,848  ⇒ FLOP = 2·MAC = 24,349,696 = 24.3 MFLOP
wall = 34,084,097 cyc ÷ 370 MHz = 0.0921 s (92.1 ms)
FLOP/s = 24.3 MFLOP ÷ 0.0921 s = 0.264 GFLOP/s
peak (fp32) = 8 cores · 2 FLOP/cyc · 370 MHz = 5.92 GFLOP/s
utilization = 0.264 / 5.92 = 4.5 %
```
So although the step is **95 % compute‑bound** (kernel‑time ≫ DMA), the **effective FPU utilization is only
4.5 %** — dominated by B0's 23.6 cyc/MAC. *"Compute‑bound" means kernel‑time‑bound, not FPU‑efficient.* The
latency lever is kernel efficiency (im2col/GEMM, the B0 shape), not DMA.

---

## 10. Benchmark counters and one fine‑tuning round

**On‑device BENCH (this run, n_accum 1):**
```
train_cycles = 34,084,097   (1 forward + 1 backward)
opt_cycles   =     61,480   (SGD weight update)
weight_sram  =     61,956 B (the 22 trainable weights held in L1/L2)
```

**One BP fine‑tuning round** (exp4 recipe: 40 epochs × ⌈54/4⌉ = **540 device steps**, n_accum 4):
```
per fwd+bwd (n_accum 1)          = 34,084,097 cyc
per device step = 4·(fwd+bwd)+opt = 4·34,084,097 + 61,480 = 136,397,868 cyc
round = 540 steps               = 73,654,848,720 cyc ≈ 73.65 G cycles
  @ 370 MHz → 73.65e9/370e6 = 199.1 s = 3.32 min
  @ 240 MHz → 73.65e9/240e6 = 306.9 s = 5.11 min
```
(Within ~3 % of exp4's independent extrapolation, 75.8 G / 3.4 min.) **Always report `single‑step BENCH ×
step count`** — the on‑device *round* counter is uint32 and wraps ~17× over a full BP round, so its printed
value is meaningless.

---

### Figures
`lat_fig1` compute/transfer + fwd/bwd · `lat_fig2` per‑operator · `lat_fig3` per‑block Conv vs ConvGrad ·
`lat_fig4` per‑block cyc/MAC efficiency · `lat_fig5` Conv vs dX vs dW kernels.
### Scripts (re‑derive everything)
`kernel_split.py` (§4–6) · `perblock_conv.py` (§7) · `flops_and_round.py` (§9–10) · `parse_profiletiling.py` (§8).
