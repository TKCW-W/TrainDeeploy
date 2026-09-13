# On-device training memory breakdown — BP baseline (exp1, argmax-off)

Single training step, SpeechNet, S01 / vocalized / fold 3, GAP9 budget **L1 = 128 KB, L2 = 2 MB**.
Source: `memory_alloc_deeployStates.html` (train graph) + `..._optimizer.html`, parsed to per-tensor
offset/size/lifetime. Figures: `mem_fig1/2/3_*.png`.

## 1. Headline — peak vs capacity per level  (fig 1)
The number that must *fit* is the arena high-water (max occupied offset) per level:

| Level | Capacity | Peak used | Utilization | Binding? |
|---|--:|--:|--:|:--:|
| **L1 (cluster TCDM)** | 128,000 B | **127,808 B** | **99.8 %** | **← binding constraint** |
| **L2** | 2,000,000 B | **1,735,180 B** | 86.8 % | headroom |
| L3 | 64,000,000 B | 0 B | 0 % | unused |

**L1 is the bottleneck** (99.8 % full): the tiler packs the working set to the byte. L2 has ~13 % headroom;
L3 is never touched. L1's high-water is set by a single tiled buffer — the block-0 Conv gradient
(117,768 B) — so any L1 relief has to target the largest tiled tensor, not the count of tensors.

## 2. L2 peak composition by training role  (fig 2)
At the busiest step (t = 24) **1,552 KB is simultaneously live** (arena reserves 1,695 KB → 8.4 % fragmentation):

| Role | Bytes | Share of peak |
|---|--:|--:|
| Layout-transpose copies (NCHW↔NHWC) | 870,016 B | **54.7 %** |
| Forward activations | 496,404 B | 31.2 % |
| I/O + loss | 163,120 B | 10.3 % |
| Gradients (backward) | 58,760 B | 3.7 % |
| Frozen BN stats | 1,152 B | 0.1 % |
| Weights (params) | ~1.6 KB (persistent) | <0.1 % |

**Key insight:** the training-specific cost (gradients) is only **3.7 %** at L2 — the backward pass is cheap in
L2 because gradient buffers are tiled through L1 (that's *why* L1 binds). L2 is dominated by the **forward
pass**: activations + their layout-transpose copies together are **86 %**. More than half of L2 is spent on
**NCHW↔NHWC layout conversion copies**, the single largest, most optimizable line item (fuse layout into the
kernel / keep one layout end-to-end).

## 2b. What blocks are resident *at* the peak  (fig 4)
Same peak step (t = 24, 1,552 KB live), now decomposed by **which network block each live tensor belongs to**,
stacked by role:

| Block present at peak | Bytes | Share | What it is |
|---|--:|--:|---|
| **Block 0** | 958 KB | **61.7 %** | fwd activation (307 KB) + its 2 layout copies (613 KB) + gradient (38 KB) |
| Block 1 | 250 KB | 16.1 % | fwd activation (77 KB) + layout copies (154 KB) + gradient (19 KB) |
| Data / graph I-O | 220 KB | 14.2 % | EMG input window (38 KB) + persistent weight buffers + fwd output pointers |
| Block 2 | 81 KB | 5.2 % | activation + layout copies |
| Block 4 | 28 KB | 1.8 % | persistent transposed weight buffer |
| Block 3 | 14 KB | 0.9 % | persistent transposed weight buffer |
| Head/FC | ~0 | 0.0 % | not yet reached at this step |

**Key insight — the whole forward stack is co-resident.** At the peak, blocks 0-4 are *all* live at once. This is
the **activation-stashing signature of backprop**: block-0's forward activation (314 KB) is allocated early and
kept alive all the way until its backward pass runs at the *end* of the step, so it overlaps with everything in
between. Blocks 3-4 appear only as their small persistent transposed-weight buffers (they haven't produced
activations yet at t = 24); the real weight is that **block-0 dominates (62 %)** because its large early-spatial
activation is stashed for the longest lifetime. This is exactly what an activation-checkpointing / recompute
scheme would target — free block-0's stash and recompute it in the backward instead of holding it.

## 2c. Peak memory per block, by operator  (fig 5)  ← main slide
The peak (1,552 KB live @ t = 24) split **per block into its operators** (Conv / BatchNorm / ReLU / MaxPool);
all graph inputs/weights collapsed into one "Input / data" bar. Each operator's bar includes its output
activation + layout-transpose copies + any live gradient.

| Block | Conv | BatchNorm | ReLU | MaxPool | Total | Share |
|---|--:|--:|--:|--:|--:|--:|
| **Block 0** | 307 KB | 307 KB | – | 345 KB | **958 KB** | **62 %** |
| Block 1 | 77 KB | 77 KB | – | 96 KB | 250 KB | 16 % |
| Block 2 | 20 KB | 20 KB | 20 KB | 20 KB | 81 KB | 5 % |
| Block 4 | 28 KB | – | – | – | 28 KB | 2 % |
| Block 3 | 14 KB | – | – | – | 14 KB | 1 % |
| Input / data | — one collapsed bar — | | | | 220 KB | 14 % |

**Reading it:**
- **Block 0's three big operators (Conv 307, BatchNorm 307, MaxPool 345 KB) are the peak.** Each holds a full
  314 KB-class tensor; MaxPool is largest because it carries both its transposed input copy and the pooled output.
- **ReLU is ~invisible** in blocks 0/1: its output is immediately aliased/consumed into the following MaxPool's
  transposed buffer, so no separate ReLU tensor is live at peak (only block 2 shows a distinct 20 KB ReLU).
- **Blocks 3-4 have no BatchNorm/MaxPool live** — pool = 1×1 (no pooling) and at t = 24 they contribute only
  their persistent transposed Conv-weight buffer.
- The **operator hierarchy is Conv ≈ BatchNorm ≈ MaxPool per block**, and blocks shrink ~4× each stage → block 0
  is where every operator is most expensive.

## 2d. Forward vs backward, per block × operator — two peaks  (fig 6)  ← most detailed
There are **two** memory peaks in a training step, and they have opposite composition:

**Are transpose (layout) copies forward or backward?** → **Forward.** They are NCHW↔NHWC conversions that feed the
Conv/MaxPool kernels on the forward pass.

> **Classifier correction (applied to fig 6 & fig 8):** a tensor counts as `backward-gradient` only when the
> `grad` marker is on its *own* activation output (matches `__N_grad`). Buffers that are forward activations merely
> *stashed for* a downstream `Grad` consumer — their names embed the consumer node, e.g.
> `..._MaxPool__0_tensor_pre_transpose_node_5_..._Conv_GradConvGrad_..._transpose_in_var` — are **`fwd-transpose`,
> not gradients**. Fixing this removes the phantom red at the forward peak.

| Role at peak | Forward peak (t=24) | Backward peak (t=94) |
|---|--:|--:|
| fwd-activation (op output tensor) | 424 KB (27.3 %) | 0 KB (0 %) |
| fwd-transpose (NCHW↔NHWC layout copy) | **907 KB (58.4 %)** | 307 KB (26.9 %) |
| backward-gradient | **0 KB (0 %)** | **614 KB (53.7 %)** |
| input / weights | 221 KB (14.3 %) | 221 KB (19.4 %) |
| **total live** | **1,552 KB** | **1,142 KB** |

- **Forward peak (global max, 1,552 KB):** forward activations + their transpose copies = **86 %**; gradients
  **0 %** (the backward hasn't started). All five blocks' forward tensors co-resident (activation stashing), block 0 = 62 %.
- **Backward peak (1,142 KB):** now **gradients are 54 %** — and it's *entirely block 0*: its Conv-input-grad
  (307 KB) + BatchNorm-grad (307 KB) plus the still-stashed 307 KB forward-layout copy of block-0's input that
  the ConvGrad needs. Deeper blocks have already been consumed and freed.
- **Per operator:** on the forward side Conv ≈ BatchNorm ≈ MaxPool each hold one ~307 KB block-0 tensor; on the
  backward side the cost is Conv-weight/-input gradients + BatchNorm gradient, again all block 0.
- **The whole training memory envelope is block-0-bound in both directions** — forward stashing sets the 1.55 MB
  global peak; the backward gradient sets a second 1.14 MB peak. Activation checkpointing of block 0 would cut
  both.

## 3. Activation memory by network block  (fig 3)
Forward activation + layout footprint in L2, per block:

| Block | Bytes | Share |
|---|--:|--:|
| **Block 0** | 1,295,520 B | **70.4 %** |
| Block 1 | 323,584 B | 17.6 % |
| Block 2 | 90,624 B | 4.9 % |
| Block 4 | 37,632 B | 2.0 % |
| Block 3 | 29,696 B | 1.6 % |
| Head (FC/loss) | 128 B | 0.0 % |

**Block 0 alone is 70 % of activation memory.** Early blocks have large spatial dimensions (700-wide EMG
window, few channels) → each block-0 tensor is 314 KB. Memory pressure is front-loaded; the deep blocks and
the classifier head are negligible. Optimizing block-0 (tiling, layout, or activation checkpointing) is where
memory savings live.

## 4. Optimizer graph (separate pass)
The SGD update runs as its own graph: **L2 peak 186 KB (9.3 %), L1 peak 86 KB (67.5 %)** — far below the train
graph, so the train forward/backward graph sets the memory envelope, not the optimizer.

## Takeaways for the slide
1. **L1 (128 KB) is the binding resource at 99.8 %**; L2 has 13 % slack, L3 unused.
2. **Training overhead (gradients) is small (3.7 % of L2)** — the forward pass dominates memory, same as inference.
3. **Layout-transpose copies are the biggest single cost (55 % of L2)** and the clearest optimization target.
4. **Memory is front-loaded: block 0 = 70 % of activations** — large early spatial dims, not depth.

*Reproduce:* `python3 /tmp/mem_breakdown.py` (parser + figures) against the two `memory_alloc_*.html` files in this dir.
