# exp16b — Findings

Date: **2026-09-11** · Branch `feat/GAP9_w_NE16` · Plan: `./Plan.md`
Sibling: `../exp16a_PW_single_layer` (all-pointwise, bit-exact at full extent)

> **Status: PERFORMANCE ANSWERED — and it contradicts the prediction. CORRECTNESS NOT ACHIEVED.**

---

## 1. Headline: the cycle model was wrong

| decomposition | dispatches | extent | errors | **cycles** |
|---|---|---|---|---|
| **dense 3×3 chunks** (exp16b) | **6** | FULL 14×87 | 14328 / 19712 ✗ | **1,362,823** |
| all-pointwise (exp16a) | 16 | FULL 14×87 | **0 / 19712** ✓ | 1,567,096 |
| cluster `pulp_nn_conv` | — | FULL 14×87 | 0 / 19712 ✓ | 337,657 |

**Dense 3×3 chunks are 1.15× FASTER than all-pointwise** — against a prediction of ~3× *slower*.

`Plan.md §3` predicted, from `ne16_matrixvec.cpp:316`, that 3×3 mode spends its 9 row-slots on
spatial taps and therefore walks the 8 bitplanes sequentially (`mv_qw_lim = qw`), giving
`6 chunks × 8 = 48` cycle-units against pointwise's `16 × 1 = 16`. Measured, the opposite happens.

> **Why the model failed: it assumed the MAC array is the bottleneck. At this problem size it is
> not.** Going from 16 dispatches to 6 removes 10 job setups *and* 10 of the 15 int32 `streamin`
> round-trips through L1. With `Cin = 8` (half of `TP_IN`) and an int32 intermediate that is 4× the
> int8 one, this layer is **DMA- and setup-bound, not MAC-bound** — so dispatch count dominates
> utilisation.
>
> This directly undercuts the reasoning in `03-qzo-ne16-plan.md §7`, where masked-3×3 was rejected
> on exactly that MAC-cycle argument. The rejection was analytically sound and empirically wrong.

Cost paid: **13,824 B of weights vs 4,096 B** (3.4×), because 6 of every 9 tap slots are zeros.

Neither variant beats the cluster yet (337,657 cycles) — see `exp16a/Findings.md §5`.

## 2. Correctness: unresolved

Shrinking to **K=3 — one chunk, no streamin, no chunk offsets** still fails:

```
b1_1x3_dense3x3_k3   431 / 1536 errors    (4x24 extent)
```

So the bug is in the **basic 3×3 embedding**, not in chunking or streamin.

The error signature is specific: **every diff is ±1**, on outputs whose range is only `[-2, 3]`.
That is an LSB effect in the requantised int8, i.e. the int32 accumulator differs slightly — not a
structurally wrong convolution.

What has been ruled out:

| hypothesis | test | result |
|---|---|---|
| H tiling breaks the 3×3 vertical receptive field | pinned the non-tapped axis to full extent | 14328 vs 14491 — not the cause (fix kept anyway; it *is* required) |
| chunk offsets / streamin | K=3 → 1 chunk, neither used | still 431/1536 |
| wrong `conf0` | decoded vs the working pointwise task | differs **only** in filter mode (`0x40`); qw, quant bits, outquant, weight-offset mode all identical |
| zero rows not cancelling → mask them instead | `filter_mask = (1<<24)|(1<<8)` | **worse**: 431 → 557. Reverted. The field's top/bottom may index columns not rows (`W_mask[i + j*fs]`), or masking interacts with padding |

The arithmetic *should* cancel: a zero weight stores as `w_u = 0 - (-128) = 128`, and the hardware
computes `Σ(w_u + Wmin)·x` with `Wmin = -128`, so zero-weight taps contribute exactly 0 — including
at padded positions, where `padding_value = 0` (`ne16_regfile.cpp:186`). Why they do not is the
open question.

Next things to try (none attempted): drop the H padding entirely and compare against a matching
`Hout = Hin-2` golden, to isolate padding from the zero-row trick; or dump a single output pixel's
int32 accumulator from GVSoC and compare against a hand-computed value.

## 3. Deeploy changes

| file | change |
|---|---|
| `Targets/NE16/Templates/Conv3x3ChunkTemplate.py` | **new** — `ceil(K/3)` dense 3×3 dispatches, per-chunk pointer offsets, native H padding, streamin |
| `Targets/NE16/Parsers.py` | `NE163x3ChunkConv2DParser` (keyed on `ne16_chunks`); both parsers now publish `ne16_halo` |
| `Targets/NE16/Bindings.py`, `Tiler.py`, `Engine.py` | binding + tiling-ready binding + mapper, ahead of the 1×K mapper |
| `Targets/NE16/TileConstraints/NE161xKConstraint.py` | halo generalised to `ne16_halo` (`taps-1` pointwise, `3*(chunks-1)+2` for chunks); **non-tapped axis pinned when `ne16_chunks` is set** — a 3×3 kernel has a vertical receptive field the constraint does not model per-tile |
| `exp16a/build_fixtures.py` | `--dense3x3`; `extra_w` now derived from the halo rather than hardcoded |

All gated on NE16-only attributes. exp16a's result is unaffected (re-verified: still `0 / 19712`).

## 4. Artefacts

```
Plan.md  Findings.md  results/results.json
fixture/b1_1x16_dense3x3_d33/   full-extent 6-chunk fixture
fixture/b1_1x3_dense3x3_k3/     single-chunk isolation fixture
fixture/Network_1x3_dense3x3.c  generated C for the single-chunk run
logs/step1..step4*.log          full extent, H-pin, single chunk, filter-mask attempt
```

## 5. Recommendation

The **performance question is answered** and is the more consequential result: dispatch count, not
MAC utilisation, drives this layer. That reframes STEP 4 — the lever is *fewer, larger dispatches*,
which points at **channel folding** (im2col to a single `Cin = 128` pointwise conv: 1 dispatch,
`TP_IN` fully packed, no streamin at all) rather than at tap-packing tricks.

Correctness of the 3×3 variant is worth one more focused session, but it is **not on the critical
path**: exp16a already gives a bit-exact NE16 implementation of the same layer.
