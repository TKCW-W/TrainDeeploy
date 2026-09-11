# exp16b — Findings

Date: **2026-09-11** · Branch `feat/GAP9_w_NE16` · Plan: `./Plan.md`
Sibling: `../exp16a_PW_single_layer` (all-pointwise, bit-exact at full extent)

> **Status: COMPLETE.** The 3×3-dense-chunk decomposition is **bit-exact at full extent** and
> **1.15× faster** than all-pointwise. Both the correctness question and the performance question
> are answered, and the performance answer contradicts the prediction that motivated the experiment.

---

## 1. Headline

| decomposition | dispatches | extent | errors | **cycles** |
|---|---|---|---|---|
| **dense 3×3 chunks** (exp16b) | **6** | FULL 14×87 | **0 / 19712** ✓ | **1,364,975** |
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
> This directly undercuts the reasoning in `docs/TRAIN_GAP9_NE16/03-qzo-ne16-plan.md §7`, where
> masked-3×3 was rejected on exactly that MAC-cycle argument. The rejection was analytically sound
> and empirically wrong.

Cost paid: **13,824 B of weights vs 4,096 B** (3.4×), because 6 of every 9 tap slots are zeros.

Neither variant beats the cluster yet (337,657 cycles, NE16 is 4.0× behind) — see
`../exp16a_PW_single_layer/Findings.md §5`.

## 2. Correctness — root cause and fix

The failing intermediate states (`14328/19712` at full extent; `431/1536` on the single-chunk
K=3 isolation fixture) had one cause, and it was **not** in the 3×3 embedding.

**File:** `Deeploy/Targets/NE16/TileConstraints/NE161xKConstraint.py`, `serializeTilingSolution`.

It unconditionally emitted the **pointwise** subtile counters and no padding:

```python
replacements["input_addr_offset"].append(0)
counters = NE162DPWConvTemplate.getCounters(inCSz, hSz, wSz, cSz, 0, 0, operatorRepresentation)
```

NE16 retires output in 3×3 subtiles and needs the *input* extent of the **border** subtile declared
explicitly in `bHi` / `bWi`. For a 1×1 job that equals the output border. For a 3×3 job it is

```
bHi = height_out_border + 2 - padding_bottom          # NE162DDenseConvTemplate.getCounters
```

the `+2` being the 3×3 receptive field. Handing a 3×3 job the pointwise formula makes its border
subtile read a window two pixels short in each spatial direction, so the edge taps are silently
dropped. Each affected output loses a small int32 amount, which after the `>>16` requantisation
shows up as **±1 LSB** on outputs whose range is only `[-2, 3]` — exactly the observed signature,
and exactly why the errors clustered at subtile boundaries.

The fix branches on the variant:

```python
isDense3x3 = 'ne16_chunks' in operatorRepresentation
if isDense3x3:
    padB = int(operatorRepresentation['padding_y_bottom'])
    padR = int(operatorRepresentation['padding_x_right'])
    replacements["input_addr_offset"].append(
        getInputAddrOffset(inWSz, yStrideIn, padding_y_top, padding_x_left))
    counters = NE162DDenseConvTemplate.getCounters(inCSz, hSz, wSz, cSz, padB, padR, operatorRepresentation)
else:
    replacements["input_addr_offset"].append(0)   # input pre-padded; no NE16 padding
    counters = NE162DPWConvTemplate.getCounters(inCSz, hSz, wSz, cSz, 0, 0, operatorRepresentation)
```

Result: `b1_1x3_dense3x3_k3` → **0/1536**, `b1_1x16_dense3x3_d33` → **0/19712**.

### 2.1 Hypotheses that were wrong — kept as a record

| hypothesis | test | verdict |
|---|---|---|
| H tiling breaks the 3×3 vertical receptive field | pinned the non-tapped axis to full extent | 14328 vs 14491 — not *the* cause, but the pin **is** genuinely required (the constraint models no per-tile vertical halo or per-tile padding) and is kept |
| chunk offsets / streamin | K=3 → one chunk, neither used | still failed → bug was upstream of chunking |
| wrong `conf0` | decoded against the working pointwise task | differs **only** in filter mode (`0x40`) — correct |
| zero-weight rows not cancelling | — | they cancel exactly: a zero weight stores as `w_u = 0 − (−128) = 128`, hardware computes `Σ(w_u + Wmin)·x` with `Wmin = −128` |
| mask the zero rows instead | `filter_mask = (1<<24)｜(1<<8)` | **worse**: 431 → 557. Reverted; `ne16_filter_mask` stays 0 |

The lesson worth carrying: a ±1-LSB error signature on a *narrow* output range reads like a rounding
or datapath subtlety, but here it was a plain geometry bug. The tell was that it was
**position-dependent** (subtile borders), which no datapath explanation accounts for.

## 3. Deeploy changes

| file | change |
|---|---|
| `Targets/NE16/Templates/Conv3x3ChunkTemplate.py` | **new** — `ceil(K/3)` dense 3×3 dispatches, per-chunk pointer offsets, native H padding, streamin |
| `Targets/NE16/Parsers.py` | `NE163x3ChunkConv2DParser` (keyed on `ne16_chunks`); both parsers now publish `ne16_halo` |
| `Targets/NE16/Bindings.py`, `Tiler.py`, `Engine.py` | binding + tiling-ready binding + mapper, ahead of the 1×K mapper |
| `Targets/NE16/TileConstraints/NE161xKConstraint.py` | halo generalised to `ne16_halo` (`taps-1` pointwise, `3*(chunks-1)+2` for chunks); non-tapped axis pinned when `ne16_chunks` is set; **variant-specific counters + padding in `serializeTilingSolution`** (§2) |
| `exp16a/build_fixtures.py` | `--dense3x3`; `extra_w` now derived from the halo rather than hardcoded |

All gated on NE16-only attributes. exp16a's result is unaffected — re-verified after every change,
still `0 / 19712` at 1,567,096 cycles.

## 4. Artefacts

```
Plan.md  Findings.md  results/results.json
fixture/b1_1x16_dense3x3_d33/            full-extent 6-chunk fixture
fixture/b1_1x3_dense3x3_k3/              single-chunk isolation fixture
fixture/Network_1x3_dense3x3.c           generated C, single-chunk run
fixture/Network_1x16_dense3x3_FULL.c     generated C, full-extent 6-chunk run (6 dispatches,
                                         0 transposes, 0 cluster convs)
logs/step1..step4*.log                   failing states: full extent, H-pin, single chunk, filter-mask
logs/step5,step8*.log                    exp16a regressions
logs/step6,step7,step9*.log              after the counters fix: K=3 0/1536, FULL 0/19712
```

## 5. What this changes for STEP 4

Dispatch count, not MAC utilisation, drives this layer. That reframes the optimisation target:
the lever is **fewer, larger dispatches**, which points at **channel folding** — im2col the 1×16
into a single `Cin = 128` pointwise conv: 1 dispatch, `TP_IN` fully packed, no streamin at all —
rather than at tap-packing tricks. The 3.4× weight blow-up of the 3×3 route is the price of the
same idea applied only halfway.

Both decompositions are now bit-exact, so STEP 4 can pick on cycles alone.

## 6. Gate for exp16c

exp16b's success opens the next experiment (`../exp16c_PW_single_layer`): **blocker 1b**
(on-device perturbation — the host currently supplies a pre-perturbed, pre-encoded weight) and
**STEP 3** (the remaining four SpeechNet convs).
