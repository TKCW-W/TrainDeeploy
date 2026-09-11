# exp16b — block-1 `1×16` as 3×3 DENSE chunks on NE16

**Parent:** `exp16_NE16_GAP9` · **Sibling:** `exp16a_PW_single_layer` (all-pointwise, bit-exact at
full extent) · Branch `feat/GAP9_w_NE16`, clean at `5fb1de9`

## 1. Goal

Measure the alternative I rejected analytically in `03-qzo-ne16-plan.md §7`: run the same block-1
`1×16` convolution as **`ceil(16/3) = 6` dense 3×3 dispatches** instead of 16 pointwise ones, and
compare cycles against exp16a on the identical fixture data.

The rejection was a cycle-model argument, not a measurement. This makes it a measurement.

## 2. The mechanism

A `1×3` slice of the kernel is embedded in a 3×3 kernel with rows 0 and 2 zeroed:

```
3×3 window            chunk c holds taps 3c, 3c+1, 3c+2
[ 0   0   0 ]
[w3c w3c+1 w3c+2]     ← only the middle row is non-zero
[ 0   0   0 ]
```

* **H padding 1/1** keeps `Hout = 14`. Legal here: padding is valid in NE16's **3×3** mode (it is
  *invalid* in 1×1 mode, which is why exp16a pre-pads). The zero rows make the padded values
  irrelevant whatever they are.
* **W offset `3·c`** on `infeat_addr` selects the chunk's tap group — the same address-offset
  trick exp16a uses, stepping by 3 instead of 1.
* **`streamin`** accumulates chunks 1..5, exactly as in exp16a. Output stays int32; the
  `RequantShift` stays a separate cluster node.
* Input padded by **2 extra columns** so the last chunk's 3-wide window stays in range; taps 16,17
  carry zero weight and contribute nothing.

## 3. What the cycle model predicts

`ne16_matrixvec.cpp:316` — the 9 row-slots per column hold **either** the 8 bitplanes (1×1 mode,
`mv_qw_lim = 1`) **or** the 9 spatial taps (3×3 mode, `mv_qw_lim = qw = 8`):

| | rows used | MAC/cycle | dispatches for 16 taps | cycle units |
|---|---|---|---|---|
| pointwise (exp16a) | 8/9 (bitplanes) | 144 | 16 | 16 × 1 = **16** |
| 3×3 with 3 real taps | 3/9 (taps) | 54 | 6 | 6 × 8 = **48** |

So **~3× worse** despite 2.7× fewer dispatches. Weight memory also grows: 3×3 encoding is
`16·1·8·18 = 2304 B` per chunk × 6 = **13,824 B**, against pointwise's 16 × 256 = **4,096 B**,
because 6 of every 9 tap slots are zeros.

**If the measurement disagrees with this, the cycle model is wrong and that is the finding.**

## 4. Acceptance

1. Bit-exact (`0 / 19712`) at full `14×87` extent against the same golden exp16a uses.
2. `grep -c ne16_nnx_dispatch` == 6 (not 16), `pulp_nn_conv` == 0.
3. Cycles recorded next to exp16a's 1,567,096 (NE16 pointwise) and 337,657 (cluster).

## 5. Reuse

Everything from exp16a except the template and the weight encoding:

| piece | reused? |
|---|---|
| fixture generator, real data, golden | ✅ `build_fixtures.py --dense3x3` |
| `weight_offset = -128` runtime-weight mechanism | ✅ unchanged |
| halo tile constraint | ✅ with `ne16_halo = 3·(chunks-1) + 2` instead of `taps-1` |
| `_merge_conv_rq_fun` guard, NHWC-native graph, `--l1 92000` | ✅ unchanged |
| template | ✗ new: `NE162DDenseConvTemplate` base, chunk stride 3, native H padding |

## 6. Scope

Performance comparison only. No change to exp16a's result, no new blocker work. If 3×3 chunks
lose as predicted, the outcome is a measured rejection and the pointwise path stands.
