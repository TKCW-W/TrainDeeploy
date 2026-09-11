# exp16c — Findings

Date: **2026-09-11** · Branch `feat/GAP9_w_NE16` · Plan: `./Plan.md`
Predecessors: `../exp16a_PW_single_layer`, `../exp16b_Dense_single_layer`

> **Phase 1 COMPLETE.** The NE16 bit-serial weight encoding now happens **on device**, and the
> block-1 layer stays bit-exact at full extent. Blocker 1b's hard half is closed.
> Phases 2–4 open.

---

## 1. Phase 1 result

| fixture | weight encoded by | dispatches | errors | cycles |
|---|---|---|---|---|
| `b1_1x16_devenc_dev` | **device** (`NE16WeightEncode`) | 16 | **0 / 19712** ✓ | 1,593,285 |
| `b1_1x16_ne16_nhwc` (exp16a) | host (`_weightEncode`) | 16 | 0 / 19712 ✓ | 1,567,096 |

Generated C confirms the structure: `encode_calls=1`, `dispatches=16`, `transposes=0`,
`cluster convs=0` (`fixture/Network_1x16_devenc.c`).

### The cost — and what it settles

```
26,189 cycles  =  1,593,285 − 1,567,096  =  1.64 % of the layer
```

That covers the L2→L1 DMA of the 2,048 B raw int8 weight **and** the bit-serial encode of 4,096 B
across 8 cores.

> **This closes the Option A / Option B question in `Plan.md §2`.** The plan document
> (`docs/TRAIN_GAP9_NE16/03-qzo-ne16-plan.md §3.2`) recommended Option B, the linearity
> decomposition `conv(w+δz,x) = conv(w,x) + δ·conv(z,x)`, whose entire purpose is to *avoid* an
> on-device 8-bit re-encode. The re-encode costs **1.64 %**. Option B would pay for that saving by
> **doubling the dispatch count** (16 → 32) — the metric exp16b showed actually dominates this
> layer — plus a new per-channel int32 scale-and-accumulate kernel, two perturbation constants
> (`δ⁺ ≠ δ⁻`), and a per-step sign-packing pass anyway.
>
> **Option A wins on the measurement.** Option B is withdrawn, not merely deprioritised.

## 2. What was built

| file | what |
|---|---|
| `TargetLibraries/GAP9/src/NE16WeightEncode.c` | **new** — the device kernel. C port of `_weightEncode` for the `H*W == 1` per tap, `bits == 8`, dense case |
| `TargetLibraries/GAP9/inc/DeeployGAP9Math.h` | prototype (kept out of pulp-nnx, which stays pristine) |
| `Deeploy/Targets/NE16/WeightEncode.py` | **new** — parser, type checker, template, binding, layer for the `NE16WeightEncode` op |
| `Deeploy/Targets/NE16/TileConstraints/NE16WeightEncodeConstraint.py` | **new** — pins both tensors full; the node is deliberately untiled |
| `Deeploy/Targets/NE16/Tiler.py` | `NE16WeightEncodeTilingReadyBindings` |
| `Deeploy/Targets/GAP9/Platform.py` | mapper + `'NE16WeightEncode'` entry in `GAP9Mapping` |
| `exp16a/build_fixtures.py` | `--device-encode` |

**No NE16 ISA change.** The kernel is ordinary C in `deeploygap9`; `TargetLibraries/third_party/pulp-nnx/`
is untouched. This runs on real GAP9 silicon.

### Why the encoder reduces to a bit transpose

`_weightEncode` (`Deeploy/Targets/NE16/TopologyOptimizationPasses/Passes.py:24`) does two things:

| host step | on device |
|---|---|
| `weight_offset = values.min()`; `values -= weight_offset` | **gone** — exp16a hardwires `weight_offset = −128`, so this is exactly `w_u = w ^ 0x80` |
| bit-plane transpose + `packbits` | reproduced verbatim — a **data-independent** permutation |

Fixing the offset at −128 is what makes this tractable: a `values.min()` offset would change every
ZO step and would have to be recomputed, reduced across cores, and pushed into the conv's
`weight_offset_factor` register. Every int8 weight satisfies `w + 128 ∈ [0,255]`, so −128 is always
valid and never needs recomputing.

For one `(row, cinMajor)` pair the kernel gathers 16 input-channel lanes and emits 16 bytes, where
output byte `b*2 + k` collects bit `b` of lanes `8k..8k+7`, LSB-first. Rows (`taps*cout` of them)
are independent, so the template chunks them across the 8 cluster cores.

### The encoder was validated before it ever reached the device

`TargetLibraries/GAP9/src/NE16WeightEncode.c` was compiled **natively** and called via `ctypes`
against Deeploy's own `_weightEncode`, for every SpeechNet conv shape:

```
b1 1x16  cout=16 cin= 8 K=16  enc(256,1,16) -> MATCH     (8-core chunked -> MATCH)
b0 1x4   cout= 8 cin= 1 K= 4  enc( 32,1,16) -> MATCH     (8-core chunked -> MATCH)
b2 1x8   cout=16 cin=16 K= 8  enc(128,1,16) -> MATCH     (8-core chunked -> MATCH)
b3 7x1   cout=32 cin=16 K= 7  enc(224,1,16) -> MATCH     (8-core chunked -> MATCH)
b4 7x1   cout=32 cin=32 K= 7  enc(224,2,16) -> MATCH     (8-core chunked -> MATCH)
odd      cout= 8 cin=20 K= 3  enc( 24,2,16) -> MATCH     (8-core chunked -> MATCH)
```

Two things fall out of this, both load-bearing for later phases:

1. **`cinMajor > 1` and non-multiple-of-16 `cin` already work** — block 4 (`cin=32`) needs no new code.
2. **`K×1` needs no new code either.** In NCHW the tap is the last index whether the kernel is
   `1×K` (H=1, W=K) or `K×1` (H=K, W=1), so `src[(co*cin + ci)*taps + j]` covers both. Blocks 3
   and 4 reuse the same encoder.

That is why phase 1 went green on its **first** device run — the two-hour class of bug that
dominated exp16a and exp16b was priced out in a 3-second native test instead.

## 3. What is still open

| | |
|---|---|
| **Phase 2** | the weight is still a *graph input*. In QZO it is an `RQSPerturbRademacher` output — wire that producer in and run L⁺/L⁻ |
| **Phase 3** | STEP 3a: blocks 2, 3, 4 (`1×8`, `7×1`, `7×1`) |
| **Phase 4** | STEP 3b: block 0, which additionally needs the signed-activation fix (BLOCKER 3) |
| STEP 4 | NE16 is still ~4× slower than `pulp_nn_conv`; not this experiment's problem |

## 4. Artefacts

```
Plan.md  Findings.md  results/results.json
fixture/Network_1x16_devenc.c     generated C (1 encode call, 16 dispatches, 0 transposes)
logs/step1_devenc_FULL.log        the passing device run
```

Fixture data lives at `DeeployTest/Tests/Models/NE16/b1_1x16_devenc_dev/`
(`inputs.npz` carries the **raw int8** weight; `weight_enc_golden.npz` keeps the host-encoded bytes
alongside, so a future mismatch can be localised to the encoder rather than the conv).
