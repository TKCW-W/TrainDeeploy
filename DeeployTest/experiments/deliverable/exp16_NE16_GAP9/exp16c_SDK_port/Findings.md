# exp16c_SDK_port — Findings

Date: **2026-09-14** · Branch `feat/GAP9_w_NE16` · Plan: `./Plan.md`
Driver: `ETH/WorkLog/GAP9_SDK_NE16_Corner_Cases.md`

> **SpeechNet QZO compiles and runs end-to-end on GAP9 with NE16 engaged**, with two convolutions
> on the accelerator, their weights perturbed *and* bit-serial encoded on device, and nothing
> hand-authored in the graph. The plumbing works.
>
> **But the full-network result is numerically WRONG** (§5): measured against the host reference
> losses the NE16 run is 20–55 % off where the cluster baseline matches to ~1e-5 — and the harness
> reports `PASSED` regardless, because its check is driven by reference losses rather than the
> device's. Every layer is bit-exact *in isolation*, including under the same tiling, so the fault
> is in the integration. §5 lists six eliminated hypotheses and the next step.
>
> Padded convolutions are also still refused, so only 5.8 % of conv MACs reach NE16 (§6).

---

## 1. Results

### Phase 1 — pipelined dispatch (SDK technique A)

The per-tap loop now blocks only while NE16's 2-deep job queue is full and waits for completion
once after the whole chain, mirroring `KerConv1D_StrideS_NE16`'s *"already commit and trigger NE16
computation, while programming the next one"*.

| block | before | after | saved | errors | cluster | ratio |
|---|---|---|---|---|---|---|
| 0 | 5,517,302 | 5,480,672 | 0.66 % | 0 / 78512 | 732,178 | 7.49× |
| 1 | 1,608,939 | 1,548,078 | **3.78 %** | 0 / 19712 | 337,999 | 4.58× |
| 2 | 420,743 | 398,739 | **5.23 %** | 0 / 5152 | 98,723 | 4.04× |
| 3 | 183,651 | 176,503 | 3.89 % | 0 / 1280 | 26,835 | 6.58× |
| 4 | 145,990 | 142,915 | 2.11 % | 0 / 320 | 30,329 | 4.71× |
| **total** | **7,876,625** | **7,746,907** | **1.65 %** | — | 1,226,064 | **6.32×** |

**This corrects exp16b.** exp16b called the layer "DMA- and **setup**-bound" and I read *setup* as
job programming. Removing nearly all of that programming cost buys **1.65 %** — so job setup was
not the dominant term. exp16b's observation (fewer dispatches *is* faster) stands, but the
mechanism is the **int32 `streamin` round-trips through L1**.

Consequences: **SDK technique B** (10 register writes instead of 24) is **dropped** — it would mean
abandoning pulp-nnx's task API for a fraction of an already-1.65 % term. **Technique C** (im2col
channel folding) matters *more*, because it removes the streamin traffic outright.

### Phase 2 — automatic `1×K` → NE16 rewrite

Verified on **plain** graphs: ordinary NCHW `Conv`, runtime weight from `RQSPerturbRademacher`,
**zero** NE16 attributes authored by hand.

| fixture | errors | cycles | dispatches | device encodes | cluster convs |
|---|---|---|---|---|---|
| `b3_plain` (7×1) | **0 / 1280** ✓ | 180,585 | 7 | 1 | 0 |
| `b4_plain` (7×1) | **0 / 320** ✓ | 146,409 | 7 | 1 | 0 |

(Above the hand-authored NHWC fixtures — 176,503 / 142,915 — because the plain graph is NCHW and
pays for the layout Transposes.)

### Phase 3 — full SpeechNet QZO on GAP9 with NE16

```
PASSED          Errors: 0 out of 8      (the harness's optimizer-output check)
NE16 dispatches 14        device weight encodes 2        cluster convs 3
```

Blocks **3 and 4** (`7×1`, unpadded) run on NE16 with their weights perturbed and encoded on
device; blocks 0/1/2 stay on the cluster because they are padded (§6).

| | NE16 | cluster-only |
|---|---|---|
| train cycles | 68,077,617 | 66,817,357 |
| optimizer cycles | 372,466 | 372,177 |

NE16 is currently **1.9 % slower** on the full training step — unsurprising, since the two blocks it
takes are 5.8 % of conv MACs and it runs them ~5–7× slower than the cluster.

## 2. Reproduction

All commands run inside the `deeploy_gap9` container, which mounts `ETH/TrainDeeploy` at
`/app/Deeploy`. Prelude for every run:

```bash
docker exec deeploy_gap9 bash -lc '
source /app/install/gap9-sdk/.gap9-venv/bin/activate
source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
pgrep -f "[g]vsoc_launcher" | xargs -r kill -9     # orphans starve every new sim
rm -rf /app/Deeploy/DeeployTest/TEST_GAP9_W_NE16
cd /app/Deeploy/DeeployTest
...'
```

**Build the fixtures** (in `agitated_hugle`, the only container with Onnx4Deeploy importable):

```bash
docker exec agitated_hugle bash -lc 'cd /app && PYTHONPATH=/app/Onnx4Deeploy:/app/TrainDeeploy \
  python3 TrainDeeploy/DeeployTest/experiments/deliverable/exp16_NE16_GAP9/exp16c_PW_single_layer/build_fixtures.py \
  --block 3 --plain'      # and --block 4 --plain ; drop --plain for the hand-authored variant
```

**Phase 1 / 2 — a single layer:**

```bash
python3 deeployRunner_tiled_gap9_w_ne16.py -t Tests/Models/NE16/b3_plain --enable-1xk \
  --l1 92000 --l2 1500000 --defaultMemLevel L2 --cores 8
```

**Phase 3 — the full QZO training step.** The `-D BN_FROZEN_STATS=ON` and `--l1 110000` are
load-bearing: without them `.weightmem_sram` overflows `L2_shared` by ~706 KB, **with or without
NE16** (verified against the cluster-only baseline, so it is not an NE16 cost).

```bash
python3 deeployMezoRunner_tiled_GAP9_w_NE16.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo12_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo12_update \
  --enable-1xk --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
  --l1 110000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON
```

Drop `--enable-1xk` for the cluster-only baseline. Check the engine split with:

```bash
D=TEST_GAP9_W_NE16/Tests/Models/Training/SpeechNet/speechnet_qzo12_train
grep -c "ne16_nnx_dispatch("      $D/TrainingNetwork.c    # 14
grep -c NE16WeightEncode_i8_u8    $D/TrainingNetwork.c    # 2
grep -c pulp_nn_conv              $D/TrainingNetwork.c    # 3
```

Logs for every run above are in `./logs/`; the generated `TrainingNetwork.c` is in `./fixture/`.

## 3. Changes made

| file | change |
|---|---|
| `Deeploy/Targets/NE16/Templates/Conv1xKTemplate.py` | **phase 1** — task declared once outside the loop and reassigned per tap; `ne16_nnx_resolve_wait` moved out of the loop so the 2-deep job queue is actually used |
| `Deeploy/Targets/NE16/Templates/Conv3x3ChunkTemplate.py` | **phase 1** — same, per chunk |
| `Deeploy/Targets/NE16/TopologyOptimizationPasses/Prepare1xKPass.py` | **new, phase 2** — the rewrite pass plus `ne16_1xkAdmissible`, the single source of truth on what NE16 may claim |
| `Deeploy/Targets/NE16/Deployer.py` | **phase 2** — insert the pass at index 1 of the lowering pipeline |
| `Deeploy/Targets/NE16/Engine.py` | **phase 2/3** — `is1xKConv` defers to `ne16_1xkAdmissible`; a 1×K conv no longer needs a `gs.Constant` weight |
| `DeeployTest/testUtils/deeployMezoRunner.py` | **phase 3** — thread `--enable-1xk` / `--enable-3x3` into `gen_args` |
| `DeeployTest/testMVPTraining.py` | **phase 3** — accept those flags and set `enable1xK` / `enable3x3` on the NE16 engine |
| `.../exp16c_PW_single_layer/build_fixtures.py` | `--plain` (emit a graph with no NE16 attributes at all) |

### Three things the pass has to do that the fixtures used to do by hand

* **Relax the engine gate.** Coloring runs *before* the pass, so a runtime weight was rejected by
  `_weightAcceptable` and the rewrite never saw the node.
* **Tag the un-merged RequantShift `channels_first = 1`.** It sits *outside* the NHWC region and
  consumes the NCHW tensor the layout pass transposes back; without the tag
  `RequantShiftLayer.computeShapes` reads the channel count from the last axis —
  `Could not broadcast rqs_mul_tensor from (32,) to [1, 5]`.
* **Decide signedness structurally.** Every `Quant` in the QZO graph declares `signed = 1`, so the
  type says nothing. The producer chain does: blocks 1–4 reach a `Relu` through `Quant ← MaxPool`,
  block 0 reaches the graph input. `ne16_unsigned_input=1` overrides it for graphs where the chain
  cannot show it — needed by the single-layer fixtures, whose activation is a bare graph input and
  where an int8 `Relu` cannot be spliced in because GAP9's only `Relu` binding is fp32
  (`Targets/GAP9/Bindings.py:381`).

### One design correction worth recording

An earlier revision refused inadmissible nodes *only inside the pass*, by dropping their `engine`
attribute — and the very next `EngineColoringPass` simply re-claimed them, yielding a conv coloured
NE16 with no `ne16_taps` that then failed to bind. Admissibility now lives in **one** predicate,
`ne16_1xkAdmissible`, called by both the engine and the pass.

## 4. Ordering constraints on the pass

1. **After engine coloring, before `PULPNCHWtoNHWCPass`**, which permutes every rank-4 conv input
   and takes the spatial rank from the *weight's* rank — both wrong for a bit-serial weight.
   `_NCHWtoNHWC_fun` skips a weight marked `ne16_weight_preencoded`, but only if the mark is already
   there. Hence `insert(1, ...)`: `EngineColoringDeployer` puts an `EngineColoringPass` at index 0,
   so nodes are coloured, while appending (as the other NE16 passes do) would land after *all*
   coloring and after NHWC.
2. **The node must still be a `Conv`.** streamin forces int32 output, so the RequantShift stays a
   separate cluster node; the existing `ne16_taps` guard in `PULPConvRequantMergePass` fires
   correctly because index 1 precedes every lowering pass.

## 5. **The full-network NE16 path is NUMERICALLY WRONG**

> This section supersedes an earlier, softer reading. The divergence is not "unestablished
> equivalence" — the NE16 result is **wrong**, and the harness does not catch it.

The update fixture ships **host reference losses**, which settles it:

| pass | NE16 run | cluster baseline | host reference |
|---|---|---|---|
| L+ #0 | **1.8863217** | 1.2166961 | **1.2166874** |
| L− #0 | **0.4562725** | 0.2909873 | **0.2909976** |
| L+ #1 | **0.1412244** | 0.1887053 | **0.1902758** |
| L+ #3 | **2.1837227** | 1.6105008 | **1.5945725** |

The cluster baseline tracks the host reference to ~1e-5 — the known fp32 last-ulp behaviour. The
NE16 run is **20–55 % off**. It is wrong.

**Why the harness still reports `PASSED`.** `Errors: 0 out of 8` checks the *update* graph, which is
driven by the reference `loss_plus` / `loss_minus` arrays from the fixture — not by the losses the
device just measured. The check is structurally blind to the forward pass. Any future NE16 work on
the full network must compare the device losses against `speechnet_qzo12_update/outputs.npz`, not
rely on the harness verdict.

### Bisection

`QW_NE16_ONLY` (a diagnostic env var in `Prepare1xKPass.py`, off by default) restricts NE16 to one
node so the network can be bisected:

| configuration | first L+ | verdict |
|---|---|---|
| block 3 only on NE16 | 1.8192 | wrong |
| block 4 only on NE16 | 1.2584 | wrong (mildly) |
| cluster only | 1.2167 | correct |

**Both convolutions are independently wrong in-network**, and block 4's case is the sharper clue:
with only block 4 on NE16, blocks 0–3 are all cluster, so block 4 receives a **known-correct input**
and still produces a wrong result — while the same convolution is bit-exact in isolation.

### What has been ruled out

| hypothesis | evidence |
|---|---|
| Layout of the un-merged RequantShift | lowered graph: conv → NHWC int32 → `Transpose` → NCHW → RQS tagged `channels_first=1`. Correct. |
| Perturbation RNG | `tile_seed_offset = 0` for every perturb node, `node_id` from the stable `idx` attribute. Identical directions in both runs. |
| Requant rounding | `b3_ref` (merged) carries `rqs_add = 32700 + 32768`, `b3_plain` (un-merged) the raw `32700`, and **both score 0/1280 against the same golden** — the fused kernel truncates, `RequantShift_s32_s8_NCHW` rounds internally (`rounding = 1` in the emitted call), and the conventions cancel. |
| Requant call arguments | verified against `RequantShift.h:105` — `(…, log2D=16, HW=40, …, rounding=1)` for block 3, i.e. `div = 65536`, `8×5 = 40`. Correct. |
| **Spatial tiling** | the single-layer fixture forced to the **same 4-way tiling** (`--l1 20000 / 12000 / 8000`, `numTiles = {0,4}`) is still **`0 / 1280`**. |
| Output-channel tiling | `nKo` / `bKo` are scalars, not per-tile arrays — `Cout` is not split, so the missing per-channel weight offset is not being exercised. |

### A structural observation that matters

**The cluster has no int8 standalone `Conv` binding** — only `PULPFPConv2DParser` / `PULPFPDWConv2DParser`.
An attempt to force the same un-merge on the cluster fails to bind. So the un-merged
`Conv(int8) + RequantShift` path **exists only on NE16**, and has never been checked against an
independent implementation of the same structure: the single-layer goldens come from
`run_onnx_graph` on that same un-merged graph. That is a weaker check than it looked.

### Next step

Dump the encoded weight and block 3's int32 conv output from the **full network** and diff them
against host-computed values (`weight_enc_golden.npz` already exists for the isolated case). The
leading hypothesis is now that something environmental — buffer lifetime or aliasing of
`weight_enc` / the int32 intermediate under full-network memory pressure — corrupts an input the
isolated fixture never stresses.

## 6. OPEN — padded convolutions are refused

`ne16_1xkAdmissible` rejects any conv with non-zero pads, so SpeechNet blocks 0/1/2 (`[0,2,0,2]`,
`[0,8,0,8]`, `[0,4,0,4]`) stay on the cluster and only **5.8 %** of conv MACs reach NE16.

The refusal is deliberate and is a **correctness** guard, not conservatism: NE16's 1×1 mode cannot
pad, and its guard in `gvsoc fsm.cpp:53` is **commented out**, so a padded job returns wrong results
rather than trapping. Every fixture so far pre-padded the activation *data*; a real graph carries
the padding on the node.

The GAP9 SDK solves this with **pointer arithmetic plus border-subtile computation**
(`NE16_ComputeBorders`, `CNN_BasicKernels_NE16.c`) rather than with NE16 padding — that is the shape
of the fix, and it belongs in `NE161xKConstraint.serializeTilingSolution`. Lifting it would take
block 1 (68 % of conv MACs on its own) to NE16.

## 7. Artefacts

```
Plan.md  Findings.md  results/results.json
fixture/TrainingNetwork_qzo_ne16.c   generated C for the full QZO step with NE16 engaged
logs/phase1_block{0..4}_pipelined.log
logs/plain3.log, logs/b4_plain.log   the automatic-pass runs
logs/qzo_full2.log                   full SpeechNet QZO with NE16
logs/qzo_baseline.log                full SpeechNet QZO, cluster only
```
