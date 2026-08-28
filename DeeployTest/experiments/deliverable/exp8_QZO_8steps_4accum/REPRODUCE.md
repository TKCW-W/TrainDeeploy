# exp8 — Quantized ZO: 8 antithetic pairs, n_accum=4 → **2 on-device weight updates**

Multi-step extension of exp7: 8 real SilentWear mini-batches run as 8 antithetic ZO pairs (θ+εz / θ−εz),
accumulated 4 pairs per update → **2 in-place weight updates executed ON DEVICE** via the zo_update graph.
Pairs 4–7 therefore run on **device-updated weights** — the first numerical validation of the on-device
QZO update path (exp7 only validated the L± forward pair).

## Result

```
[loss+ 0..7] diff = 0.000000 ×7, 0.000001 ×1 (loss+ 6)
[loss- 0..7] diff = 0.000000 ×8
Errors: 0 out of 16
✓ Test speechnet_qzo8_train PASSED - No errors found
```

15/16 device losses are print-identical to the host reference; the single `1e-6` (pair 6, TOL `1e-3`) is a
last-ulp fp32 operation-ordering difference between the device FPU kernels and numpy in that forward's float
tail (BN/GAP/fc/softmax) — characterized, not a logic gap. Host per-step projections: `g_proj = [9.4171,
15.4078]`, coefficients ≈ `−9.4e-5 / −1.5e-4`. Logs: `logs/sim.log`, `logs/export.log`.

## What is (and is not) updated — frozen quantization scales

The ZO update moves **only**: the int8 conv weights + int32 conv biases (integer grid) and the fp32 BN γ/β +
fc weight/bias. **All quantization scales are frozen at export**: activation scales (`Quant`/`Dequant`
attributes, from real-data PTQ calibration), per-channel weight scales, and their fused RequantShift
`mul[c]`/perturb `pmul[c]` constants. The integer update is
`w += round((coeff/ε)·pmul[c]) >> S` — at `lr=1e-5` this rounds to **zero** (sub-LSB stall, expected): in
this experiment the int8/int32 params stay static and the fp32 BN/fc params receive the real updates. The
planned remedy for the stall is the **master-weight** scheme (fp32 shadow copies accumulating updates,
re-quantized with the same frozen scale); adaptive scale re-calibration is out of scope.

## Update mechanism (new in this experiment)

Device loop per update step `u` (harness `deeploymezotest.c`): `seed_base = u·q`; over 4 pairs
`acc += (L+−L−)` (fp32, on cluster); `g_proj = acc/(2·ε·n_accum)`; `perturb_eps_override = −lr·g_proj`; then
the zo_update graph runs with the SAME `seed_base` (identical Rademacher z). The **integer** RQSPerturb
kernels now honor the override by scaling their baked per-channel `mul` with
`eps_scale = override / perturb_eps_baked` (`lrintf`, round-half-even) — new in TrainDeeploy `ec870e2`
(previously only the float PerturbRademacher honored the override). The host reference (Onnx4Deeploy
`e7c8ca1`) mirrors the loop bit-for-bit, including the device's fp32 operation order for `g_proj`/`coeff`
and `np.rint` for the mul scaling.

## Fixture contents

| file | role |
|---|---|
| `network_zo_train.onnx` | ZO training graph (packed as `speechnet_qzo8_train/network.onnx`) |
| `network_zo_update.onnx` | ZO weight-update graph (packed as `speechnet_qzo8_update/network.onnx`) |
| `inputs.npz` | `arr_0000..arr_0023` (mb0 window/label + 22 initial params, graph-input order) + `mb1..mb7_arr_0000/0001` (windows 1–7) + `meta_data_size/n_batches/n_accum` |
| `outputs.npz` | host reference: `loss_plus[8]`, `loss_minus[8]`, `grad[2]` (per-step g_proj), `log_prob`, `updated_*` (22 post-training params) |

## Prerequisites

Branch `feat/QZO` in both repos, at least: **TrainDeeploy `ec870e2`**, **Onnx4Deeploy `e7c8ca1`**
(on top of the exp7 baseline `7c22a0e`/`a819f19`). Containers and data paths identical to exp7
(`exp7_QZO_single_step/REPRODUCE.md`).

## Quick reproduce (from this fixture)

```bash
cd /Users/qiwenwu/ETH/TrainDeeploy/DeeployTest
mkdir -p Tests/Models/Training/SpeechNet/speechnet_qzo8_train Tests/Models/Training/SpeechNet/speechnet_qzo8_update
cp experiments/deliverable/exp8_QZO_8steps_4accum/fixture/network_zo_train.onnx  Tests/Models/Training/SpeechNet/speechnet_qzo8_train/network.onnx
cp experiments/deliverable/exp8_QZO_8steps_4accum/fixture/network_zo_update.onnx Tests/Models/Training/SpeechNet/speechnet_qzo8_update/network.onnx
for d in speechnet_qzo8_train speechnet_qzo8_update; do
  cp experiments/deliverable/exp8_QZO_8steps_4accum/fixture/inputs.npz  Tests/Models/Training/SpeechNet/$d/
  cp experiments/deliverable/exp8_QZO_8steps_4accum/fixture/outputs.npz Tests/Models/Training/SpeechNet/$d/
done
```
Then run Step 3 below.

## Full reproduce

### Step 1 — export (container `agitated_hugle`)

```bash
docker exec -it agitated_hugle bash
cd /app/Onnx4Deeploy
find onnx4deeploy -name "__pycache__" -type d -exec rm -rf {} +
PYTHONDONTWRITEBYTECODE=1 python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train \
  --noise-type rqs_rademacher \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --condition vocalized --batch 1 \
  --data-size 8 --n-accum 4 --lr 1e-5 \
  -o /app/Onnx4Deeploy/QZO_exp/exp8
```
`--data-size 8 --n-accum 4` → the exporter runs the 2-update host sim and prints per-pair
`u{step} a{accum} mb{k}: L+=… L-=…` plus per-step `g_proj`. `--lr 1e-5` MUST equal the runner's `--lr`
(the reference update depends on it). Export ends with
`✅ QZO multi-step export complete: 2 update steps × 4 accum (8 loss pairs)`.

### Step 2 — pack (container `traindeeploy`)

```bash
docker exec -it traindeeploy bash
cd /app/ETH/TrainDeeploy/DeeployTest
python3 experiments/zo_smoke/pack_2step_fixture.py \
  /app/ETH/Onnx4Deeploy/QZO_exp/exp8 \
  Tests/Models/Training/SpeechNet speechnet_qzo8_train speechnet_qzo8_update
```

### Step 3 — codegen + GVSoC sim (container `traindeeploy`)

```bash
docker exec -it traindeeploy bash
pgrep -f gvsoc_launcher | xargs -r kill -9
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployMezoRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo8_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo8_update \
  --n-steps 2 --n-accum 4 --num-data-inputs 2 \
  --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2
```

`--n-steps 2 --n-accum 4` = 2 update steps × 4 accumulated pairs = the 8 fixture mini-batches (no cycling).
Expected tail = the **Result** block above. Same caveats as exp7 (macOS Docker FS cache → `docker restart
traindeeploy` after host edits; kill orphan gvsoc by PID; eps/seed must match the export).
