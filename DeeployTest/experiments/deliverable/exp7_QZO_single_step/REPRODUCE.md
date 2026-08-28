# exp7 — Quantized ZO (QZO): single-step on-device smoke test — **BIT-EXACT**

Single-step on-device **quantized zeroth-order (MeZO)** training probe of SpeechNet on Siracusa/GVSoC.
One antithetic pair (θ+εz forward → L+, θ−εz forward → L−) runs fully on device; the device losses are
compared against the host reference (`run_onnx_graph` on the identical graph, baked into `outputs.npz`).

Datapath: input fp32 → **Quant(online)** → `Conv int8 ×5` (2-input, per-output-channel int8 weight + int32
bias folded into the requant `add`, both **perturbed on-device** by `RQSPerturbRademacher`) → per-channel
`RequantShift` → `Dequant` → fp32 `BatchNormInternal` (frozen running stats; γ/β perturbed by float
`PerturbRademacher`) → `ReLU` → `MaxPool` → … → **float fc** (3-input `Gemm`, weight+bias float-perturbed —
the shipped QMCUNetZO convention: quantize the Conv layers only) → `SoftmaxCrossEntropyLoss`.
22 trainable parameters are **graph inputs** (weights-as-inputs); real SilentWear window + real PTQ
calibration; pretrained fold-3 checkpoint.

## Result

```
[loss+ 0] computed=1.245066  ref=1.245066  diff=0.000000  TOL=0.001000
[loss- 0] computed=1.277979  ref=1.277979  diff=0.000000  TOL=0.001000
Errors: 0 out of 2
✓ Test speechnet_qzo_train PASSED - No errors found
```

**Device L+ and L− are bit-exact to the host reference** (`fixture/outputs.npz`: `loss_plus=1.245066`,
`loss_minus=1.277979`, `grad=-1.645637`). Log: `logs/sim.log` (device run) · `logs/export.log` (fixture export).

## Fixture contents

| file | role |
|---|---|
| `network_zo_train.onnx` | the ZO **training** graph the device compiles+runs (packed as `speechnet_qzo_train/network.onnx`) — antithetic forward with on-device perturbation + SCE loss; 24 inputs = `input`, `label`, 22 trainable params |
| `network_zo_update.onnx` | the ZO **weight-update** graph (packed as `speechnet_qzo_update/network.onnx`) — 22 params in → 22 `*_updated` out |
| `inputs.npz` | `arr_0000..arr_0023` = input window, label, 22 initial params — **positional keys in graph-input order** |
| `outputs.npz` | host reference (`run_onnx_graph` on the identical zo_train graph): `loss_plus/loss_minus/grad/log_prob` |

NOTE: the export (Step 1) also emits an intermediate `network.onnx` (the raw `create_quant_pipeline` base
graph the two ZO graphs are derived from). It is **not consumed** by the device run and is not a validated
standalone graph, so it is deliberately **not part of this fixture**.

## Prerequisites

- Branch **`feat/QZO`** in BOTH repos, at least:
  - `TrainDeeploy` commit **`7c22a0e`** (RQSPerturb per-channel `/` indexing, QuantParser `1/scale`,
    square-padding guard re-commented, `_ONNX_ELEM_TO_PTR` input typing)
  - `Onnx4Deeploy` commit **`a819f19`** (`arr_NNNN` graph-order npz keys, 4-D calibration shape,
    per-output-channel host perturb, 2-input conv + float fc)
- Containers (both mount the same host tree `/Users/qiwenwu/ETH`):
  - **`agitated_hugle`** — export (has onnxruntime + brevitas + vendored DeepQuant). Mount: host
    `/Users/qiwenwu/ETH` → `/app` (so Onnx4Deeploy is `/app/Onnx4Deeploy`).
  - **`traindeeploy`** — Deeploy codegen + GVSoC sim. Mount: host `/Users/qiwenwu/ETH` → `/app/ETH`
    (so TrainDeeploy is `/app/ETH/TrainDeeploy`).
- Data/checkpoint (inside `agitated_hugle`):
  - checkpoint `/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`
  - dataset `/app/SilentWear/SilentWear_data/data_raw_and_filt`

## Quick reproduce (from the fixture shipped in this folder — skips the export)

```bash
# host: copy this fixture into the runner's test dirs (same as pack_2step_fixture.py does)
cd /Users/qiwenwu/ETH/TrainDeeploy/DeeployTest
mkdir -p Tests/Models/Training/SpeechNet/speechnet_qzo_train Tests/Models/Training/SpeechNet/speechnet_qzo_update
cp experiments/deliverable/exp7_QZO_single_step/fixture/network_zo_train.onnx  Tests/Models/Training/SpeechNet/speechnet_qzo_train/network.onnx
cp experiments/deliverable/exp7_QZO_single_step/fixture/network_zo_update.onnx Tests/Models/Training/SpeechNet/speechnet_qzo_update/network.onnx
for d in speechnet_qzo_train speechnet_qzo_update; do
  cp experiments/deliverable/exp7_QZO_single_step/fixture/inputs.npz  Tests/Models/Training/SpeechNet/$d/
  cp experiments/deliverable/exp7_QZO_single_step/fixture/outputs.npz Tests/Models/Training/SpeechNet/$d/
done
```
Then jump to **Step 3**.

## Full reproduce (from export)

### Step 1 — export the QZO fixture (container `agitated_hugle`)

```bash
docker exec -it agitated_hugle bash
cd /app/Onnx4Deeploy
find onnx4deeploy -name "__pycache__" -type d -exec rm -rf {} +   # bust stale bytecode (macOS Docker FS cache)
PYTHONDONTWRITEBYTECODE=1 python3 Onnx4Deeploy.py -model SpeechNet -mode q-zo-train \
  --noise-type rqs_rademacher \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --condition vocalized --batch 1 \
  -o /app/Onnx4Deeploy/QZO_exp/exp2
```

Produces in `/app/Onnx4Deeploy/QZO_exp/exp2/`: `network_zo_train.onnx`
(24 inputs = input + label + 22 params; RQSPerturbRademacher×10, PerturbRademacher×12), `network_zo_update.onnx`
(22 in / 22 out), `inputs.npz` (**positional `arr_0000..arr_0023` keys in graph-input order** — required by the
runner), `outputs.npz` (host `loss_plus/loss_minus/grad/log_prob`), plus an intermediate `network.onnx`
(the `create_quant_pipeline` base graph — not consumed further, see Fixture contents). The export log must show real-data
calibration (NO "using random calibration" line) and ends with
`L+=1.245066  L-=1.277979  grad=-1.645637` (fixed seed 42 / ε 0.01 baked at export).

### Step 2 — pack into the runner's fixture dirs (container `traindeeploy`)

```bash
docker exec -it traindeeploy bash
cd /app/ETH/TrainDeeploy/DeeployTest
python3 experiments/zo_smoke/pack_2step_fixture.py \
  /app/ETH/Onnx4Deeploy/QZO_exp/exp2 \
  Tests/Models/Training/SpeechNet speechnet_qzo_train speechnet_qzo_update
```

### Step 3 — codegen + GVSoC sim (container `traindeeploy`)

```bash
docker exec -it traindeeploy bash
# kill orphan gvsoc by explicit PID (killall/pkill-by-name do NOT work) + clean build tree
pgrep -f gvsoc_launcher | xargs -r kill -9
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployMezoRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_qzo_train \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo_update \
  --n-steps 1 --n-accum 1 --num-data-inputs 2 \
  --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2
```

Expected tail of the output = the **Result** block above (`diff=0.000000`, `PASSED`).

## Flag notes

- `--num-data-inputs 2` — count of per-mini-batch data inputs (`input` + `label`); the other 22 graph inputs
  are the persistent trainable-weight buffers (aliased between zo_train and zo_update **by tensor name**).
- `--eps 0.01 --seed 42` must equal the values baked at export (perturbation magnitudes `*_pmul` and the
  reference losses depend on them).
- `--l1 128000` — the runner default (64000) is too small: the block-0 conv tile pattern needs ≈88 KB
  ("geometrical constraints infeasible" from the tiler means MEMORY here, not geometry).
- `BN_FROZEN_STATS=ON` is passed automatically by the MeZo runner (`testUtils/trainingUtils.py`) — the
  device BatchNormInternal must use the frozen pretrained running stats, like the host reference.

## Caveats

- **macOS Docker FS cache**: after editing any TrainDeeploy `.py`/`.c` on the host, `docker restart
  traindeeploy` before re-running (the container serves stale file content otherwise). For Onnx4Deeploy
  edits, clear `__pycache__` + `PYTHONDONTWRITEBYTECODE=1` in `agitated_hugle`.
- GVSoC orphans starve every new sim — always kill by explicit PID (see Step 3) before a run.
- Full debugging history (the 5 device-vs-host root causes: npz key order, Quant scale convention,
  calibration fallback, graph-input typing, RQSPerturb channel indexing):
  `Onnx4Deeploy/QZO_exp/Report.md`, iterations 12–20.
