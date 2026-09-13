# exp6 — ZO (MeZO) single-step latency with profileTiling

**Date created:** 2026-08-20
**Branches:** TrainDeeploy `feat/BP+ZO`, Onnx4Deeploy `feat/BP+ZO` (both host and container-side confirmed).
**Platform:** Siracusa (GAP9 / PULP 8-core cluster), GVSoC simulation.

## Task & goal

Collect **latency** data for the zeroth-order (MeZO) fine-tuning path of SpeechNet on device, as a baseline
*before* we extend ZO to a quantized integer datapath. Specifically: run a **single ZO step with `n_accum = 1`**
(i.e. one accumulation window → weights updated once) under `--profileTiling`, and produce a per-tile / per-node
latency breakdown.

**Why now / what this prepares:** the quantization extension (see `Deeploy/Quantized_ZO.md`) will let us move the
integer forward onto the **NPU (N-EUREKA)** and later optimize the **perturbation** at the RTL level. To justify and
target that work we first need to know where the ZO step actually spends its cycles today — and in particular how
expensive the **perturbation kernels** (the `Perturb*`/RNG copies that generate `θ±εz`) are relative to the
Conv/forward compute. ZO is **forward-only** (two perturbed forwards + one scalar update; no backward), so the
perturbation overhead is proportionally more visible here than in BP.

## Method

Mirror the proven exp3_ZO single-step flow (bit-exact, 0/2 errors), swapping `--plotMemAlloc` for `--profileTiling`.
Two-graph flow: `zo_train` (perturbed forward → loss, run twice for +ε/−ε) + `zo_update` (in-place θ update).

- Recipe (latency is independent of lr magnitude — matched to the known-passing exp3 config): `eps=0.01`,
  `lr=0.001`, `q=1`, `seed=42`, `n_accum=1`, `n_steps=1`, `cores=8`, `L1=128000`, `L2=2000000`, `defaultMemLevel=L2`.
- Deliverables: `logs/runner.log` (full runner + GVSoC stdout, incl. the `[node][SB...]` profileTiling UART lines and
  the `BENCH ... cycles` counters), `logs/profiletiling.log` (extracted tiling lines), `Finding.md` (analysis).

## Plan / steps

1. Confirm both repos on `feat/BP+ZO` (host + container).
2. **Export** a fresh ZO fixture with Onnx4Deeploy in `agitated_hugle` (n_steps 1, n_accum 1).
3. **Pack** the export into the runner's 2-dir layout (`_train` + `_update`) in `traindeeploy`.
4. Kill any GVSoC orphans by explicit PID; `rm -rf TEST_SIRACUSA`.
5. **Run** `deeployMezoRunner_tiled_siracusa.py` with `--profileTiling` (single step), capture all output.
6. Extract the profileTiling lines + BENCH counters; write `Finding.md` with the tiling latency breakdown,
   focusing on the perturbation process.

## Reproduction — exact commands & paths

Containers: export in **`agitated_hugle`** (Onnx4Deeploy at `/app/Onnx4Deeploy`), simulate in **`traindeeploy`**
(TrainDeeploy at `/app/ETH/TrainDeeploy`, and the *same* Onnx4Deeploy tree is visible there at `/app/ETH/Onnx4Deeploy`).

**0. Confirm branches**
```bash
git -C /Users/qiwenwu/ETH/TrainDeeploy  branch --show-current      # feat/BP+ZO
git -C /Users/qiwenwu/ETH/Onnx4Deeploy  branch --show-current      # feat/BP+ZO
docker exec traindeeploy   git -C /app/ETH/TrainDeeploy branch --show-current   # feat/BP+ZO
docker exec agitated_hugle git -C /app/Onnx4Deeploy    branch --show-current    # feat/BP+ZO
```

**1. Export ZO fixture — `agitated_hugle`**
```bash
docker exec agitated_hugle bash -lc '
ln -sfn /app/SilentWear/SilentWear_data /app/SilentWear_data
cd /app/Onnx4Deeploy
CKPT=/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt
EXPORT_BASE="--noise-type rademacher --bn-frozen-stats --dataset silentwear --pretrained-weights $CKPT --subject S01 --session 3 --condition vocalized"
python3 Onnx4Deeploy.py -model SpeechNet -mode zo-train $EXPORT_BASE \
  --n-steps 1 --n-accum 1 --lr 0.001 -o ./onnx/model/speechnet_zo_exp6'
```
Output: `/app/Onnx4Deeploy/onnx/model/speechnet_zo_exp6/{network_zo_train,network_zo_update,network_infer}.onnx`,
`inputs.npz`, `outputs.npz` (22 final weights + log_prob + loss_plus/minus). Same tree from `traindeeploy`:
`/app/ETH/Onnx4Deeploy/onnx/model/speechnet_zo_exp6/`.

**2. Pack into 2-dir layout — `traindeeploy`**
```bash
docker exec traindeeploy bash -lc '
cd /app/ETH/TrainDeeploy/DeeployTest
python3 experiments/zo_smoke/pack_2step_fixture.py \
  /app/ETH/Onnx4Deeploy/onnx/model/speechnet_zo_exp6 \
  Tests/Models/Training/SpeechNet speechnet_zo_train_exp6 speechnet_zo_update_exp6'
```
Produces `Tests/Models/Training/SpeechNet/speechnet_zo_train_exp6` (24 inputs / 10 inits) and
`..._update_exp6` (22 inputs / 22 outputs).

**3. Single-step MeZO sim with profileTiling — `traindeeploy`**
```bash
docker exec traindeeploy bash -lc '
cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA
python3 deeployMezoRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_zo_train_exp6 \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_zo_update_exp6 \
  --n-steps 1 --n-accum 1 --num-data-inputs 2 \
  --eps 0.01 --lr 0.001 --q 1 --seed 42 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --cores 8 --profileTiling \
  > experiments/deliverable/exp6_ZO_single_step_latency/logs/runner.log 2>&1'
```

**GVSoC orphan hygiene** (before each run; kill by explicit PID, non-self-matching pattern):
```bash
docker exec traindeeploy bash -lc 'ps -eo pid,args | grep "[g]vsoc"'          # inspect
docker exec traindeeploy bash -lc 'pgrep -f "[g]vsoc_launcher" | xargs -r kill -9'   # kill if any
```

## Host paths (deliverables)
- Experiment dir: `TrainDeeploy/DeeployTest/experiments/deliverable/exp6_ZO_single_step_latency/`
  - `plan.md` (this file), `Finding.md` (analysis)
  - `logs/runner.log` (full runner + GVSoC stdout), `logs/profiletiling.log` (extracted `[node]...` lines)
  - `fixture/` (copy of the exported ONNX + npz for archival)
- Fixture (container): `Tests/Models/Training/SpeechNet/speechnet_zo_{train,update}_exp6`
- Export (container): `/app/Onnx4Deeploy/onnx/model/speechnet_zo_exp6` (= host `Onnx4Deeploy/onnx/model/speechnet_zo_exp6`)
