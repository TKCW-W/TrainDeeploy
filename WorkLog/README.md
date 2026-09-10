# WorkLog — QZO / ZO on-device training: handoff for the next agent

Purpose: bring a fresh agent up to speed on the **zeroth-order (ZO / MeZO) and quantized-ZO (QZO)
on-device fine-tuning of SpeechNet** project, end to end, so it can continue (next target: **NE16
accelerator on GAP9**) without re-deriving the pipeline or tripping on stale settings.

> **Read this whole file first.** Several settings CHANGED across experiments. Every place that
> matters is marked **CURRENT** (use this) vs **SUPERSEDED** (do NOT reuse). When in doubt, the
> newest experiment wins: **exp12** (QZO round-1, Siracusa) and **exp13** (QZO on GAP9) are the
> canonical current references; **exp15** (ECO) is the latest host-sim study.

Auto-loaded memory (`~/.claude/.../memory/MEMORY.md`) indexes deeper notes; this README is the
operational map. Branch for ALL work: **`feat/QZO`** in both repos.

---

## 0. The two repos + one data repo

| repo | role | branch |
|---|---|---|
| `Onnx4Deeploy` | PyTorch/Brevitas → ONNX **export** (fixtures + host reference sim + calibration) | `feat/QZO` |
| `TrainDeeploy` | vendored **Deeploy** — device **compile + GVSoC run** (Siracusa, GAP9) | `feat/QZO` |
| `SilentWear` | EMG dataset + pretrained SpeechNet checkpoints (read-only data source) | — |

Standalone `ETH/Deeploy` and `ETH/Onnx4Deeploy_ZO` are **READ-ONLY reference mirrors** — never edit.

---

## 1. Containers (each has a DIFFERENT mount path — a common trap)

All three are Docker containers, currently running. **The repo path prefix differs per container:**

| container | image | role | mount → in-container path |
|---|---|---|---|
| **agitated_hugle** | `python:3.10` | **host** export + calibration + PyTorch/Brevitas host-sim | `ETH → /app` (so `Onnx4Deeploy` = `/app/Onnx4Deeploy`, `TrainDeeploy` = `/app/TrainDeeploy`, `SilentWear` = `/app/SilentWear`) |
| **traindeeploy** | `deeploy:traindeeploy-dev` | **Siracusa** device build + GVSoC run (pulp-sdk + LLVM) | `ETH → /app/ETH` (so `TrainDeeploy` = `/app/ETH/TrainDeeploy`) |
| **deeploy_gap9** | `ghcr.io/runwangdl/deeploy:gap9` | **GAP9** device build + GVSoC run (private GAP9 SDK; amd64 under qemu) | `TrainDeeploy → /app/Deeploy` |

- `agitated_hugle` has **brevitas 0.13** (needed for `create_brevitas_model`); `traindeeploy`/`deeploy_gap9` do NOT.
- **deeploy_gap9 setup** (required before any GAP9 command; the image is private — the container
  is already running, but if it must be re-created: `docker login ghcr.io -u TKCW-W` with a classic
  PAT scoped `read:packages`, then `docker run -dit --name deeploy_gap9 -v $(pwd)/..:/app/Deeploy
  ghcr.io/runwangdl/deeploy:gap9 tail -f /dev/null` from the TrainDeeploy dir, then
  `pip install -e .` inside). Every GAP9 exec must first:
  ```
  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
  ```
- **Before every GVSoC run, kill orphan sims** (they starve new runs; killall/pkill-by-name do NOT
  work): `pgrep -f "[g]vsoc_launcher" | xargs -r kill -9`.
- **When switching a fixture's kind** (train↔inference) reuse of a build dir errors — `rm -rf TEST_SIRACUSA` / `rm -rf TEST_GAP9` first.

---

## 2. Data & artifacts (SilentWear)

Paths as seen in **agitated_hugle** (prefix `/app/SilentWear/...`):

- **Dataset:** `/app/SilentWear/SilentWear_data/data_raw_and_filt`
- **Pretrained checkpoint (the one we use):**
  `/app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt`
  (key `model_state_dict`). This is **fold 3** = leave-out session 3; near-converged inter-session model.
- **Protocol we standardize on:** subject S01, condition vocalized, session 3. Fine-tune on 54
  stratified windows (30% of 180, seed 42) of a batch; evaluate on the WHOLE next batch (180 windows).
  Pretraining calibration data = the 1800 windows of S01/vocalized sessions 1+2 (batches 1–5).

---

## 3. The pipeline — Onnx4Deeploy → TrainDeeploy

Same 5 stages for float ZO, QZO/Siracusa, QZO/GAP9; only the mode, calibration, lr, and target differ.

```
[export in agitated_hugle]                    [device build+run in traindeeploy / deeploy_gap9]
 (1) calibration + PyTorch host reference  →  (3) pack fixture into Tests/Models/Training/SpeechNet
 (2) Onnx4Deeploy.py -mode <...> export     →  (4) deeployMezoRunner_tiled_<target>.py  (GVSoC)
                                               (5) dump weights → build int8 inference fixture → eval
```

`Onnx4Deeploy.py -mode` values (verified in `Onnx4Deeploy.py`/`base_exporter.py`): `infer`, `train`,
**`zo-train`** (float ZO), **`q-zo-train`** (quantized ZO). QZO export is `_export_qzo_training`.

### 3a. FLOAT ZO (baseline / upper bound)
- **Export** (agitated_hugle, `/app/Onnx4Deeploy`): `-mode zo-train` (NO calibration, NO
  `QZO_POOLED_THRESHOLDS`). All conv w/b + BN γ/β + fc w/b trainable, eval-mode (frozen) BN.
- **lr = 3e-6** (CURRENT float-ZO lr). Fixtures: `speechnet_zo_train` / `speechnet_fzo_train`.
- **Device run** (traindeeploy): `deeployMezoRunner_tiled_siracusa.py` on the float fixture.
- Float ZO is bit-exact within tolerance on device (its update `θ−lr·g·z` is smooth; no `round()`).

### 3b. QZO on **Siracusa** — the canonical flow (follow exp12 exactly)
Authoritative repro: `TrainDeeploy/DeeployTest/experiments/deliverable/exp12_QZO_clean_round_1/Findings.md`.
CURRENT fixture = **`speechnet_qzo12_{train,update}`** (fresh pooled@99.99, fold3). Steps:

1. **Calibration + fc-float PyTorch reference** (agitated_hugle):
   `exp12.../run_ref_and_calib.py` → `fixture/pooled_9999_fold3_fresh.json` + ref **89.44%** (fc-float, lr 1e-5).
2. **Export fixture** (agitated_hugle, `/app/Onnx4Deeploy`):
   ```
   QZO_POOLED_THRESHOLDS=<...>/pooled_9999_fold3_fresh.json python3 Onnx4Deeploy.py \
     -model SpeechNet -mode q-zo-train --noise-type rqs_rademacher --dataset silentwear \
     --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
     --pretrained-weights <CKPT above> --subject S01 --session 3 --condition vocalized \
     --batch 1 --data-size 54 --stratified --n-accum 4 --lr 1e-5 -o QZO_exp/exp12_clean
   ```
3. **Pack** into the device test tree (agitated_hugle or traindeeploy):
   ```
   python3 DeeployTest/experiments/zo_smoke/pack_2step_fixture.py QZO_exp/exp12_clean \
     DeeployTest/Tests/Models/Training/SpeechNet speechnet_qzo12_train speechnet_qzo12_update
   ```
4. **Device round on GVSoC/Siracusa** (traindeeploy, `/app/ETH/TrainDeeploy/DeeployTest`):
   ```
   python3 deeployMezoRunner_tiled_siracusa.py \
     -t Tests/Models/Training/SpeechNet/speechnet_qzo12_train \
     --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo12_update \
     --n-steps 2700 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
     --l1 128000 --l2 2000000 --cores 8 -D BN_FROZEN_STATS=ON DUMP_WEIGHTS=ON
   ```
   (`-D BN_FROZEN_STATS=ON` is **required** — the graph carries `BatchNormInternal`, a flag-gated
   frozen-stats op; without it BN uses per-window batch stats → wrong. `--n-steps 1` = single-step
   smoke; `2700` = full round-1.)
5. **Eval**: extract dumped int8 weights → `build_qzo_infer_fixture11.py` → untiled batch-2 inference.
   exp12 on-device eval = **88.33%** (fc-float, ffast-math).

### 3c. QZO on **GAP9** — DONE, passes (exp13)
Same fixture (`speechnet_qzo12_{train,update}`) and export as 3b. Only the device stage changes:
runner + container + two GAP9-specific memory settings. Authoritative:
`exp13_QZO_GAP9_single_step/Findings.md`.

```
docker exec deeploy_gap9 bash -lc '
  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
  pgrep -f "[g]vsoc_launcher" | xargs -r kill -9 ; rm -rf /app/Deeploy/DeeployTest/TEST_GAP9
  cd /app/Deeploy/DeeployTest
  python3 deeployMezoRunner_tiled_GAP9.py \
    -t Tests/Models/Training/SpeechNet/speechnet_qzo12_train \
    --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_qzo12_update \
    --n-steps 1 --n-accum 4 --num-data-inputs 2 --eps 0.01 --lr 1e-5 --q 1 --seed 42 \
    --l1 110000 --l2 1500000 --defaultMemLevel L2 -D BN_FROZEN_STATS=ON'
```
Result: `update (zo_update) DONE; Errors: 0 out of 8; PASSED`.
- **CURRENT GAP9 musts:** `--l1 110000` (NOT 128000) and slave stacks relocated to L2 (already in
  `Platforms/GAP9/src/deeploymezotest.c`). See §5 for why.

---

## 4. CURRENT vs SUPERSEDED settings (consistency table — READ THIS)

| topic | **CURRENT (use)** | SUPERSEDED (do NOT reuse) |
|---|---|---|
| Activation calibration | **pooled@99.99** (percentile over 1800 pretrain windows; `calib_pooled.py`) | `old8` / `old54` (per-batch EMA on FT windows); abs-max; other percentiles |
| Weight scale | per-channel **abs-max**, frozen at export | abs-percentile (tested, does NOT help) |
| fc head on device | **fc-float** (`build_int8_forward`: dequant→float fc); device-faithful ref = **88.33–89.44%** | fc-int8 (gave ~90%, NOT device-faithful) — note exp15 host sim still uses fc-int8 |
| Direct-int8 lr | **1e-5** (overcomes LSB stall) | 3e-6 (stalls: 0% conv weights move; = BN-only ~85%) |
| Float-ZO lr | **3e-6** | — |
| Fixture | **`speechnet_qzo12_*`** (exp12, fresh pooled@99.99) | qzo8/qzo9/qzo11 (older calibration/scale-bug era) |
| fp build flags | **ffast-math default** (learning-neutral) | strict-fp32 = *debug tool only* for bit-exactness, not accuracy |
| GAP9 `--l1` | **110000** (with stacks→L2) | 128000 (fatal on GAP9: usable L1 ~93KB); 90000 (forwards only) |
| Accuracy figures | exp12 device **88.33%**; exp15 multiseed ECO+SR **86.35±0.59**, direct@1e-5 **86.35±1.08**, ECO mf **85.73±0.92** | 83.89→87.22 (exp10-era, `qzo_weight_integerize.py` comments); single-seed "+0.84" ECO claim |

---

## 5. Important fixes & findings (each was a real bug or a settled conclusion)

**Forward-fidelity / accuracy (host + device):**
- **Per-layer RequantShift scale bug** (`Onnx4Deeploy/onnx4deeploy/transform/qzo_weight_integerize.py`,
  fixed): `build_int8_forward` used a uniform 1/128 requant ratio instead of per-layer traced scales.
  Harmless at old uniform calibration, but under pooled@99.99 it distorted every block (cos 0.87 vs
  Brevitas, ~chance) → **all device runs before this fix used a wrong forward**. Fixed → cos 0.9994.
- **Requant rounding = truncate vs round-to-nearest** (same file, ~line 375): the requant
  `(acc·mul+add)>>16` truncates; round-to-nearest needs `+div/2` (=32768) in `add`. Deeploy's merge
  pass bakes `+div/2` only for a *constant* bias; our *perturbable* bias skipped it → floored. Floor's
  −0.5 LSB is invisible to inference but NOT antithetic-symmetric → it survives ZO's `L+−L−` and
  **sign-flips the gradient** (`g −1.81 → +17.06`). Fix: bake `+div/2` into the bias initializer.
- **BN_FROZEN_STATS**: `BatchNormInternal` is flag-gated; device training/inference of the QZO graph
  needs `-D BN_FROZEN_STATS=ON` or BN uses single-window batch stats (wrong).

**Bit-exactness (device vs host reference):**
- Root cause of per-step device≠host = fp32 non-associativity under `-ffast-math` (FMA + reassoc in
  `BatchNorm.c`/`Gemm.c`) + SCE `expf`/`logf` (picolibc vs numpy), ≤1 ulp, amplified by `round()`.
  Fix for a bit-exact *debug* build = `DEEPLOY_STRICT_FP32(_FILES)` (`-fno-fast-math -ffp-contract=off`
  on those files) → full round-1 0/21600. **ffast-math is learning-neutral** (88.33 vs 89.44 = noise),
  so it is the deployment default; bit-exactness is a bring-up gate, not an accuracy requirement.

**LSB stall (the core QZO issue):**
- At small lr the int8 conv update `round(coeff·z/s_w)` rounds sub-LSB steps to 0 → conv weights never
  move (stall). Fixes: lr 1e-5 (strong-signal filter), OR master weights (fp32 shadow), OR **ECO**
  (exp15 — error-feedback via SGD momentum, no master buffer). Calibration does NOT fix it.

**GAP9 device port (exp13) — ~12 backward-compatible fixes** (all committed, none affect Siracusa):
runner PATH/kconfiglib, `perf_utils.h` counter macros, `math.h`/`lrintf`, ZO-runtime includes in
`DeeployGAP9Math.h`, Mchan API via `mchan_v7.h` (GAP9 IS Mchan v7), `-Wno-error` on optimizer_network,
cluster-task `void(void*)` wrappers, drop `fflush`/`stdout` (pmsis printf). **Two runtime memory
levers**: GAP9 usable L1 is ~93KB (not 128KB) because the ~30KB cluster slave stacks live in L1
(Siracusa's lighter runtime leaves ~128KB free) → **relocate slave stacks to an L2 buffer** via
`pi_cluster_task_stacks()` (frees ~30KB → usable ~123KB) so `--l1 110000` fits. MaxPool
`_HWC` kernel-name warning is **verified benign** (correct kernel defined+linked; args all
pointer/int; 0/8 pass).

---

## 6. Experiments index (path · goal · current-or-superseded)

**Onnx4Deeploy/QZO_exp/** (host PyTorch/Brevitas studies):
- `exp_calibration/` — **CURRENT** calibration authority. Built pooled@99.99 (best, 85.56% zero-shot);
  proved LSB stall is calibration-invariant. Has `run_study.py`, `calib_pooled.py`, `run_incremental*.py`,
  `run_round1.py`, `results.json`. Reusable infra for host sims.
- `exp15_QZO_eco/` — **CURRENT / latest host study.** ECO (error-feedback via momentum) transferred to
  QZO. Findings: ECO+SR & exact-EF match/beat direct@1e-5 at lr 3e-6 with no master buffer; ablation
  proves EF (not momentum) breaks the stall; `cos(e)=0.998` (memory-free heuristic holds). Includes the
  matched multiseed + a `_verify_faithful.py` (direct_1e5 reproduces the sim bit-identically) + a HW
  implementation plan in `Findings.md`. Runs in **agitated_hugle**.
- `exp14_fold_BN/` — folded-BN sim study (fold needs exact-max calib; direct-int8 never beats zero-shot
  under scale-invariant ZO). Context, not a shipped result.
- `exp12_clean` / `exp9*` / earlier expN — **older**; exp12_clean is the fixture source for exp12/exp13.
  Do NOT cite exp8/9/10-era accuracy numbers (SUPERSEDED; forward-scale bug era for some).

**TrainDeeploy/DeeployTest/experiments/deliverable/** (device):
- `exp12_QZO_clean_round_1/` — **CURRENT** canonical QZO/Siracusa reference. Clean-room full pipeline;
  device 88.33% == host-executor (ffast-math learning-neutral). `Findings.md` has the 5-part flow. The
  `speechnet_qzo12_*` fixture originates here.
- `exp13_QZO_GAP9_single_step/` — **CURRENT** GAP9 port. Single-step QZO **PASSES on GVSoC/GAP9**
  (Errors 0/8). `Findings.md` = the full build/run journey + GAP9 fixes + `--l1 110000`/stacks-L2.
  Logs: `gap9_PASS_l1_110000_stacksL2.log` (the pass).
- `exp10_QZO_single_step_profiling/` — QZO step latency profiling on Siracusa (Quant/Dequant QCDQ was
  the bottleneck; fixed kernels → QZO ~4.6× faster than float ZO). Latency reference.
- (exp1–exp11, exp11_bitexact_SCE*) — earlier BP/ZO/bit-exactness studies; consult only for the specific
  topic in their name; accuracy numbers may be superseded.

---

## 7. Next target: NE16 accelerator on GAP9

See memory `ne16-gap9-training-study` + guide `ETH/docs/TRAIN_GAP9_NE16/`. Scope = **QZO only** (BP out
of scope). The GAP9 SDK gate is now OPEN (exp13 passes) so NE16 can build+run+measure against the
passing QZO baseline. Two blockers + the fix are pre-analyzed in that memory: (1) NE16 filter modes
(1×1/3×3) don't match SpeechNet's dense 1×4/7×1 convs → `1×k`→k pointwise + streamin; (2) runtime
perturbed weights → exploit conv linearity `conv(w+δz,x)=conv(w,x)+δ·conv(z,x)` (base = offline-encoded
constant, sign conv = 1-bit bit-serial). Acceptance: bit-exact bring-up gate, then accuracy within ~1
eval window of 88.33%.

---

## 8. Standing working constraints (from user feedback, in memory)
Act as a serious researcher, no faked results, conservative. **Comment-out-don't-delete** shipped/
original SpeechNet files (implement alongside). **Match the existing pipeline** (native ZO-graph
emission), no band-aids. **Verify before diagnosing** (run ORT/sim first; never claim absence by grep —
trace flows + inspect artifacts). Proceed autonomously; only stop for destructive/outward-facing actions.
