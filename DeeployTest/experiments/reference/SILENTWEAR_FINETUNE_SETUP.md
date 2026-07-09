# SilentWear Fine-Tuning Setup — Reference

Authoritative summary of the **SilentWear** (host/offline, full-precision) fine-tuning protocol
for the SpeechNet EMG gesture classifier, extracted directly from the SilentWear repository.
This is the *reference* protocol our on-device fine-tuning is compared against.

> Sources (all paths under `SilentWear/SilentWear/`):
> - `config/paper_ft_config.yaml` — FT hyperparameters
> - `config/models_configs/speechnet_config.yaml` — model + base train_cfg
> - `config/paper_models_config.yaml` — data / window / CV settings
> - `offline_experiments/IV_inter_session_with_ft.py` — FT driver (data selection, splits, loop)
> - `offline_experiments/Model_Fine_Tuner.py` — fine-tuner (loads weights, fits all layers)
> - `models/TorchTrainer.py` — training loop, optimizer, dataloaders

---

## 1. Model

SpeechNet (`speechnet_config.yaml`), 5 conv blocks → global pool → linear classifier:

| Block | out_channels | kernel | pool (block-level) |
|------:|-------------:|--------|--------------------|
| 1 | 8  | [1, 4]  | [1, 8] |
| 2 | 16 | [1, 16] | [1, 4] |
| 3 | 16 | [1, 8]  | [1, 4] |
| 4 | 32 | [7, 1]  | [1, 1] → Identity |
| 5 | 32 | [7, 1]  | [1, 1] → Identity |

- Each block: Conv2d → **BatchNorm2d** → ReLU → pool.
- Block pooling is **MaxPool** for blocks 1–3 (size > 1), Identity for blocks 4–5.
- `global_pool: avg` → GlobalAveragePool over the 32-channel feature map.
- `p_dropout: 0.5` before the final Linear(32 → num_classes).
- `num_classes = 9` (8 vocalized gestures + **rest**; `include_rest: true`).

**Input:** `(N, 1, 14, 700)` — 14 EMG channels (`channel_order` lists 14 indices),
`window_size_s = 1.4` at 500 Hz → 700 time steps. Condition: **vocalized**, subject **S01**.

---

## 2. Cross-validation / held-out structure

- `cv.global_cv_mode: leave_one_batch_out`, `n_splits: 5` (a "batch" = one recording
  session/day). Base inter-session models are trained leaving one session out; the held-out
  session is then used for fine-tuning + evaluation.
- `experiment.seed: 42`, `val_size: 0.2` (for base training).
- The deployed/held-out fold we compare against is `leave_one_session_out_fold_3`
  (S01 / vocalized / `w1400ms` / `model_1`).

---

## 3. Fine-tuning hyperparameters (the headline numbers)

| Setting | Value | Source |
|---|---|---|
| **Batch size** | **32** | `TorchTrainer.create_dataloader_from_df(batch_size=32)`, used by `fit()` |
| **Optimizer** | **Adam** | `speechnet_config.yaml` `optimizer_cfg.name: adam` → `torch.optim.Adam` |
| **Learning rate** | **1e-3** | `paper_ft_config.yaml` `ft_lr: 1.0e-3` |
| **Epochs** | **50** | `paper_ft_config.yaml` `num_ft_epochs: 50` |
| **Weight decay** | **1e-4** | `speechnet_config.yaml` `weight_decay` |
| **LR scheduler** | ReduceLROnPlateau (factor 0.1, patience 2, mode min) | `speechnet_config.yaml` |
| **Early stopping** | patience 10 (on val loss) | `speechnet_config.yaml` `early_stop_patience` |
| **Loss** | CrossEntropyLoss | `TorchTrainer` |
| **Trainable params** | **ALL layers** (full fine-tuning, no freezing) | `Model_Fine_Tuner`: "fine tune all layers" |
| **Eval metric** | **balanced accuracy** | driver `metrics["balanced_accuracy"]` |

---

## 4. Data selection per fine-tune round

Per held-out session, the driver iterates over that session's recording **batches** and
fine-tunes **progressively** (each batch's FT starts from the previous batch's fine-tuned model;
the first starts from the base inter-session model):

```
model_to_ft = base_model                 if batch_id == first
            = fold_{id}_ft_{batch_id-1}  otherwise      # progressive/cumulative FT
```

For each batch (`IV_inter_session_with_ft.py:265–283`):

1. **Rest-class downsampling ONLY.** `min_samples = value_counts().min()` (size of the smallest
   class). The **rest** class is randomly downsampled to `min_samples`; all non-rest classes are
   kept in full. → class balance is fixed only by trimming the over-represented rest class.
2. **Stratified train/val split**, `test_size = 0.3`, `shuffle=True`, `random_state=42`,
   `stratify=Label_int` → **70% train / 30% val**, label-stratified.
3. Fine-tune on the 70% train split, validate on the 30% val split (early stopping + scheduler
   use val loss).

So a SilentWear FT round trains on **~70% of the (rest-balanced) windows of one session batch**
— i.e. effectively all the available adaptation data for that batch, not a small subsample.

---

## 5. BatchNorm handling (the part that matters for on-device comparison)

SilentWear fine-tunes **all layers with `model.train()`** (`TorchTrainer` epoch loop). Therefore
BatchNorm runs in **standard training mode**:

- **Forward** normalizes with the **current mini-batch (32 windows) statistics** — a reasonable
  estimate of the population.
- **Running stats are updated** every step via momentum EMA.
- At evaluation (`model.eval()`) BN uses those **updated running stats**.

This is the textbook-correct BN training path. It works precisely because the batch is 32 windows
and the running-stat update is active — **neither of which holds on-device** (see
`BN_AND_NACCUM_FINDINGS.md`).

---

## 6. One-line characterization

> SilentWear = **full** fine-tuning of **all** layers, **Adam** (lr 1e-3, wd 1e-4) + ReduceLROnPlateau
> + early stopping, **batch 32**, **50 epochs**, on **~70% of one rest-balanced session batch**,
> **progressively** across batches, with BatchNorm in **normal training mode** (32-window batch
> stats + running-stat updates), scored by **balanced accuracy**.

---

## 7. Gap vs our on-device setup (quick contrast)

| Axis | SilentWear (reference) | On-device (Deeploy/Siracusa) |
|---|---|---|
| Batch size (windows/forward) | **32** | **1** (pinned by L2 memory) |
| BN statistics | 32-window batch stats | 1-window stats (degenerate) → **folded/frozen** instead |
| BN running-stat update | yes (EMA) | none (update path dead) |
| Optimizer | Adam (+ wd, scheduler, early stop) | **plain SGD** (no momentum/Adam kernel) |
| Trainable scope | all layers | **last layer only** (head) |
| Epochs | 50 | few (bit-exact window before drift) |
| Data | ~70% of rest-balanced batch | stratified ~10% subsample (≈2/class) |
| Effective batch knob | true batch (32) | gradient accumulation `n_accum` (≠ BN batch) |

See `BN_AND_NACCUM_FINDINGS.md` for why the batch-size / BN gap cannot be closed with
gradient accumulation, and why folding BN is the resolution.
