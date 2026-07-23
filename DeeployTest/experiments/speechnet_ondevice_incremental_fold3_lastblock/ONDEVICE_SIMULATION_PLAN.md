# K=1 (last-block + fc) on-device incremental simulation — procedure

**Date:** 2026-07-23
**Goal:** a complete functional on-device flow for K=1 incremental FT — from Onnx4Deeploy graph/data
preparation through tiled GVSoC training on Siracusa — for S01 vocalized fold 3, batches b1→b5, carried
forward, compared to a matched PyTorch K=1 host reference.

## K=1 scope
Trainable = 6 tensors: `blocks_4_0_weight/bias` (last conv), `blocks_4_1_weight/bias` (last BN affine),
`fc_weight/bias`. BN uses frozen pretrained stats (`--bn-frozen-stats` export ⇒ device `BN_FROZEN_STATS=ON`).

## Per-round procedure (round r: FT on batch r → eval batch r+1)
1. **Export** the fixture with Onnx4Deeploy (in the Onnx4Deeploy container):
   `python3 Onnx4Deeploy.py -model SpeechNet -mode train -o .../speechnet_train_k1_b{r}_fold3
   --dataset silentwear --subject S01 --session 3 --batch r --condition vocalized --data-size 54
   --n-epochs 40 --n-accum 4 --lr 0.001 --training-strategy custom
   --custom-trainable-params blocks_4_0_weight blocks_4_0_bias blocks_4_1_weight blocks_4_1_bias fc_weight fc_bias
   --bn-frozen-stats --stratified` with `--pretrained-weights` = previous round's carry checkpoint.
2. **Train on device** (in the TrainDeeploy container): `deeployTrainingRunner_tiled_siracusa.py
   -t Tests/Models/Training/SpeechNet/speechnet_train_k1_b{r}_fold3 --n-steps 540 --n-accum 4 --cores 8
   --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max
   -D DUMP_WEIGHTS=ON BN_FROZEN_STATS=ON`. Exit 1 is a FALSE failure (see RESULTS reference-loss caveat);
   the `[WDUMP]` block is emitted regardless.
3. **Extract + eval** (host): `process_round_k1.py --round r --fixture .../speechnet_train_k1_b{r}_fold3
   --gvsoc-log logs/round{r}_gvsoc_train.log --eval-batch r+1 --out-carry /tmp/carry_k1_b{r}_fold3.pt`.
   Parses the 6 device tensors from `[WDUMP]`, value-matches them to the ORT reference, builds the carry
   checkpoint (frozen backbone + device block4 + device fc), evaluates the next batch (bit-exact host
   inference), and saves the carry for round r+1.

## Tiling note
The default tiler reports "geometrical constraints infeasible" for K=1's heavier backward; the explicit
`--l1/--l2/--defaultMemLevel/--memAllocStrategy/--searchStrategy` flags above fix it. Head-only tiles with
defaults because its backward is trivial.

## Reference
`run_pytorch_k1_ondevicedata_chain.py` runs the PyTorch K=1 chain on the SAME Onnx4Deeploy 54-window draws
(removing the stratified-draw confound), producing `pytorch_k1_ondevicedata.csv` — the host reference the
device chain is validated against. Results in `RESULTS.md` / `k1_vs_headonly_vs_paper.csv`.
