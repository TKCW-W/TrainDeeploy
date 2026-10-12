#!/usr/bin/env bash
# GVSoC runs of the four block-0 kernel fixtures on GAP9 (cluster and NE16), from SpeechNet_Test.
# Run from the HOST; needs the `deeploy_gap9` container (TrainDeeploy mounted at /app/Deeploy).
# Settings = exp6 (kernel harness deeploytest.c: usable L1 ~98 KB, hence --l1 92000).
#   bash run_smoke_gap9.sh [unfused_cluster fused_cluster unfused_NE16 fused_NE16]
set -uo pipefail
H=/Users/qiwenwu/ETH/TrainDeeploy/SpeechNet_Test
P="--l1 92000 --l2 1500000 --defaultMemLevel L2 --cores 8 --input-type-map blocks.0.conv.bias_rqsadd=int32_t --input-offset-map blocks.0.conv.bias_rqsadd=0"
ENV='source /app/install/gap9-sdk/.gap9-venv/bin/activate; source /app/install/gap9-sdk/configs/gap9_evk_audio.sh;
     export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation'
mkdir -p "$H/logs/smoke"

run() {  # $1 = name, $2 = runner, $3 = fixture
  docker exec deeploy_gap9 bash -lc "$ENV; cd /app/Deeploy/DeeployTest && rm -rf TEST_GAP9/build_sg_$1 TEST_GAP9_W_NE16/build_sg_$1 &&
    PYTEST_XDIST_WORKER=sg_$1 python3 $2 -t ../SpeechNet_Test/$3 $P" > "$H/logs/smoke/gap9_$1.log" 2>&1
  echo "$1: $(grep -aE 'Runtime: [0-9]+|Errors: [0-9]+ out of [0-9]+' "$H/logs/smoke/gap9_$1.log" | tail -2 | tr '\n' ' ')"
}
unfused_cluster() { run unfused_cluster deeployRunner_tiled_gap9.py         speechnet_unfused_b0_cluster; }  # exp6 A    734449
fused_cluster()   { run fused_cluster   deeployRunner_tiled_gap9.py         speechnet_fused_b0_cluster;   }  # exp6 C_cl 925593
unfused_NE16()    { run unfused_NE16    deeployRunner_tiled_gap9_w_ne16.py  speechnet_unfused_b0_NE16;    }  # exp6 B    379969
fused_NE16()      { run fused_NE16      deeployRunner_tiled_gap9_w_ne16.py  speechnet_fused_b0_NE16;      }  # exp6 C    541345

targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=(unfused_NE16 fused_NE16 unfused_cluster fused_cluster)
docker exec deeploy_gap9 bash -lc 'pgrep -f "[g]vsoc_launcher" | xargs -r kill -9' || true
for t in "${targets[@]}"; do $t; done
