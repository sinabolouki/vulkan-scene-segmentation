#!/usr/bin/env bash
set -e

CONDA_ENV="sam2"          # <- your conda environment name
SAM2_DIR="/home/sina/sam_proj/sam2"           # <- where the sam2 repo (with checkpoints/) lives

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$(dirname "$0")"            # run from python/, NOT from the parent of the sam2 repo

python sam2_server.py \
    --checkpoint-dir "$SAM2_DIR/checkpoints" \
    --model base_plus \
    --host 0.0.0.0 --port 8765 \
    --warmup-size 1280x720 --warmup-runs 2 \
    --debug-dir debug_out \
    "$@"