#!/bin/bash
set -e

GPU=${1:-0}
echo "[RUN] Using GPU: $GPU"

apptainer exec --nv \
  --bind $PWD:/workspace \
  --env CUDA_VISIBLE_DEVICES=$GPU \
  tapes.sif \
  python3 /workspace/Table-meets-LLM/inference.py