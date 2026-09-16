#!/usr/bin/env bash
# run_all.sh — Person 1 full pipeline: data -> SFT -> compare/evaluate
set -e
cd "$(dirname "$0")/src"

echo "==> Step 1: prepare SFT data"
python prepare_sft_data.py

echo "==> Step 2: train SFT (DistilGPT-2)"
python train_sft.py --epochs 8 --batch_size 4 --lr 5e-5

echo "==> Steps 3 & 4: Base vs SFT comparison + evaluation"
python evaluate_base_vs_sft.py

echo "==> Done. See outputs/ for models, plots, tables, and the human-eval sheet."
