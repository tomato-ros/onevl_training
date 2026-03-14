#!/bin/bash
set -x

# ============================================================
# Latent CoT debug training script (single GPU, no DeepSpeed)
#
# Uses the ORIGINAL Qwen3-VL model (not the -latent variant).
# Latent tokens (<|latent|>, <|start-latent|>, etc.) are added
# as regular (non-special) tokens at runtime.
#
# Usage:
#   bash run_script/train/navisim/sft_debug.sh
#
# Debug with breakpoints:
#   CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 --wait-for-client \
#     -m swift sft <...same args...>
# ============================================================

# ---------- Environment ----------
source /e2e-data/evad-tech-vla/huangzhijian/projects/ms-swift/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

# ---------- GPU selection (default: single GPU 0) ----------
GPU=${GPU:-0}
NPROC=${NPROC:-1}

# ---------- Model paths ----------
MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct"
AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct"
VISUAL_AUX_MODEL_PATH=""
DATASET_PATH="${SCRIPT_DIR}/data/navsim_latent_cot_full.jsonl"
VAL_DATASET_PATH="${SCRIPT_DIR}/data/navsim_val_latent_cot.jsonl"

# ---------- Latent CoT configuration ----------
export LATENT_COT_C_THOUGHT=6
export LATENT_COT_C_THOUGHT_VISUAL=0
export LATENT_COT_AUX_MODEL_PATH="${AUX_MODEL_PATH}"
export LATENT_COT_VISUAL_AUX_MODEL_PATH="${VISUAL_AUX_MODEL_PATH}"
export LATENT_COT_EXPLAIN_LOSS_WEIGHT=0.5
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_AUX_VISUAL_CONDITION=true
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=false
export LATENT_COT_FREEZE_VISUAL_AUX_DECODER=false
export LATENT_COT_FREEZE_AUX_DECODER=false
export LATENT_COT_FREEZE_MAIN_MODEL=false
# Keep original vocab unchanged: latent markers are sub-tokenized, positions
# found via |latent| pattern matching. No add_tokens / resize_embeddings.
export LATENT_COT_LATENT_CE_LOSS=false
export LATENT_COT_LATENT_USE_ALL_SUBTOKENS=false
export LATENT_COT_USE_ORIGINAL_VOCAB=true

# ---------- Launch training ----------
mkdir -p "${SCRIPT_DIR}/logs/navsim"

CUDA_VISIBLE_DEVICES=${GPU} \
NPROC_PER_NODE=${NPROC} \
swift sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl_latent_cot \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-5 \
    --loss_type latent_cot \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 4 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --eval_metric acc \
    --metric_for_best_model token_acc \
    --load_best_model_at_end true \
    --logging_steps 1 \
    --max_length 4096 \
    --warmup_steps 50 \
    --weight_decay 0.05 \
    --freeze_vit false \
    --dataloader_num_workers 0 \
    --output_dir "${SCRIPT_DIR}/outputs/navsim/qwen3_vl_latent_cot_debug" \
    --gradient_checkpointing true
