#!/bin/bash
set -x

# ============================================================
# Latent CoT stage-2 debug script (single GPU, no DeepSpeed)
#
# Loads a trained latent-CoT checkpoint (including _latent_cot_*
# aux decoder + projection weights) and continues training.
# The loader auto-restores latent CoT weights that from_pretrained
# would otherwise drop.
#
# Usage:
#   bash run_script/train/navisim/sft_debug_stage2.sh
#
# Override GPU:
#   GPU=1 bash run_script/train/navisim/sft_debug_stage2.sh
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
# Point --model to the trained checkpoint; the loader will:
#   1. Load base Qwen3-VL weights via from_pretrained
#   2. Patch with latent CoT modules (fresh from AUX_MODEL_PATH)
#   3. Restore _latent_cot_* weights from the checkpoint safetensors
MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/ms-swift/outputs/navsim/qwen3_vl_latent_cot_stage1/v0-20260313-124424/checkpoint-1000"
AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct"
VISUAL_AUX_MODEL_PATH=""
DATASET_PATH="${SCRIPT_DIR}/data/navsim_latent_cot_full.jsonl"
VAL_DATASET_PATH="${SCRIPT_DIR}/data/navsim_val_latent_cot.jsonl"

# ---------- Latent CoT configuration ----------
export LATENT_COT_C_THOUGHT=6
export LATENT_COT_C_THOUGHT_VISUAL=0
export LATENT_COT_AUX_MODEL_PATH="${AUX_MODEL_PATH}"
export LATENT_COT_VISUAL_AUX_MODEL_PATH="${VISUAL_AUX_MODEL_PATH}"
export LATENT_COT_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_AUX_VISUAL_CONDITION=true
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=false
export LATENT_COT_FREEZE_VISUAL_AUX_DECODER=false
export LATENT_COT_FREEZE_AUX_DECODER=false
export LATENT_COT_FREEZE_MAIN_MODEL=false
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
    --template qwen3_vl_latent_cot \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
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
    --output_dir "${SCRIPT_DIR}/outputs/navsim/qwen3_vl_latent_cot_debug_stage2" \
    --gradient_checkpointing true
