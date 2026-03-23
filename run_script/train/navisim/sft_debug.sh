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
source /e2e-data/evad-tech-vla/huangzhijian5/projects/ms-swift/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

# ---------- GPU selection (default: single GPU 0) ----------
GPU=${GPU:-0}
NPROC=${NPROC:-1}

# ---------- Model paths ----------
MODEL_PATH="//e2e-data/embodied-research-data/opendata/roadworks/models/qwen3vl/Qwen3-VL-4B-Instruct"
AUX_MODEL_PATH="//e2e-data/embodied-research-data/opendata/roadworks/models/qwen3vl/Qwen3-VL-4B-Instruct"
VISUAL_AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/veomni_xiaomi/outputs/navsim/qwen3_vl_visual_aux_decoder_512/checkpoints/global_step_15634/hf_ckpt"
DATASET_PATH="${SCRIPT_DIR}/data/navsim_vis4_text2_test100.jsonl"
VAL_DATASET_PATH="${SCRIPT_DIR}/data/navsim_vis4_text2_test100.jsonl"

# ---------- Latent CoT configuration ----------
export LATENT_COT_C_THOUGHT=2
export LATENT_COT_C_THOUGHT_VISUAL=4
export LATENT_COT_AUX_MODEL_PATH="${AUX_MODEL_PATH}"
export LATENT_COT_VISUAL_AUX_MODEL_PATH="${VISUAL_AUX_MODEL_PATH}"
export LATENT_COT_EXPLAIN_LOSS_WEIGHT=0.5
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_AUX_VISUAL_CONDITION=false ## whether text aux decoder input vit embeddings
export LATENT_COT_VISUAL_AUX_VISUAL_CONDITION=false ## whether visual aux decoder input vit embeddings
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=true ## use separate visual latent tokens (text aux=<|latent|>, vis aux=<|latent-vis|>)
export LATENT_COT_FREEZE_VISUAL_AUX_DECODER=false ## freeze visual aux decoder
export LATENT_COT_FREEZE_AUX_DECODER=false ## freeze text aux decoder
export LATENT_COT_FREEZE_MAIN_MODEL=true ## freeze main model
# Keep original vocab unchanged: latent markers are sub-tokenized, positions
# found via |latent| pattern matching. No add_tokens / resize_embeddings.
export LATENT_COT_LATENT_CE_LOSS=true ## whether compute latent token ce loss in main model
export LATENT_COT_LATENT_USE_ALL_SUBTOKENS=true ## whether use all subtokens
export LATENT_COT_USE_ORIGINAL_VOCAB=true ## whether use original vocab

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
    --gradient_checkpointing true 2>&1 | tee "${SCRIPT_DIR}/logs/navsim/qwen3_vl_latent_cot_debug.log"
