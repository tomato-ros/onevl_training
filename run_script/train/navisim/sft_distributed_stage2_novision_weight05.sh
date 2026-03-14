#!/bin/bash
set -x

# ============================================================
# Latent CoT distributed training script
# Supports single-node multi-GPU and multi-node multi-GPU
#
# Usage:
#   Single-node (auto-detect GPUs):
#     bash run_script/sft_distributed.sh
#
#   Multi-node (run on each node):
#     NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash run_script/sft_distributed.sh  # node 0
#     NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash run_script/sft_distributed.sh  # node 1
# ============================================================

# ---------- Environment ----------
source /e2e-data/evad-tech-vla/huangzhijian/projects/ms-swift/.venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

# ---------- Distributed settings ----------
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=${MLP_WORKER_0_PORT:-29500}

# ---------- Model paths ----------
MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/ms-swift/outputs/navsim/qwen3_vl_latent_cot_stage1_novision_subtokens/v0-20260314-071222/checkpoint-500"
AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct"
# VISUAL_AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/veomni_xiaomi/outputs/roadwork/qwen3_vl_visual_aux_decoder_ad/checkpoints/global_step_13040/hf_ckpt"
VISUAL_AUX_MODEL_PATH=""
DATASET_PATH="${SCRIPT_DIR}/data/navsim_latent_cot_full_latent_6.jsonl"
VAL_DATASET_PATH="${SCRIPT_DIR}/data/navsim_val_latent_cot.jsonl"

# ---------- Latent CoT configuration ----------
export LATENT_COT_C_THOUGHT=6
export LATENT_COT_C_THOUGHT_VISUAL=0
export LATENT_COT_AUX_MODEL_PATH="${AUX_MODEL_PATH}"
export LATENT_COT_VISUAL_AUX_MODEL_PATH="${VISUAL_AUX_MODEL_PATH}"
export LATENT_COT_EXPLAIN_LOSS_WEIGHT=0.5
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_AUX_VISUAL_CONDITION=false
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=false
# With DeepSpeed zero3 + multi-GPU, memory is sufficient to train aux decoders
export LATENT_COT_FREEZE_VISUAL_AUX_DECODER=false
export LATENT_COT_FREEZE_AUX_DECODER=false
export LATENT_COT_FREEZE_MAIN_MODEL=false
export LATENT_COT_LATENT_CE_LOSS=true
export LATENT_COT_LATENT_USE_ALL_SUBTOKENS=true
export LATENT_COT_USE_ORIGINAL_VOCAB=true

# ---------- Launch training ----------
mkdir -p "${SCRIPT_DIR}/logs"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc_per_node-1))) \
NPROC_PER_NODE=$nproc_per_node \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
swift sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl_latent_cot \
    --template qwen3_vl_latent_cot \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-6 \
    --loss_type latent_cot \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 4 \
    --eval_metric acc \
    --metric_for_best_model token_acc \
    --load_best_model_at_end false \
    --logging_steps 1 \
    --max_length 4096 \
    --warmup_steps 100 \
    --weight_decay 0.05 \
    --freeze_aligner False \
    --freeze_llm False \
    --freeze_vit False \
    --dataloader_num_workers 4 \
    --output_dir "${SCRIPT_DIR}/outputs/navsim/qwen3_vl_latent_cot_stage2_novision_subtokens_weight05" \
    --gradient_checkpointing true \
    --deepspeed zero3 \
  2>&1 | tee "${SCRIPT_DIR}/logs/navsim/qwen3_vl_latent_cot_stage2_novision_subtokens_weight05.log"
