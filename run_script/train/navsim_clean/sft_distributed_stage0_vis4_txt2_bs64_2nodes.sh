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
MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct"
DATASET_PATH="${SCRIPT_DIR}/data/navsim_vis4_text2.jsonl"
VAL_DATASET_PATH="/e2e-data/evad-tech-vla/huangzhijian/projects/ms-swift/data/navsim_test_cot_full_idx_trainfmt.json"

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
    --model_type qwen3_vl \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 4e-5 \
    --loss_type latent_cot \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_steps 100 \
    --weight_decay 0.05 \
    --freeze_aligner False \
    --freeze_llm False \
    --freeze_vit False \
    --dataloader_num_workers 8 \
    --output_dir "${SCRIPT_DIR}/outputs/navsim/qwen3vl_stage0_vis4_txt2_2nodes" \
    --deepspeed zero3 \
  2>&1 | tee "${SCRIPT_DIR}/logs/navsim/qwen3vl_stage0_vis4_txt2_2nodes.log"
