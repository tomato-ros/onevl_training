set -x

source /e2e-data/evad-tech-vla/huangzhijian/projects/ms-swift/.venv/bin/activate

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-8}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# OUTPUT_DIR=./internvl_sft

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
mkdir -p logs
export LOCAL_RANK=${LOCAL_RANK:-0} 
export TORCH_EXTENSIONS_DIR="${OUTPUT_DIR}/torch_ext_$LOCAL_RANK"

NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
MASTER_PORT=${MLP_WORKER_0_PORT:-$(shuf -i 10000-50000 -n1)}

/e2e-data/evad-tech-vla/huangzhijian/projects/ms-swift/.venv/bin/torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=${GPUS}  \
    --master_port=$MASTER_PORT \
  swift/cli/sft.py \
    --model /e2e-data/evad-tech-vla/lujinghui/models/InternVL3-8B \
    --train_type full \
    --freeze_vit False \
    --freeze_aligner False \
    --dataset '/e2e-data/evad-tech-vla/lujinghui/lujinghui/datasets/navsim/dataset_navsim_traj1_new.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-5 \
    --target_modules all-linear \
    --lr_scheduler_type "cosine" \
    --gradient_accumulation_steps 2 \
    --save_steps 400 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 3560 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --model_type internvl2_5 \
    --output_dir outputs/internvl_8B_sft \
  2>&1 | tee logs/internvl_8B_sft.log
    # --async_generate true \
    # --num_infer_workers 2 \
    # --system 'examples/train/grpo/prompt_baseline.txt'