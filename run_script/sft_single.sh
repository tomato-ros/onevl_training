set -x

PARTITION=${PARTITION:-"Intern5"}
# 单卡训练：device_map='auto' 与分布式不兼容，必须 num_processes=1
GPUS=${GPUS:-4}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
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
export LOCAL_RANK=${LOCAL_RANK:-0} 
export TORCH_EXTENSIONS_DIR="${OUTPUT_DIR}/torch_ext_$LOCAL_RANK"

torchrun --nproc_per_node=${GPUS} \
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
    --gradient_accumulation_steps 4 \
    --save_steps 400 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 3560 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --model_type internvl2_5 \
    --output_dir internvl_8B_sft \
    # --async_generate true \
    # --num_infer_workers 2 \
    # --system 'examples/train/grpo/prompt_baseline.txt'