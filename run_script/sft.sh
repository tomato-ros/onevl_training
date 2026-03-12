set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# ---------- Latent CoT configuration ----------
export LATENT_COT_C_THOUGHT=6
export LATENT_COT_C_THOUGHT_VISUAL=6
export LATENT_COT_AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct-latent"
export LATENT_COT_VISUAL_AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/veomni_xiaomi/outputs/roadwork/qwen3_vl_visual_aux_decoder_ad/checkpoints/global_step_13040/hf_ckpt"
export LATENT_COT_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=1.0
export LATENT_COT_AUX_VISUAL_CONDITION=true
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=false
# Freeze aux decoders to reduce memory; set to false if multi-GPU and enough memory
export LATENT_COT_FREEZE_VISUAL_AUX_DECODER=true
export LATENT_COT_FREEZE_AUX_DECODER=true
# -----------------------------------------------

mkdir -p logs

# Single-GPU debug run. For multi-GPU, replace with:
#   torchrun --nproc_per_node=N swift/cli/sft.py ...
CUDA_VISIBLE_DEVICES=0 python swift/cli/sft.py \
    --model /e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct-latent \
    --model_type qwen3_vl_latent_cot \
    --train_type full \
    --dataset 'data/navsim_latent_cot_100.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 4e-5 \
    --loss_type latent_cot \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 16 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --output_dir outputs/qwen3_vl_latent_cot_sft \
    --gradient_checkpointing true \
    --ddp_find_unused_parameters true \
    --freeze_parameters _latent_cot_aux_decoder _latent_cot_visual_aux_decoder \
  2>&1 | tee logs/qwen3_vl_latent_cot_sft.log
