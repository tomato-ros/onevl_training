cd /e2e-data/evad-tech-vla/lujinghui/ms-swift && \
source /e2e-data/evad-tech-vla/huangzhijian/projects/ms-swift/.venv/bin/activate && \
export PYTHONPATH="$(pwd):${PYTHONPATH}" && \
export PYTHONUNBUFFERED=1 && \
export TF_CPP_MIN_LOG_LEVEL=3 && \
export LATENT_COT_C_THOUGHT=6 && \
export LATENT_COT_C_THOUGHT_VISUAL=6 && \
export LATENT_COT_AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct-latent" && \
export LATENT_COT_VISUAL_AUX_MODEL_PATH="/e2e-data/evad-tech-vla/lujinghui/veomni_xiaomi/outputs/roadwork/qwen3_vl_visual_aux_decoder_ad/checkpoints/global_step_13040/hf_ckpt" && \
export LATENT_COT_EXPLAIN_LOSS_WEIGHT=1.0 && \
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=1.0 && \
export LATENT_COT_AUX_VISUAL_CONDITION=true && \
export LATENT_COT_USE_SEPARATE_VISUAL_LATENT_TOKENS=false && \
export LATENT_COT_FREEZE_VISUAL_AUX_DECODER=false && \
export LATENT_COT_FREEZE_AUX_DECODER=false && \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 swift sft --model /e2e-data/evad-tech-vla/lujinghui/lujinghui/models/qwen3vl/Qwen3-VL-4B-Instruct-latent --model_type qwen3_vl_latent_cot --train_type full --dataset data/navsim_latent_cot_100.jsonl --torch_dtype bfloat16 --max_steps 10 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 4e-5 --loss_type latent_cot --lr_scheduler_type cosine --gradient_accumulation_steps 2 --logging_steps 1 --max_length 4096 --warmup_ratio 0.05 --dataloader_num_workers 4 --output_dir outputs/qwen3_vl_latent_cot_dist_test2 --gradient_checkpointing true --deepspeed zero3 --save_steps 999 2>&1 | tee /tmp/latent_cot_dist_test2.log | tail -40