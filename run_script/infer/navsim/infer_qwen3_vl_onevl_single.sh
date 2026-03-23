cd /e2e-data/evad-tech-vla/lujinghui/ms-swift && \
source /e2e-data/evad-tech-vla/huangzhijian5/projects/ms-swift/.venv/bin/activate && \
CUDA_VISIBLE_DEVICES=0 python3 run_script/infer/qwen3_vl_infer_onevl.py \
    --model_path /e2e-data/evad-tech-vla/lujinghui/ms-swift/outputs/qwen3_vl_latent_cot_distributed/v0-20260312-150448/checkpoint-3228 \
    --test_set_path /tmp/onevl_debug_2.json \
    --output_path /tmp/onevl_debug_explain.json \
    --device cuda:0 \
    --num_latent 6 \
    --max_new_tokens 128 \
    --decoder_explain \
    --aux_model_path //e2e-data/embodied-research-data/opendata/roadworks/models/qwen3vl/Qwen3-VL-4B-Instruct-latent \
    --aux_visual_condition \
    --c_thought 6 \
    --max_explain_tokens 128 \
    2>&1