#!/bin/bash
# Multi-Node + Multi-GPU parallel OneVL inference using qwen3_vl_infer_onevl.py
# Supports single-node (same as original) or multi-node via env vars.
# Requires shared filesystem for OUTPUT_DIR and TEST_SET_PATH (all nodes must see same paths).
#
# Single-node: ./infer_qwen3_vl_all_multinodes.sh   (same behavior as original script)
#
# Multi-node: use same env as train.sh (MLP_* from launcher).
#   NNODES=${MLP_WORKER_NUM:-1}, NODE_RANK=${MLP_ROLE_INDEX:-0}
#   When NNODES>1, set RUN_ID to same value on all nodes if launcher does not set it.
#   Assumes same number of GPUs per node.
set -e

PYTHON=/e2e-data/evad-tech-vla/huangzhijian/projects/ms-swift/.venv/bin/python3

# ---- Configuration (edit these) ----
MODEL_PATH=/e2e-data/evad-tech-vla/lujinghui/ms-swift/outputs/ar1/qwen3vl_stage0_answer/v6-20260317-161123/checkpoint-3892
TEST_SET_PATH=/e2e-data/evad-tech-vla/lujinghui/ms-swift/data/ar1/test_answer.jsonl
OUTPUT_PATH=${MODEL_PATH}/infer_results/qwen3_vl_infer_ar_merged.json

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
ADD_ASSISTANT_PREFIX=""
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INFER_SCRIPT="${SCRIPT_DIR}/../qwen3_vl_infer.py"

# ---- Multi-node: same as train.sh (MLP_WORKER_NUM, MLP_ROLE_INDEX) ----
NNODES=${MLP_WORKER_NUM:-1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
MASTER_PORT=${MLP_WORKER_0_PORT:-$(shuf -i 10000-50000 -n1)}

OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
RUN_ID=${RUN_ID:-${MLP_JOB_ID:-${SLURM_JOB_ID:-$$}}}

if [ "${NNODES}" -gt 1 ] && [ "${RUN_ID}" = "$$" ]; then
    INFER_RUN_ID_FILE="${OUTPUT_DIR}/._infer_multinode_run_id"
    if [ "${NODE_RANK}" -eq 0 ]; then
        mkdir -p "${OUTPUT_DIR}"
        RUN_ID=$(date +%s)
        echo "${RUN_ID}" > "${INFER_RUN_ID_FILE}"
        sync
        echo "Rank 0: set RUN_ID=${RUN_ID} for this multi-node run"
    else
        echo "[Node ${NODE_RANK}] Waiting for RUN_ID file from rank 0 (ensure OUTPUT_DIR is on shared storage) ..."
        while [ ! -f "${INFER_RUN_ID_FILE}" ]; do sleep 2; done
        RUN_ID=$(cat "${INFER_RUN_ID_FILE}")
        echo "[Node ${NODE_RANK}] Got RUN_ID=${RUN_ID}"
    fi
fi

# ---- Step 1: Detect local GPUs ----
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[Node ${NODE_RANK}] === Detected ${NUM_GPUS} GPUs (node ${NODE_RANK}/${NNODES}) ==="

TOTAL_WORKERS=$((NNODES * NUM_GPUS))
MY_SHARD_START=$((NODE_RANK * NUM_GPUS))
MY_SHARD_END=$(((NODE_RANK + 1) * NUM_GPUS))

SPLIT_DIR="${OUTPUT_DIR}/_splits_${RUN_ID}"
BARRIER_DIR="${SPLIT_DIR}"
DONE_FILE="${BARRIER_DIR}/done.${NODE_RANK}"
SPLIT_DONE_FILE="${BARRIER_DIR}/split_done"

# ---- Step 2: Split data (only rank 0 in multi-node, or single-node) ----
if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "${SPLIT_DIR}"
    echo "=== Splitting test set into ${TOTAL_WORKERS} shards ==="
    $PYTHON -c "
import json, math
if '${TEST_SET_PATH}'.endswith('.jsonl'):
    data = []
    with open('${TEST_SET_PATH}') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
else:
    with open('${TEST_SET_PATH}') as f:
        data = json.load(f)
n = ${TOTAL_WORKERS}
shard_size = math.ceil(len(data) / n) if n else 0
for i in range(n):
    shard = data[i*shard_size : (i+1)*shard_size]
    if not shard:
        continue
    with open(f'${SPLIT_DIR}/shard_{i}.json', 'w') as f:
        json.dump(shard, f, ensure_ascii=False)
    print(f'Shard {i}: {len(shard)} samples')
"
    touch "${SPLIT_DONE_FILE}"
else
    echo "[Node ${NODE_RANK}] Waiting for rank 0 to finish splitting ..."
    while [ ! -f "${SPLIT_DONE_FILE}" ]; do sleep 2; done
    echo "[Node ${NODE_RANK}] Split ready."
fi

# ---- Step 3: Launch inference for this node's shards only ----
echo "[Node ${NODE_RANK}] === Launching inference for shards ${MY_SHARD_START}..$((MY_SHARD_END - 1)) on ${NUM_GPUS} GPUs ==="
PIDS=()
LOCAL_GPU=0
for SHARD_ID in $(seq ${MY_SHARD_START} $((MY_SHARD_END - 1))); do
    SHARD_INPUT="${SPLIT_DIR}/shard_${SHARD_ID}.json"
    SHARD_OUTPUT="${SPLIT_DIR}/predict_${SHARD_ID}.json"

    if [ ! -f "${SHARD_INPUT}" ]; then
        echo "  [Node ${NODE_RANK}] Shard ${SHARD_ID}: no file, skipping"
        LOCAL_GPU=$((LOCAL_GPU + 1))
        continue
    fi

    echo "  [Node ${NODE_RANK}] Shard ${SHARD_ID} on cuda:${LOCAL_GPU} ..."
    $PYTHON "${INFER_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --test_set_path "${SHARD_INPUT}" \
        --output_path "${SHARD_OUTPUT}" \
        --device "cuda:${LOCAL_GPU}" \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        ${ADD_ASSISTANT_PREFIX} &

    PIDS+=($!)
    LOCAL_GPU=$((LOCAL_GPU + 1))
done

FAIL=0
for PID in "${PIDS[@]}"; do
    wait ${PID} || FAIL=1
done

if [ ${FAIL} -ne 0 ]; then
    echo "[Node ${NODE_RANK}] ERROR: one or more inference processes failed"
    exit 1
fi

touch "${DONE_FILE}"
echo "[Node ${NODE_RANK}] === Inference done, wrote ${DONE_FILE} ==="

# ---- Step 4: Merge (only rank 0); others wait then exit ----
if [ "${NODE_RANK}" -ne 0 ]; then
    echo "[Node ${NODE_RANK}] Exit (merge on rank 0)."
    exit 0
fi

echo "=== Rank 0: Waiting for all nodes to finish inference ==="
for r in $(seq 0 $((NNODES - 1))); do
    while [ ! -f "${BARRIER_DIR}/done.${r}" ]; do sleep 3; done
    echo "  Node ${r} done."
done

echo "=== All nodes done, merging results ==="
$PYTHON -c "
import json, glob
shards = sorted(glob.glob('${SPLIT_DIR}/predict_*.json'), key=lambda p: int(p.rsplit('_', 1)[1].replace('.json', '')))
merged = []
for path in shards:
    with open(path) as f:
        data = json.load(f)
        merged.extend(data)
with open('${OUTPUT_PATH}', 'w') as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)
print(f'Merged {len(merged)} samples from {len(shards)} shards -> ${OUTPUT_PATH}')
"

rm -rf "${SPLIT_DIR}"
echo "=== Done. Output saved to ${OUTPUT_PATH} ==="
