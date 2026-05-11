#!/bin/bash
# Multi-node inference: split data by nodes, each node runs swift infer on its split, rank 0 merges.
# Single-node: NNODES=1 (default), one split = full data.
# Multi-node: set WORKER_NUM, ROLE_INDEX (same as train launcher). RUN_ID synced via OUTPUT_DIR when launcher does not set it.
# Requires shared filesystem: OUTPUT_DIR (result_path's dir) and VAL_DATASET must be visible to all nodes (e.g. NFS); otherwise some nodes will hang waiting for split_done.
source projects/ms-swift/.venv/bin/activate
rm -rf /root/.cache

cd ms-swift

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MS_SWIFT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${MS_SWIFT_ROOT}"
export PYTHONPATH="${MS_SWIFT_ROOT}:${PYTHONPATH:-}"

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MODEL=outputs/internvl_8B_sft/v0-20260310-124057/checkpoint-3990
VAL_DATASET=lujinghui/datasets/navsim/dataset_navsim_test_traj1.jsonl
RESULT_PATH=outputs/internvl_8B_sft/results/predict_full_6e.jsonl
# Canonical absolute path so all nodes use same SPLIT_DIR (required for shared storage)
OUTPUT_DIR=$(cd "$(dirname "${RESULT_PATH}")" && pwd)
SWIFT_INFER_EXTRA="--infer_backend pt --max_batch_size 8 --max_new_tokens 512 --temperature 0"

# ---- Multi-node env (same as reference: * from launcher, RUN_ID sync when needed) ----
NNODES=${WORKER_NUM:-1}
NODE_RANK=${ROLE_INDEX:-0}
# Same RUN_ID on all nodes for shared SPLIT_DIR. If launcher does not set it, rank 0 writes to shared OUTPUT_DIR and others read.
RUN_ID=${RUN_ID:-${JOB_ID:-${SLURM_JOB_ID:-$$}}}
INFER_RUN_ID_FILE="${OUTPUT_DIR}/._infer_multinode_run_id"

if [ "${NNODES}" -gt 1 ] && [ "${RUN_ID}" = "$$" ]; then
    # Launcher did not set RUN_ID; rank 0 writes RUN_ID to shared dir, other nodes read (OUTPUT_DIR must be on shared storage)
    if [ "${NODE_RANK}" -eq 0 ]; then
        mkdir -p "${OUTPUT_DIR}"
        RUN_ID=$(date +%s)
        echo "${RUN_ID}" > "${INFER_RUN_ID_FILE}"
        sync
        echo "[Rank 0] set RUN_ID=${RUN_ID} for this multi-node run"
    else
        echo "[Node ${NODE_RANK}] Waiting for RUN_ID file from rank 0 (ensure OUTPUT_DIR is on shared storage) ..."
        while [ ! -f "${INFER_RUN_ID_FILE}" ]; do sleep 2; done
        RUN_ID=$(cat "${INFER_RUN_ID_FILE}")
        echo "[Node ${NODE_RANK}] Got RUN_ID=${RUN_ID}"
    fi
fi

SPLIT_DIR="${OUTPUT_DIR}/_infer_splits_${RUN_ID}"
SPLIT_DONE_FILE="${SPLIT_DIR}/split_done"

# ---- Step 1: Split data by nodes (rank 0 only) ----
if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "${SPLIT_DIR}"
    touch "${SPLIT_DIR}/.split_in_progress"
    sync
    TOTAL_LINES=$(wc -l < "${VAL_DATASET}")
    LINES_PER_NODE=$(( (TOTAL_LINES + NNODES - 1) / NNODES ))
    echo "=== Splitting ${TOTAL_LINES} lines into ${NNODES} node splits (≈${LINES_PER_NODE} lines each) ==="
    for i in $(seq 0 $((NNODES - 1))); do
        START=$(( i * LINES_PER_NODE + 1 ))
        END=$(( (i + 1) * LINES_PER_NODE ))
        if [ "${END}" -gt "${TOTAL_LINES}" ]; then
            END=${TOTAL_LINES}
        fi
        if [ "${START}" -le "${TOTAL_LINES}" ]; then
            sed -n "${START},${END}p" "${VAL_DATASET}" > "${SPLIT_DIR}/split_${i}.jsonl"
            COUNT=$(wc -l < "${SPLIT_DIR}/split_${i}.jsonl")
            echo "  split_${i}.jsonl: ${COUNT} lines"
        fi
    done
    touch "${SPLIT_DONE_FILE}"
    sync
    echo "[Rank 0] Split done (synced); other nodes can proceed."
else
    echo "[Node ${NODE_RANK}] Waiting for rank 0 to finish splitting (OUTPUT_DIR must be shared: ${OUTPUT_DIR}) ..."
    sync
    # First wait for SPLIT_DIR to appear (NFS may delay directory visibility)
    WAIT_COUNT=0
    while [ ! -d "${SPLIT_DIR}" ]; do
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 1))
        if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
            sync
            echo "[Node ${NODE_RANK}] Still waiting for SPLIT_DIR to appear (${SPLIT_DIR}) ..."
        fi
    done
    # Then wait for split_done; periodically sync to refresh NFS view
    WAIT_COUNT=0
    while [ ! -f "${SPLIT_DONE_FILE}" ]; do
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 1))
        if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
            sync
            echo "[Node ${NODE_RANK}] Still waiting for split_done ..."
        fi
    done
    # Ensure our split file is visible (NFS cache)
    MY_SPLIT="${SPLIT_DIR}/split_${NODE_RANK}.jsonl"
    WAIT_COUNT=0
    while [ ! -f "${MY_SPLIT}" ]; do
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 1))
        if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
            sync
            echo "[Node ${NODE_RANK}] Still waiting for ${MY_SPLIT} ..."
        fi
    done
    echo "[Node ${NODE_RANK}] Split ready."
fi

# ---- Step 2: This node runs inference on its split ----
MY_SPLIT="${SPLIT_DIR}/split_${NODE_RANK}.jsonl"
MY_RESULT="${SPLIT_DIR}/result_${NODE_RANK}.jsonl"

if [ ! -f "${MY_SPLIT}" ]; then
    echo "[Node ${NODE_RANK}] No split file ${MY_SPLIT}, skipping inference (empty node)."
    touch "${SPLIT_DIR}/done.${NODE_RANK}"
    if [ "${NODE_RANK}" -eq 0 ]; then
        : # rank 0 will merge and may create empty or partial output
    else
        exit 0
    fi
else
    LINES_MY=$(wc -l < "${MY_SPLIT}")
    echo "[Node ${NODE_RANK}] === Running swift infer on ${LINES_MY} samples ==="
    swift infer \
        --model "${MODEL}" \
        --val_dataset "${MY_SPLIT}" \
        --result_path "${MY_RESULT}" \
        ${SWIFT_INFER_EXTRA}
    touch "${SPLIT_DIR}/done.${NODE_RANK}"
    echo "[Node ${NODE_RANK}] === Inference done, wrote ${MY_RESULT} ==="
fi

# ---- Step 3: Merge (rank 0 only); other nodes exit ----
if [ "${NODE_RANK}" -ne 0 ]; then
    echo "[Node ${NODE_RANK}] Exit (merge on rank 0)."
    exit 0
fi

echo "=== Rank 0: Waiting for all nodes to finish ==="
for r in $(seq 0 $((NNODES - 1))); do
    while [ ! -f "${SPLIT_DIR}/done.${r}" ]; do sleep 3; done
    echo "  Node ${r} done."
done

echo "=== Merging results ==="
mkdir -p "${OUTPUT_DIR}"
: > "${RESULT_PATH}"
for r in $(seq 0 $((NNODES - 1))); do
    F="${SPLIT_DIR}/result_${r}.jsonl"
    if [ -f "${F}" ]; then
        cat "${F}" >> "${RESULT_PATH}"
    fi
done
TOTAL_OUT=$(wc -l < "${RESULT_PATH}")
echo "Merged ${TOTAL_OUT} lines -> ${RESULT_PATH}"

rm -rf "${SPLIT_DIR}"
echo "=== Multi-node inference finished ==="
