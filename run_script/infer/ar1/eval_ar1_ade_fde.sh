#!/usr/bin/env bash
# 评估 AR1 合并推理结果的 ADE / FDE（默认：stage0_answer vs stage0_cot）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/eval_ar1_ade_fde.py" "$@"
