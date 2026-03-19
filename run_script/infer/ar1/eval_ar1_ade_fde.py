#!/usr/bin/env python3
"""
评估 AR1 推理合并结果 qwen3_vl_infer_ar_merged.json 的 ADE / FDE（米）。

每条样本从 GT、output_text 中解析 <answer>[[x,y], ...]</answer> 轨迹；
CoT 模型 output_text 可在 <answer> 前有 </think> 等前缀，本脚本只解析 <answer> 块。

用法:
  python eval_ar1_ade_fde.py
  python eval_ar1_ade_fde.py --json path/to/a.json path/to/b.json
  python eval_ar1_ade_fde.py --json a.json --latency
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MS_SWIFT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))

ANSWER_TRAJ_RE = re.compile(
    r"<answer>\s*(\[\[.*?\]\])\s*</answer>", re.DOTALL | re.IGNORECASE
)


def parse_trajectory(text: str):
    if not text:
        return None
    m = ANSWER_TRAJ_RE.search(text)
    if not m:
        return None
    try:
        return np.array(json.loads(m.group(1)))
    except (json.JSONDecodeError, ValueError):
        return None


def compute_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    n = min(len(pred), len(gt))
    if n == 0:
        return float("nan")
    pred, gt = pred[:n], gt[:n]
    d = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    return float(np.mean(d))


def compute_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    if len(pred) == 0 or len(gt) == 0:
        return float("nan")
    return float(np.sqrt(np.sum((pred[-1] - gt[-1]) ** 2)))


def eval_one_json(path: str, with_latency: bool) -> dict:
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    ades, fdes, lats = [], [], []
    failed = 0

    for item in data:
        gt_traj = parse_trajectory(item.get("GT", "") or "")
        pred_traj = parse_trajectory(item.get("output_text", "") or "")
        if gt_traj is None or pred_traj is None:
            failed += 1
            continue
        if len(gt_traj) == 0 or len(pred_traj) == 0:
            failed += 1
            continue
        ades.append(compute_ade(pred_traj, gt_traj))
        fdes.append(compute_fde(pred_traj, gt_traj))
        if with_latency and item.get("latency") is not None:
            try:
                lats.append(float(item["latency"]))
            except (TypeError, ValueError):
                pass

    ar1_root = os.path.join(_MS_SWIFT_ROOT, "outputs", "ar1")
    try:
        label = os.path.relpath(path, ar1_root)
    except ValueError:
        label = path
    out = {
        "path": path,
        "name": os.path.basename(path),
        "label": label,
        "total": len(data),
        "evaluated": len(ades),
        "failed_parse": failed,
        "ade_mean": float(np.mean(ades)) if ades else float("nan"),
        "ade_std": float(np.std(ades)) if ades else float("nan"),
        "fde_mean": float(np.mean(fdes)) if fdes else float("nan"),
        "fde_std": float(np.std(fdes)) if fdes else float("nan"),
    }
    if with_latency and lats:
        out["latency_mean"] = float(np.mean(lats))
        out["latency_std"] = float(np.std(lats))
        out["latency_median"] = float(np.median(lats))
    return out


def default_json_paths():
    return [
        os.path.join(
            _MS_SWIFT_ROOT,
            "outputs",
            "ar1",
            "qwen3vl_stage0_answer",
            "v6-20260317-161123",
            "checkpoint-3892",
            "infer_results",
            "qwen3_vl_infer_ar_merged.json",
        ),
        os.path.join(
            _MS_SWIFT_ROOT,
            "outputs",
            "ar1",
            "qwen3vl_stage0_cot",
            "v3-20260317-160634",
            "checkpoint-3892",
            "infer_results",
            "qwen3_vl_infer_ar_merged.json",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="AR1 merged infer: ADE/FDE (meters)")
    parser.add_argument(
        "--json",
        nargs="*",
        default=None,
        help="一个或多个 qwen3_vl_infer_ar_merged.json；默认对比 answer 与 cot 两个合并文件",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="同时统计 latency 均值/标准差/中位数",
    )
    args = parser.parse_args()

    paths = args.json if args.json else default_json_paths()
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        for p in missing:
            print(f"[missing] {p}", file=sys.stderr)
        sys.exit(1)

    rows = [eval_one_json(p, args.latency) for p in paths]

    sep_top = 120
    print("=" * sep_top)
    print("AR1 ADE / FDE (meters), ego-frame xy in <answer>")
    print("=" * sep_top)
    w = max(len(r["label"]) for r in rows) if rows else 40
    w = min(w, 110)
    hdr = f"{'run / file':<{w}} {'N':>7} {'ADE':>12} {'FDE':>12}"
    if args.latency:
        hdr += f" {'latency(s)':>14}"
    sep_w = w + 48 if args.latency else w + 34
    print(hdr)
    print("-" * sep_w)
    for r in rows:
        ade_s = f"{r['ade_mean']:.4f}±{r['ade_std']:.4f}"
        fde_s = f"{r['fde_mean']:.4f}±{r['fde_std']:.4f}"
        lab = r["label"][:w]
        line = f"{lab:<{w}} {r['evaluated']:>7} {ade_s:>12} {fde_s:>12}"
        if args.latency and "latency_mean" in r:
            line += f" {r['latency_mean']:.2f}±{r['latency_std']:.2f}"
        print(line)
    print("-" * sep_w)
    for r in rows:
        print(
            f"  {r['label']}: total={r['total']}, ok={r['evaluated']}, parse_fail={r['failed_parse']}"
        )
        if args.latency and "latency_median" in r:
            print(
                f"    latency median={r['latency_median']:.2f}s mean={r['latency_mean']:.2f}s"
            )
    print("=" * sep_top)


if __name__ == "__main__":
    main()
