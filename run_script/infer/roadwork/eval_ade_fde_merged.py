#!/usr/bin/env python3
"""Compute ADE / FDE from qwen3_vl_infer_onevl_merged.json (roadwork trajectories).

Coordinates in JSON are assumed **normalized** to a 0–1000 scale. Before metrics,
they are denormalized to **pixel space** using the first user image in ``messages``:

  ``x_px = x / 1000 * w``, ``y_px = y / 1000 * h``  (``w, h`` = image width/height).

Build an augmented prediction: **GT[:prefix_k] + parsed(output_text)** (simulates
prefilling the first ``prefix_k`` ground-truth points ahead of model coordinates),
then compare to **full GT** on the first ``horizon`` timesteps (default 20).

- **ADE**: mean L2 over timesteps ``0 .. n-1``, with
  ``n = min(horizon, len(gt), len(aug_pred))``.
- **FDE**: L2 at timestep ``n - 1`` (end of that evaluation window).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_PAIR_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def first_user_image_path_from_messages(messages: Any) -> Optional[str]:
    """Return the first ``image`` path from a user turn (Qwen-VL chat list content)."""
    if not isinstance(messages, list):
        return None
    for turn in messages:
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image":
                continue
            img = part.get("image") or part.get("image_url")
            if isinstance(img, str) and img.strip():
                return img.strip()
    return None


def image_wh(path: str, cache: Dict[str, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Load image size (width, height); cache by path."""
    if path in cache:
        return cache[path]
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with Image.open(p) as im:
            w, h = im.size
    except OSError:
        return None
    cache[path] = (int(w), int(h))
    return cache[path]


def denorm_xy_norm1000(pts: np.ndarray, w: float, h: float) -> np.ndarray:
    """``x_px = x/1000*w``, ``y_px = y/1000*h`` for array [N, 2+]."""
    out = np.asarray(pts, dtype=np.float64)
    if out.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if out.ndim == 1 and out.shape[0] == 2:
        out = out.reshape(1, 2)
    if out.ndim != 2 or out.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    out = out.copy()
    out[:, 0] = out[:, 0] / 1000.0 * w
    out[:, 1] = out[:, 1] / 1000.0 * h
    return out


def parse_gt_waypoints(gt_str: str) -> List[List[float]]:
    if not gt_str or not isinstance(gt_str, str):
        return []
    s = gt_str.strip()
    if not s:
        return []
    try:
        data = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        try:
            data = ast.literal_eval("[" + s + "]")
        except (SyntaxError, ValueError):
            return []
    if not data:
        return []
    if isinstance(data[0], (int, float)):
        return [list(map(float, data))]
    return [list(map(float, p)) for p in data]


def parse_output_waypoints(text: str) -> List[List[float]]:
    """Extract all [x, y] pairs from model output (may contain </answer> / junk)."""
    if not text or not isinstance(text, str):
        return []
    cut = text.split("</answer>", 1)[0]
    cut = cut.split("</think>", 1)[0]
    pts: List[List[float]] = []
    for m in _PAIR_RE.finditer(cut):
        pts.append([float(m.group(1)), float(m.group(2))])
    return pts


def augment_pred_with_gt_prefix(
    gt: Sequence[Sequence[float]],
    pred_out: Sequence[Sequence[float]],
    prefix_k: int,
) -> np.ndarray:
    """``pred_aug = gt[:prefix_k] + pred_out`` as float array [T, 2]."""
    g = np.asarray(gt, dtype=np.float64)
    p = np.asarray(pred_out, dtype=np.float64)
    if g.size == 0:
        return np.zeros((0, 2))
    k = min(prefix_k, g.shape[0])
    if p.size == 0:
        return g[:k].copy()
    return np.concatenate([g[:k], p], axis=0)


def ade_fde_first_horizon(
    gt: Sequence[Sequence[float]],
    pred_aug: Sequence[Sequence[float]],
    horizon: int,
) -> Tuple[np.ndarray, float, int]:
    """Per-step L2 over first n steps, FDE at step n-1, n = min(horizon, len(gt), len(pred_aug))."""
    g = np.asarray(gt, dtype=np.float64)
    p = np.asarray(pred_aug, dtype=np.float64)
    if g.ndim != 2 or p.ndim != 2 or g.shape[1] < 2 or p.shape[1] < 2:
        return np.array([]), float("nan"), 0
    n = int(min(horizon, g.shape[0], p.shape[0]))
    if n <= 0:
        return np.array([]), float("nan"), 0
    d = np.linalg.norm(p[:n, :2] - g[:n, :2], axis=1)
    return d, float(d[-1]), n


def main() -> None:
    p = argparse.ArgumentParser(description="ADE / FDE from merged OneVL infer JSON")
    p.add_argument(
        "--merged_json",
        type=Path,
        default="/e2e-data/evad-tech-vla/lujinghui/ms-swift/outputs/roadwork/qwen3_vl_latent_cot_stage2_vis4_txt2_fixbug_512_bs64_with_viscondition/v0-20260324-031631/checkpoint-1260/infer_results_prefill/qwen3_vl_infer_onevl_merged.json",
        help="Path to qwen3_vl_infer_onevl_merged.json",
    )
    p.add_argument(
        "--prefix_k",
        type=int,
        default=5,
        help="Prepend this many GT points to parsed output before metrics (default 5)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Evaluate ADE/FDE on the first N timesteps after alignment (default 20)",
    )
    p.add_argument(
        "--per_sample_json",
        type=Path,
        default=None,
        help="Optional path to write per-sample ade/fde JSON",
    )
    p.add_argument(
        "--no_denorm",
        action="store_true",
        help="Do not denormalize (treat coordinates as already in pixel space)",
    )
    args = p.parse_args()

    with open(args.merged_json, encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    ades: List[float] = []
    fdes: List[float] = []
    skipped: List[Tuple[int, str]] = []
    per_sample: List[Dict[str, Any]] = []
    hw_cache: Dict[str, Tuple[int, int]] = {}

    for i, item in enumerate(data):
        gt = parse_gt_waypoints(item.get("GT", ""))
        pred = parse_output_waypoints(item.get("output_text", ""))

        denorm_image: Optional[str] = None
        denorm_wh: Optional[Tuple[int, int]] = None
        if not args.no_denorm:
            denorm_image = first_user_image_path_from_messages(item.get("messages"))
            if not denorm_image:
                reason = "no_image_in_messages"
                skipped.append((i, reason))
                per_sample.append(
                    {"index": i, "ade": None, "fde": None, "n_eval": 0, "skip": reason}
                )
                continue
            denorm_wh = image_wh(denorm_image, hw_cache)
            if denorm_wh is None:
                reason = f"image_unreadable:{denorm_image}"
                skipped.append((i, reason))
                per_sample.append(
                    {"index": i, "ade": None, "fde": None, "n_eval": 0, "skip": reason}
                )
                continue
            w, h = denorm_wh
            gt_arr = denorm_xy_norm1000(np.asarray(gt, dtype=np.float64), w, h)
            pred_arr = denorm_xy_norm1000(np.asarray(pred, dtype=np.float64), w, h)
            gt = gt_arr.tolist()
            pred = pred_arr.tolist()

        if len(gt) < args.prefix_k:
            reason = f"gt_len={len(gt)}<prefix_k={args.prefix_k}"
            skipped.append((i, reason))
            per_sample.append({"index": i, "ade": None, "fde": None, "n_eval": 0, "skip": reason})
            continue
        if not pred:
            reason = "no_pred_points"
            skipped.append((i, reason))
            per_sample.append({"index": i, "ade": None, "fde": None, "n_eval": 0, "skip": reason})
            continue

        pred_aug = augment_pred_with_gt_prefix(gt, pred, args.prefix_k)
        d, fde, n = ade_fde_first_horizon(gt, pred_aug, args.horizon)

        if n == 0 or not np.isfinite(fde):
            reason = "n_eval=0"
            skipped.append((i, reason))
            per_sample.append({"index": i, "ade": None, "fde": None, "n_eval": 0, "skip": reason})
            continue

        ade = float(d.mean())
        fde_f = float(fde)
        ades.append(ade)
        fdes.append(fde_f)
        row: Dict[str, Any] = {
            "index": i,
            "ade": ade,
            "fde": fde_f,
            "n_eval": n,
            "prefix_k": args.prefix_k,
            "horizon": args.horizon,
            "gt_len": len(gt),
            "pred_raw_len": len(pred),
            "pred_aug_len": int(pred_aug.shape[0]),
        }
        if denorm_image and denorm_wh:
            row["denorm_image"] = denorm_image
            row["image_wh"] = {"w": denorm_wh[0], "h": denorm_wh[1]}
        per_sample.append(row)

    n_ok = len(ades)
    n_total = len(data)
    print(f"merged_json: {args.merged_json}")
    if args.no_denorm:
        print("denorm: off (coordinates used as-is)")
    else:
        print("denorm: x_px=x/1000*w, y_px=y/1000*h (w,h from first user image in messages)")
    print(
        f"pred_aug = GT[:{args.prefix_k}] + output_points; "
        f"metrics on first min(horizon={args.horizon}, len(gt), len(pred_aug)) steps"
    )
    print(f"samples: {n_ok} evaluated / {n_total} total, skipped: {len(skipped)}")
    if n_ok:
        print(f"ADE (mean over samples): {float(np.mean(ades)):.6f}")
        print(f"FDE (mean over samples): {float(np.mean(fdes)):.6f}")
        print(f"ADE (median): {float(np.median(ades)):.6f}")
        print(f"FDE (median): {float(np.median(fdes)):.6f}")
    else:
        print("No valid samples for metrics.")
        sys.exit(1)

    if skipped and len(skipped) <= 20:
        for idx, r in skipped:
            print(f"  skip[{idx}]: {r}")
    elif skipped:
        print(f"  (first 10 skips): {skipped[:10]}")

    if args.per_sample_json:
        out = {
            "merged_json": str(args.merged_json),
            "prefix_k": args.prefix_k,
            "horizon": args.horizon,
            "mean_ade": float(np.mean(ades)),
            "mean_fde": float(np.mean(fdes)),
            "median_ade": float(np.median(ades)),
            "median_fde": float(np.median(fdes)),
            "n_evaluated": n_ok,
            "n_skipped": len(skipped),
            "per_sample": per_sample,
        }
        args.per_sample_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_sample_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Wrote per-sample details -> {args.per_sample_json}")


if __name__ == "__main__":
    main()
