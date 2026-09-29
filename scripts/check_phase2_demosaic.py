#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 2 Demosaic Benchmark
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase2-Demosaic-Check-v0.1.0
# =============================================================================
"""Compare Bayer demosaic methods on synthetic ground truth and real SID RAW."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import demosaic_bilinear, demosaic_opencv  # noqa: E402
from isp.baseline import pack_bayer  # noqa: E402
from isp.raw_io import load_normalized_raw  # noqa: E402


METHODS = ("bilinear", "vng", "edge_aware")


def _synthetic_rgb(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= max(width - 1, 1)
    yy /= max(height - 1, 1)
    image = np.zeros((height, width, 3), np.float32)
    image[...] = (0.08, 0.10, 0.12)
    image[(yy > 0.10) & (yy < 0.90) & (xx > 0.08) & (xx < 0.92)] = (0.55, 0.18, 0.08)
    image[(yy > 0.22) & (yy < 0.78) & (xx > 0.20) & (xx < 0.80)] = (0.08, 0.58, 0.26)
    image[(np.abs(yy - (0.18 + 0.64 * xx)) < 0.012)] = (0.9, 0.9, 0.9)
    image[(np.abs(xx - 0.68) < 0.008)] = (0.85, 0.14, 0.75)
    texture = np.sin(2 * np.pi * xx * 31.0) * np.sin(2 * np.pi * yy * 17.0)
    image += 0.08 * texture[..., None]
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _rgb_to_bayer(rgb: np.ndarray, pattern: str = "RGGB") -> tuple[np.ndarray, np.ndarray]:
    channels = {
        "R": rgb[..., 0],
        "G": 0.5 * (rgb[..., 1] + rgb[..., 1]),
        "B": rgb[..., 2],
    }
    pattern_channels = {
        "RGGB": ("R", "G", "G", "B"),
        "BGGR": ("B", "G", "G", "R"),
        "GRBG": ("G", "R", "B", "G"),
        "GBRG": ("G", "B", "R", "G"),
    }[pattern]
    mosaic = np.empty(rgb.shape[:2], np.float32)
    for (row, col), name in zip(((0, 0), (0, 1), (1, 0), (1, 1)), pattern_channels):
        mosaic[row::2, col::2] = channels[name][row::2, col::2]
    return mosaic, pack_bayer(mosaic, pattern)


def _gradient_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    pred_gray = np.mean(predicted, axis=-1)
    ref_gray = np.mean(reference, axis=-1)
    pred_grad = np.hypot(np.diff(pred_gray, axis=1, prepend=pred_gray[:, :1]), np.diff(pred_gray, axis=0, prepend=pred_gray[:1, :]))
    ref_grad = np.hypot(np.diff(ref_gray, axis=1, prepend=ref_gray[:, :1]), np.diff(ref_gray, axis=0, prepend=ref_gray[:1, :]))
    return float(np.mean(np.abs(pred_grad - ref_grad)))


def _false_color_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    pred_chroma = np.stack((predicted[..., 0] - predicted[..., 1], predicted[..., 2] - predicted[..., 1]), axis=-1)
    ref_chroma = np.stack((reference[..., 0] - reference[..., 1], reference[..., 2] - reference[..., 1]), axis=-1)
    return float(np.mean(np.abs(pred_chroma - ref_chroma)))


def _evaluate_methods(packed: np.ndarray, reference: np.ndarray, pattern: str, repeats: int) -> list[dict[str, object]]:
    results = []
    for method in METHODS:
        for _ in range(2):
            if method == "bilinear":
                prediction = demosaic_bilinear(packed, pattern)
            else:
                prediction = demosaic_opencv(packed, pattern, method)
        started = time.perf_counter()
        for _ in range(repeats):
            if method == "bilinear":
                prediction = demosaic_bilinear(packed, pattern)
            else:
                prediction = demosaic_opencv(packed, pattern, method)
        elapsed = (time.perf_counter() - started) / repeats
        prediction = np.clip(prediction, 0.0, 1.0)
        mse = float(np.mean((prediction - reference) ** 2))
        psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
        ssim = float(structural_similarity(reference, prediction, channel_axis=-1, data_range=1.0))
        results.append(
            {
                "method": method,
                "psnr": psnr,
                "ssim": ssim,
                "gradient_mae": _gradient_error(prediction, reference),
                "false_color_mae": _false_color_error(prediction, reference),
                "mean_seconds": elapsed,
                "output_min": float(prediction.min()),
                "output_max": float(prediction.max()),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Demosaic方法对比")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase2_color_tone/demosaic")
    args = parser.parse_args()

    reference = _synthetic_rgb(args.height, args.width)
    _, packed = _rgb_to_bayer(reference, "RGGB")
    synthetic_results = _evaluate_methods(packed, reference, "RGGB", args.repeats)
    print("Phase 2 Demosaic合成真值对比")
    print("方法              PSNR(dB)   SSIM    梯度MAE    假色MAE    单次耗时(s)")
    for row in synthetic_results:
        print(f"{row['method']:<16} {row['psnr']:>8.3f} {row['ssim']:>8.5f} {row['gradient_mae']:>10.6f} {row['false_color_mae']:>10.6f} {row['mean_seconds']:>12.5f}")

    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]
    frame = load_normalized_raw(pair.short_path)
    real_rows = []
    print(f"\n真实SID运行时间：pair {args.pair_index}，ISO {pair.iso}，pattern {frame.pattern}")
    for method in METHODS:
        started = time.perf_counter()
        for _ in range(args.repeats):
            if method == "bilinear":
                output = demosaic_bilinear(frame.packed, frame.pattern)
            else:
                output = demosaic_opencv(frame.packed, frame.pattern, method)
        elapsed = (time.perf_counter() - started) / args.repeats
        row = {"method": method, "mean_seconds": elapsed, "output_min": float(output.min()), "output_max": float(output.max())}
        real_rows.append(row)
        print(f"{method:<16} mean={elapsed:.5f}s, range=[{output.min():.5f}, {output.max():.5f}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"pair_{args.pair_index:04d}_demosaic_report.json"
    csv_path = args.output_dir / f"pair_{args.pair_index:04d}_demosaic_summary.csv"
    report = {
        "experiment_version": "Phase2-Demosaic-Check-v0.1.0",
        "synthetic": {"shape": list(reference.shape), "pattern": "RGGB", "rows": synthetic_results},
        "real_sid": {"pair_index": args.pair_index, "iso": pair.iso, "pattern": frame.pattern, "rows": real_rows},
        "limitation": "VNG/Edge-Aware是OpenCV工程后端对比，不等同于严格复现论文MHC；无真实RGB真值时只能比较运行范围和视觉伪影。",
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(synthetic_results[0]))
        writer.writeheader()
        writer.writerows(synthetic_results)
    print(f"中文CSV汇总: {csv_path.resolve()}")
    print(f"JSON报告: {json_path.resolve()}")
    print("结论边界：合成真值用于算法排序；SID真实RAW没有RGB真值，不能仅凭单张图宣布某种Demosaic最好。")


if __name__ == "__main__":
    main()
