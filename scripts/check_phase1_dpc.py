#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 1 Defective Pixel Correction Check
# Timestamp: 2026-09-28 | Author: Guozh | Version: Phase1-DPC-Check-v0.1.0
# =============================================================================
"""Validate DPC with controlled defects, then audit its trigger rate on SID RAW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import DpcConfig, correct_defective_pixels_packed, linear_to_srgb, reinhard_tone_map  # noqa: E402
from isp.raw_io import load_normalized_raw  # noqa: E402


CHANNEL_NAMES = ("R", "G1", "G2", "B")


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError("未找到中文字体，请安装 fonts-noto-cjk")


def _synthetic_packed(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= max(width - 1, 1)
    yy /= max(height - 1, 1)
    base = 0.08 + 0.28 * xx + 0.12 * yy
    base += 0.018 * np.sin(2.0 * np.pi * xx * 7.0) * np.cos(2.0 * np.pi * yy * 3.0)
    base[:, width // 2 :] += 0.12
    channel_scales = np.asarray((1.08, 1.0, 0.98, 0.90), np.float32)
    packed = base[..., None] * channel_scales.reshape(1, 1, 4)
    packed += rng.normal(0.0, 0.0008, packed.shape).astype(np.float32)
    return np.clip(packed, 0.0, 0.85).astype(np.float32)


def _inject_defects(
    clean: np.ndarray,
    hot_count: int,
    dead_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width, channels = clean.shape
    candidates = np.arange((height - 4) * (width - 4) * channels)
    chosen = rng.choice(candidates, size=hot_count + dead_count, replace=False)
    local_channel = chosen % channels
    spatial = chosen // channels
    rows = spatial // (width - 4) + 2
    cols = spatial % (width - 4) + 2

    corrupted = clean.copy()
    truth_hot = np.zeros(clean.shape, dtype=bool)
    truth_dead = np.zeros(clean.shape, dtype=bool)
    for index, (row, col, channel) in enumerate(zip(rows, cols, local_channel)):
        if index < hot_count:
            corrupted[row, col, channel] = 1.0
            truth_hot[row, col, channel] = True
        else:
            corrupted[row, col, channel] = 0.0
            truth_dead[row, col, channel] = True
    return corrupted, truth_hot, truth_dead


def _classification_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    true_positive = int(np.count_nonzero(predicted & truth))
    false_positive = int(np.count_nonzero(predicted & ~truth))
    false_negative = int(np.count_nonzero(~predicted & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _packed_preview(packed: np.ndarray, gain: float) -> np.ndarray:
    camera_rgb = np.stack(
        [packed[..., 0], 0.5 * (packed[..., 1] + packed[..., 2]), packed[..., 3]],
        axis=-1,
    )
    display_linear = reinhard_tone_map(np.maximum(camera_rgb * gain, 0.0), white_point=4.0)
    return np.clip(linear_to_srgb(display_linear) * 255.0, 0.0, 255.0).astype(np.uint8)


def _mark_mask(rgb: np.ndarray, packed_mask: np.ndarray) -> np.ndarray:
    marked = rgb.copy()
    mask = np.any(packed_mask, axis=-1).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    marked[mask] = (255, 32, 32)
    return marked


def _panel(rgb: np.ndarray, title: str, target_size: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 42), fill=(0, 0, 0))
    draw.text((10, 7), title, font=_font(20), fill=(255, 255, 255))
    return np.asarray(image)


def _write_preview(
    synthetic_before: np.ndarray,
    synthetic_after: np.ndarray,
    synthetic_mask: np.ndarray,
    real_before: np.ndarray,
    real_after: np.ndarray,
    real_mask: np.ndarray,
    gain: float,
    path: Path,
) -> None:
    size = (600, 360)
    synthetic_rgb = _packed_preview(synthetic_before, gain=1.0)
    real_rgb = _packed_preview(real_before, gain=gain)
    rows = (
        (
            _panel(synthetic_rgb, "合成RAW：注入坏点", size),
            _panel(_packed_preview(synthetic_after, 1.0), "合成RAW：DPC修复后", size),
            _panel(_mark_mask(synthetic_rgb, synthetic_mask), "合成RAW：红色为检测位置", size),
        ),
        (
            _panel(real_rgb, "真实SID RAW：DPC前", size),
            _panel(_packed_preview(real_after, gain), "真实SID RAW：DPC后", size),
            _panel(_mark_mask(real_rgb, real_mask), "真实SID RAW：红色为触发位置", size),
        ),
    )
    canvas = np.vstack([np.hstack(row) for row in rows])
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入预览图: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 DPC 合成验收与真实SID触发率审计")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--hot-count", type=int, default=160)
    parser.add_argument("--dead-count", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--absolute-threshold", type=float, default=0.002)
    parser.add_argument("--relative-threshold", type=float, default=0.05)
    parser.add_argument("--mad-scale", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase1_raw_frontend/dpc",
    )
    args = parser.parse_args()

    config = DpcConfig(
        absolute_threshold=args.absolute_threshold,
        relative_threshold=args.relative_threshold,
        mad_scale=args.mad_scale,
    )
    clean = _synthetic_packed(256, 320, args.seed)
    corrupted, truth_hot, truth_dead = _inject_defects(
        clean,
        hot_count=args.hot_count,
        dead_count=args.dead_count,
        seed=args.seed + 1,
    )
    synthetic_result = correct_defective_pixels_packed(corrupted, config)
    truth = truth_hot | truth_dead
    synthetic_metrics = _classification_metrics(synthetic_result.defect_mask, truth)
    synthetic_metrics["repair_mae"] = float(
        np.mean(np.abs(synthetic_result.corrected[truth] - clean[truth]))
    )
    synthetic_metrics["uncorrupted_change_ratio"] = float(
        np.mean(np.abs(synthetic_result.corrected[~truth] - clean[~truth]) > 1e-8)
    )

    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]
    frame = load_normalized_raw(pair.short_path)
    real_result = correct_defective_pixels_packed(frame.packed, config)
    channel_stats = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        count = int(np.count_nonzero(real_result.defect_mask[..., channel]))
        channel_stats[name] = {
            "corrected_count": count,
            "trigger_ratio": count / real_result.defect_mask[..., channel].size,
            "hot_count": int(np.count_nonzero(real_result.hot_mask[..., channel])),
            "dead_count": int(np.count_nonzero(real_result.dead_mask[..., channel])),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / f"pair_{args.pair_index:04d}_dpc_check.png"
    report_path = args.output_dir / f"pair_{args.pair_index:04d}_dpc_report.json"
    _write_preview(
        corrupted,
        synthetic_result.corrected,
        synthetic_result.defect_mask,
        frame.packed,
        real_result.corrected,
        real_result.defect_mask,
        min(pair.exposure_gain, 300.0),
        preview_path,
    )

    report = {
        "experiment_version": "Phase1-DPC-Check-v0.1.0",
        "config": {
            "absolute_threshold": config.absolute_threshold,
            "relative_threshold": config.relative_threshold,
            "mad_scale": config.mad_scale,
        },
        "synthetic": {
            "shape": list(clean.shape),
            "injected_hot": int(np.count_nonzero(truth_hot)),
            "injected_dead": int(np.count_nonzero(truth_dead)),
            **synthetic_metrics,
        },
        "real_sid": {
            "pair_index": args.pair_index,
            "input": str(pair.short_path),
            "iso": pair.iso,
            "exposure_gain": pair.exposure_gain,
            "shape": list(frame.packed.shape),
            "corrected_count": real_result.corrected_count,
            "trigger_ratio": float(np.mean(real_result.defect_mask)),
            "channels": channel_stats,
        },
        "preview": str(preview_path.resolve()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("Phase 1 DPC 合成验收")
    print(
        f"precision={synthetic_metrics['precision']:.5f}, "
        f"recall={synthetic_metrics['recall']:.5f}, F1={synthetic_metrics['f1']:.5f}"
    )
    print(
        f"修复MAE={synthetic_metrics['repair_mae']:.8f}, "
        f"正常像素改动比例={synthetic_metrics['uncorrupted_change_ratio']:.8%}"
    )
    print("真实SID RAW触发统计（无坏点真值，只用于审计误检风险）")
    print(f"pair={args.pair_index}, ISO={pair.iso}, gain={pair.exposure_gain:.3f}")
    print(f"总触发数={real_result.corrected_count}, 总触发比例={np.mean(real_result.defect_mask):.8%}")
    for name in CHANNEL_NAMES:
        stats = channel_stats[name]
        print(
            f"{name}: count={stats['corrected_count']}, ratio={stats['trigger_ratio']:.8%}, "
            f"hot/dead={stats['hot_count']}/{stats['dead_count']}"
        )
    print(f"中文对比图: {preview_path.resolve()}")
    print(f"指标报告: {report_path.resolve()}")

    if synthetic_metrics["recall"] < 0.95 or synthetic_metrics["precision"] < 0.95:
        raise SystemExit("DPC合成验收未通过：precision或recall低于0.95")
    print("Phase 1 DPC合成验收: PASS（真实RAW仍需人工检查红色触发位置）")


if __name__ == "__main__":
    main()
