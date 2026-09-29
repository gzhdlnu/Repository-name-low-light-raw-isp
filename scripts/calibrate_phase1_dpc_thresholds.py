#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 1 DPC Threshold Calibration
# Timestamp: 2026-09-28 | Author: Guozh | Version: Phase1-DPC-Cal-v0.1.0
# =============================================================================
"""Compare conservative DPC threshold profiles on synthetic and real SID RAW."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import DpcConfig, correct_defective_pixels_packed  # noqa: E402
from isp.raw_io import load_normalized_raw  # noqa: E402


PROFILES = {
    "当前基线": DpcConfig(absolute_threshold=0.002, relative_threshold=0.05, mad_scale=6.0),
    "增强局部噪声抑制": DpcConfig(absolute_threshold=0.002, relative_threshold=0.05, mad_scale=9.0),
    "高ISO保守": DpcConfig(absolute_threshold=0.003, relative_threshold=0.075, mad_scale=10.0),
    "高ISO严格": DpcConfig(absolute_threshold=0.004, relative_threshold=0.10, mad_scale=12.0),
}


def _synthetic_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width = 192, 256
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= width - 1
    yy /= height - 1
    base = 0.06 + 0.30 * xx + 0.14 * yy
    base += 0.02 * np.sin(12.0 * np.pi * xx) * np.cos(6.0 * np.pi * yy)
    base[:, width // 2 :] += 0.10
    clean = base[..., None] * np.asarray((1.08, 1.0, 0.98, 0.90), np.float32)
    clean += rng.normal(0.0, 0.0008, (height, width, 4)).astype(np.float32)
    clean = np.clip(clean, 0.0, 0.85).astype(np.float32)

    corrupted = clean.copy()
    truth = np.zeros(clean.shape, dtype=bool)
    candidates = np.arange((height - 4) * (width - 4) * 4)
    chosen = rng.choice(candidates, size=320, replace=False)
    channels = chosen % 4
    spatial = chosen // 4
    rows = spatial // (width - 4) + 2
    cols = spatial % (width - 4) + 2
    for index, (row, col, channel) in enumerate(zip(rows, cols, channels)):
        corrupted[row, col, channel] = 1.0 if index < 160 else 0.0
        truth[row, col, channel] = True
    return clean, corrupted, truth


def _synthetic_metrics(config: DpcConfig, seed: int) -> dict[str, float]:
    clean, corrupted, truth = _synthetic_case(seed)
    result = correct_defective_pixels_packed(corrupted, config)
    true_positive = int(np.count_nonzero(result.defect_mask & truth))
    false_positive = int(np.count_nonzero(result.defect_mask & ~truth))
    false_negative = int(np.count_nonzero(~result.defect_mask & truth))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    repair_mae = float(np.mean(np.abs(result.corrected[truth] - clean[truth])))
    return {"precision": precision, "recall": recall, "repair_mae": repair_mae}


def main() -> None:
    parser = argparse.ArgumentParser(description="标定DPC阈值在合成坏点和不同ISO真实RAW上的取舍")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument(
        "--pair-indices",
        default="0,123,542,556,570",
        help="默认选择两个ISO 200和三个ISO 8000代表场景",
    )
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase1_raw_frontend/dpc/calibration",
    )
    args = parser.parse_args()

    indices = [int(value.strip()) for value in args.pair_indices.split(",") if value.strip()]
    pairs = load_sid_list(args.sid_list)
    frames = []
    for pair_index in indices:
        if not 0 <= pair_index < len(pairs):
            raise IndexError(f"pair-index {pair_index} outside [0, {len(pairs)})")
        pair = pairs[pair_index]
        frames.append((pair_index, pair.iso, load_normalized_raw(pair.short_path).packed))

    rows: list[dict[str, object]] = []
    for profile_name, config in PROFILES.items():
        synthetic = _synthetic_metrics(config, args.seed)
        profile_rows = []
        for pair_index, iso, packed in frames:
            result = correct_defective_pixels_packed(packed, config)
            ratio = result.corrected_count / result.defect_mask.size
            profile_rows.append(
                {
                    "profile": profile_name,
                    "pair_index": pair_index,
                    "iso": iso,
                    "trigger_count": result.corrected_count,
                    "trigger_ratio": ratio,
                }
            )
        rows.extend(profile_rows)
        print(f"\n{profile_name}")
        print(
            f"合成 precision/recall/repair MAE: {synthetic['precision']:.5f} / "
            f"{synthetic['recall']:.5f} / {synthetic['repair_mae']:.8f}"
        )
        for iso in sorted({row["iso"] for row in profile_rows}):
            ratios = [row["trigger_ratio"] for row in profile_rows if row["iso"] == iso]
            print(f"ISO {iso} 真实RAW触发比例 mean/max: {np.mean(ratios):.8%} / {np.max(ratios):.8%}")

    summaries = []
    for profile_name, config in PROFILES.items():
        synthetic = _synthetic_metrics(config, args.seed)
        profile_rows = [row for row in rows if row["profile"] == profile_name]
        summaries.append(
            {
                "profile": profile_name,
                "absolute_threshold": config.absolute_threshold,
                "relative_threshold": config.relative_threshold,
                "mad_scale": config.mad_scale,
                "synthetic_precision": synthetic["precision"],
                "synthetic_recall": synthetic["recall"],
                "synthetic_repair_mae": synthetic["repair_mae"],
                "iso200_mean_trigger_ratio": float(
                    np.mean([row["trigger_ratio"] for row in profile_rows if row["iso"] == 200])
                ),
                "iso200_max_trigger_ratio": float(
                    np.max([row["trigger_ratio"] for row in profile_rows if row["iso"] == 200])
                ),
                "iso8000_mean_trigger_ratio": float(
                    np.mean([row["trigger_ratio"] for row in profile_rows if row["iso"] == 8000])
                ),
                "iso8000_max_trigger_ratio": float(
                    np.max([row["trigger_ratio"] for row in profile_rows if row["iso"] == 8000])
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dpc_threshold_profiles.csv"
    json_path = args.output_dir / "dpc_threshold_profiles.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    json_path.write_text(
        json.dumps(
            {
                "experiment_version": "Phase1-DPC-Cal-v0.1.0",
                "pairs": indices,
                "summaries": summaries,
                "per_pair": rows,
                "selection_rule": (
                    "先保证合成precision/recall均不低于0.95，再比较真实RAW触发率；"
                    "真实触发率不是坏点真值，不能单独用于自动选型。"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"\n中文CSV汇总: {csv_path.resolve()}")
    print(f"JSON报告: {json_path.resolve()}")
    print("本脚本只提供阈值取舍证据，不自动把最低触发率配置判为最佳。")


if __name__ == "__main__":
    main()
