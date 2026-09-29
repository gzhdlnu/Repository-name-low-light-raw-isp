#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 2 AWB Estimator Check
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase2-AWB-Check-v0.1.0
# =============================================================================
"""Compare metadata WB and statistical AWB estimators on synthetic and SID RAW."""

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
from isp import gray_edge_gains, gray_world_gains, shades_of_gray_gains  # noqa: E402
from isp.raw_io import load_normalized_raw  # noqa: E402


METHODS = {
    "Gray World": gray_world_gains,
    "Shades-of-Gray": shades_of_gray_gains,
    "Gray Edge": gray_edge_gains,
}
CHANNEL_NAMES = ("R", "G1", "G2", "B")


def _synthetic_cast(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    height, width = 256, 320
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx /= width - 1
    yy /= height - 1
    scene = 0.08 + 0.34 * xx + 0.16 * yy
    scene += 0.04 * np.sin(8 * np.pi * xx) * np.cos(5 * np.pi * yy)
    scene[:, width // 2 :] += 0.10
    scene = np.clip(scene, 0.01, 0.85)
    neutral = scene[..., None] * np.asarray((1.0, 1.0, 1.0, 1.0), np.float32)
    cast = np.asarray((1.42, 1.0, 1.0, 0.72), np.float32)
    observed = neutral * cast.reshape(1, 1, 4)
    observed += rng.normal(0.0, 0.0008, observed.shape).astype(np.float32)
    return np.clip(observed, 0.0, 1.0), cast


def _gain_error(estimated: tuple[float, float, float, float], expected: np.ndarray) -> float:
    estimated_array = np.asarray(estimated, np.float32)
    expected_normalized = expected / (0.5 * (expected[1] + expected[2]))
    return float(np.mean(np.abs(estimated_array - 1.0 / expected_normalized)))


def _parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("pair-indices必须是非空且不重复的整数列表")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 AWB统计估计器检查")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-indices", type=_parse_indices, default=_parse_indices("0,14,74,123,542,556,570"))
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase2_color_tone/awb")
    args = parser.parse_args()

    synthetic, cast = _synthetic_cast(args.seed)
    synthetic_rows = []
    print("Phase 2 AWB合成色偏回收")
    for name, estimator in METHODS.items():
        gains = estimator(synthetic)
        error = _gain_error(gains, cast)
        synthetic_rows.append({"method": name, "gain_error": error, **{f"{channel}_gain": gains[index] for index, channel in enumerate(CHANNEL_NAMES)}})
        print(f"{name}: gains={tuple(round(value, 5) for value in gains)}, 反向增益MAE={error:.6f}")

    pairs = load_sid_list(args.sid_list)
    rows = []
    for sequence, pair_index in enumerate(args.pair_indices, start=1):
        if not 0 <= pair_index < len(pairs):
            raise IndexError(f"pair-index {pair_index} outside [0, {len(pairs)})")
        pair = pairs[pair_index]
        frame = load_normalized_raw(pair.short_path)
        # AWB estimation follows the actual low-light pipeline: estimate on
        # normalized linear RAW after the requested exposure compensation,
        # while WB gains themselves remain independent of display tone mapping.
        applied_gain = min(pair.exposure_gain, 300.0)
        awb_input = np.clip(frame.packed * applied_gain, 0.0, 1.0).astype(np.float32)
        metadata = frame.white_balance_gains
        common = {
            "pair_index": pair_index,
            "iso": pair.iso,
            "exposure_gain": pair.exposure_gain,
            "applied_exposure_gain": applied_gain,
            "metadata_R_gain": metadata[0],
            "metadata_G1_gain": metadata[1],
            "metadata_G2_gain": metadata[2],
            "metadata_B_gain": metadata[3],
        }
        print(f"完成 {sequence}/{len(args.pair_indices)}: pair {pair_index}，ISO {pair.iso}")
        for name, estimator in METHODS.items():
            try:
                gains = estimator(awb_input)
                error_to_metadata = float(np.mean(np.abs(np.asarray(gains) - np.asarray(metadata))))
                row = {
                    **common,
                    "method": name,
                    "R_gain": gains[0],
                    "G1_gain": gains[1],
                    "G2_gain": gains[2],
                    "B_gain": gains[3],
                    "mean_abs_diff_to_metadata": error_to_metadata,
                    "valid": True,
                }
            except ValueError as exc:
                row = {**common, "method": name, "valid": False, "error": str(exc)}
            rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "awb_estimators_sid_summary.csv"
    json_path = args.output_dir / "awb_estimators_report.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "experiment_version": "Phase2-AWB-Check-v0.1.0",
                "synthetic": {"cast": cast.tolist(), "rows": synthetic_rows},
                "sid_pair_indices": args.pair_indices,
                "rows": rows,
                "interpretation": "SID没有逐像素灰点真值，统计AWB与metadata差异只用于稳定性审计，不代表某个方法绝对正确。",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("\nSID AWB统计审计")
    for name in METHODS:
        valid = [row for row in rows if row["method"] == name and row["valid"]]
        if valid:
            diffs = [row["mean_abs_diff_to_metadata"] for row in valid]
            print(f"{name}: 与metadata gain平均绝对差={np.mean(diffs):.6f}, 最大={np.max(diffs):.6f}")
    print(f"中文CSV汇总: {csv_path.resolve()}")
    print(f"JSON报告: {json_path.resolve()}")
    print("注意：统计AWB依赖场景中性物体；大片彩色物体、多光源和极暗场景可能导致失败。")


if __name__ == "__main__":
    main()
