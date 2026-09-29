#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 6 Cross-Camera RAW Manifest Audit
# Timestamp: 2026-09-24 | Author: Guozh | Version: Phase6-Manifest-v0.1.0
# =============================================================================
"""Audit a CSV manifest before using an external paired RAW dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json

REQUIRED_COLUMNS = {
    "scene_id",
    "short_path",
    "long_path",
    "camera",
    "iso",
    "short_exposure_s",
    "long_exposure_s",
    "bayer_pattern",
    "split",
}
VALID_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}
VALID_SPLITS = {"train", "val", "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 外部 RAW 数据 manifest 审计")
    parser.add_argument("manifest", type=Path, help="UTF-8 CSV manifest")
    parser.add_argument("--data-root", type=Path, help="相对路径的根目录，默认使用 manifest 所在目录")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON 报告路径，默认写入 outputs/phase6_generalization",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data_root = (args.data_root or manifest_path.parent).resolve()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"manifest 缺少列: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest 为空")

    errors: list[str] = []
    split_scenes: defaultdict[str, set[str]] = defaultdict(set)
    camera_scenes: defaultdict[str, set[str]] = defaultdict(set)
    records: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=2):
        scene = row["scene_id"].strip()
        camera = row["camera"].strip()
        split = row["split"].strip().lower()
        pattern = row["bayer_pattern"].strip().upper()
        if not scene or not camera:
            errors.append(f"第 {row_number} 行 scene_id/camera 为空")
            continue
        if split not in VALID_SPLITS:
            errors.append(f"第 {row_number} 行 split 无效: {split}")
        if pattern not in VALID_PATTERNS:
            errors.append(f"第 {row_number} 行 Bayer pattern 无效: {pattern}")
        try:
            iso = int(row["iso"])
            short_exposure = float(row["short_exposure_s"])
            long_exposure = float(row["long_exposure_s"])
            if iso <= 0 or short_exposure <= 0 or long_exposure <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"第 {row_number} 行 ISO/曝光时间必须为正数")
            continue
        short_path = (data_root / row["short_path"]).resolve()
        long_path = (data_root / row["long_path"]).resolve()
        if not short_path.is_file():
            errors.append(f"第 {row_number} 行短曝光不存在: {short_path}")
        if not long_path.is_file():
            errors.append(f"第 {row_number} 行长曝光不存在: {long_path}")
        split_scenes[split].add(scene)
        camera_scenes[camera].add(scene)
        records.append(
            {
                "scene_id": scene,
                "camera": camera,
                "split": split,
                "iso": iso,
                "short_exposure_s": short_exposure,
                "long_exposure_s": long_exposure,
                "exposure_gain": long_exposure / short_exposure,
                "bayer_pattern": pattern,
                "short_exists": short_path.is_file(),
                "long_exists": long_path.is_file(),
            }
        )

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_scenes[left] & split_scenes[right]
        if overlap:
            errors.append(f"场景泄漏: {left}/{right} 交集={sorted(overlap)[:10]}")
    if errors:
        raise RuntimeError("\n".join(errors[:30]) + ("\n..." if len(errors) > 30 else ""))

    summary = {
        "experiment_version": "Phase6-Manifest-v0.1.0",
        "manifest": str(manifest_path),
        "data_root": str(data_root),
        "rows": len(records),
        "scenes_by_split": {key: len(value) for key, value in split_scenes.items()},
        "cameras": {key: len(value) for key, value in camera_scenes.items()},
        "isos": sorted({int(item["iso"]) for item in records}),
        "bayer_patterns": sorted({str(item["bayer_pattern"]) for item in records}),
        "exposure_gain_min": min(float(item["exposure_gain"]) for item in records),
        "exposure_gain_max": max(float(item["exposure_gain"]) for item in records),
        "records": records,
        "status": "PASS",
    }
    output = args.output or (
        Path(__file__).resolve().parents[1]
        / "outputs/phase6_generalization/manifest_audit.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Phase 6 RAW manifest 审计")
    print(f"样本数: {len(records)}")
    print(f"场景数: {summary['scenes_by_split']}")
    print(f"相机: {summary['cameras']}")
    print(f"ISO: {summary['isos']}，Bayer: {summary['bayer_patterns']}")
    print(
        f"曝光增益范围: {summary['exposure_gain_min']:.4f}"
        f" - {summary['exposure_gain_max']:.4f}"
    )
    print(f"审计报告: {output}")
    print("Phase 6 RAW manifest 审计: PASS")


if __name__ == "__main__":
    main()
