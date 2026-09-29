#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 1 Multi-scene DPC Audit
# Timestamp: 2026-09-28 | Author: Guozh | Version: Phase1-DPC-Batch-v0.1.0
# =============================================================================
"""Audit DPC trigger rate and fixed-coordinate recurrence across SID scenes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from data.sid_torch_dataset import sid_scene_id  # noqa: E402
from isp import DpcConfig, correct_defective_pixels_packed, linear_to_srgb, reinhard_tone_map  # noqa: E402
from isp.raw_io import load_normalized_raw  # noqa: E402


CHANNEL_NAMES = ("R", "G1", "G2", "B")


def _parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("pair-indices不能为空")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("pair-indices不能包含重复值")
    return indices


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError("未找到中文字体，请安装fonts-noto-cjk")


def _packed_preview(packed: np.ndarray, gain: float) -> np.ndarray:
    camera_rgb = np.stack(
        [packed[..., 0], 0.5 * (packed[..., 1] + packed[..., 2]), packed[..., 3]],
        axis=-1,
    )
    display_linear = reinhard_tone_map(np.maximum(camera_rgb * gain, 0.0), white_point=4.0)
    return np.clip(linear_to_srgb(display_linear) * 255.0, 0.0, 255.0).astype(np.uint8)


def _overlay_mask(rgb: np.ndarray, packed_mask: np.ndarray) -> np.ndarray:
    marked = rgb.copy()
    display_mask = np.any(packed_mask, axis=-1).astype(np.uint8)
    display_mask = cv2.dilate(display_mask, np.ones((5, 5), np.uint8)) > 0
    marked[display_mask] = (255, 32, 32)
    return marked


def _preview_panel(rgb: np.ndarray, title: str, width: int = 560) -> np.ndarray:
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 44), fill=(0, 0, 0))
    draw.text((10, 7), title, font=_font(19), fill=(255, 255, 255))
    return np.asarray(image)


def _write_contact_sheet(items: list[dict[str, object]], path: Path) -> None:
    panels = []
    for item in items:
        preview = _packed_preview(item["packed"], item["gain"])
        marked = _overlay_mask(preview, item["mask"])
        title = (
            f"pair {item['pair_index']}｜场景 {item['scene_id']}｜ISO {item['iso']}｜"
            f"触发 {item['count']} ({item['ratio']:.6%})"
        )
        panels.append(_preview_panel(marked, title))
    columns = 2
    if len(panels) % columns:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[index : index + columns]) for index in range(0, len(panels), columns)]
    canvas = np.vstack(rows)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入多场景审计图: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 DPC多场景触发率与固定坐标复现审计")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument(
        "--pair-indices",
        type=_parse_indices,
        default=_parse_indices("0,14,74,86,98,123,542,556,570"),
    )
    parser.add_argument("--minimum-recurrence", type=int, default=3)
    parser.add_argument("--absolute-threshold", type=float, default=0.002)
    parser.add_argument("--relative-threshold", type=float, default=0.05)
    parser.add_argument("--mad-scale", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase1_raw_frontend/dpc/batch_audit",
    )
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    if args.minimum_recurrence < 2 or args.minimum_recurrence > len(args.pair_indices):
        raise ValueError("minimum-recurrence必须在2和样本数之间")
    for index in args.pair_indices:
        if not 0 <= index < len(pairs):
            raise IndexError(f"pair-index {index} outside [0, {len(pairs)})")

    config = DpcConfig(
        absolute_threshold=args.absolute_threshold,
        relative_threshold=args.relative_threshold,
        mad_scale=args.mad_scale,
    )
    recurrence: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    rows: list[dict[str, object]] = []
    preview_items: list[dict[str, object]] = []

    for sequence, pair_index in enumerate(args.pair_indices, start=1):
        pair = pairs[pair_index]
        frame = load_normalized_raw(pair.short_path)
        if expected_shape is None:
            expected_shape = frame.packed.shape
            recurrence = np.zeros(expected_shape, dtype=np.uint16)
        elif frame.packed.shape != expected_shape:
            raise ValueError("批量固定坐标审计要求所有RAW具有相同传感器尺寸")

        result = correct_defective_pixels_packed(frame.packed, config)
        recurrence += result.defect_mask.astype(np.uint16)
        channel_counts = [int(np.count_nonzero(result.defect_mask[..., channel])) for channel in range(4)]
        count = int(sum(channel_counts))
        ratio = count / result.defect_mask.size
        scene_id = sid_scene_id(pair.short_path)
        rows.append(
            {
                "pair_index": pair_index,
                "scene_id": scene_id,
                "iso": pair.iso,
                "exposure_gain": pair.exposure_gain,
                "total_count": count,
                "trigger_ratio": ratio,
                **{f"{name}_count": channel_counts[channel] for channel, name in enumerate(CHANNEL_NAMES)},
            }
        )
        preview_items.append(
            {
                "pair_index": pair_index,
                "scene_id": scene_id,
                "iso": pair.iso,
                "gain": min(pair.exposure_gain, 300.0),
                "packed": frame.packed,
                "mask": result.defect_mask,
                "count": count,
                "ratio": ratio,
            }
        )
        print(
            f"完成 {sequence}/{len(args.pair_indices)}: pair {pair_index}，场景 {scene_id}，"
            f"ISO {pair.iso}，触发 {count} ({ratio:.8%})"
        )

    assert recurrence is not None
    persistent_mask = recurrence >= args.minimum_recurrence
    coordinates = np.argwhere(persistent_mask)
    persistent = sorted(
        (
            {
                "packed_row": int(row),
                "packed_col": int(col),
                "channel": CHANNEL_NAMES[int(channel)],
                "recurrence": int(recurrence[row, col, channel]),
            }
            for row, col, channel in coordinates
        ),
        key=lambda item: (-item["recurrence"], item["channel"], item["packed_row"], item["packed_col"]),
    )
    recurrence_histogram = {
        str(count): int(np.count_nonzero(recurrence == count))
        for count in range(1, len(args.pair_indices) + 1)
        if np.any(recurrence == count)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dpc_multiscene_audit.json"
    csv_path = args.output_dir / "dpc_multiscene_summary.csv"
    preview_path = args.output_dir / "dpc_multiscene_trigger_map.png"
    _write_contact_sheet(preview_items, preview_path)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "experiment_version": "Phase1-DPC-Batch-v0.1.0",
        "config": {
            "absolute_threshold": config.absolute_threshold,
            "relative_threshold": config.relative_threshold,
            "mad_scale": config.mad_scale,
            "minimum_recurrence": args.minimum_recurrence,
        },
        "sample_count": len(rows),
        "independent_scene_count": len({row["scene_id"] for row in rows}),
        "mean_trigger_ratio": float(np.mean([row["trigger_ratio"] for row in rows])),
        "maximum_trigger_ratio": float(np.max([row["trigger_ratio"] for row in rows])),
        "recurrence_histogram": recurrence_histogram,
        "persistent_candidate_count": len(persistent),
        "persistent_candidates": persistent,
        "samples": rows,
        "preview": str(preview_path.resolve()),
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("\nDPC多场景审计汇总")
    print(f"样本/独立场景: {len(rows)}/{report['independent_scene_count']}")
    print(f"平均/最大触发比例: {report['mean_trigger_ratio']:.8%} / {report['maximum_trigger_ratio']:.8%}")
    print(f"复现次数直方图: {recurrence_histogram}")
    print(f"至少复现{args.minimum_recurrence}次的固定坐标候选: {len(persistent)}")
    for item in persistent[:12]:
        print(
            f"  ({item['packed_row']}, {item['packed_col']}, {item['channel']}): "
            f"{item['recurrence']}/{len(rows)}"
        )
    print(f"中文触发图: {preview_path.resolve()}")
    print(f"中文CSV汇总: {csv_path.resolve()}")
    print(f"JSON报告: {json_path.resolve()}")
    print("注意：固定坐标复现是坏点证据，不是绝对真值；仍需结合红色触发图检查场景结构误检。")


if __name__ == "__main__":
    main()
