#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 4 SID Dataset Smoke Test
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase4-DataCheck-v0.1.0
# =============================================================================
"""Validate SID split isolation and one aligned packed-RAW training patch."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from data.sid_torch_dataset import SidRawPatchDataset, sid_scene_id  # noqa: E402
from isp import linear_to_srgb, reinhard_tone_map  # noqa: E402


def _scene_set(list_path: Path) -> set[str]:
    return {sid_scene_id(pair.short_path) for pair in load_sid_list(list_path)}


def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(tensor.min()),
        "maximum": float(tensor.max()),
        "mean": float(tensor.mean()),
        "standard_deviation": float(tensor.std()),
        "finite": bool(torch.isfinite(tensor).all()),
        "near_black_ratio": float((tensor <= 0.01).float().mean()),
        "near_white_ratio": float((tensor >= 0.99).float().mean()),
    }


def _packed_preview(tensor: torch.Tensor) -> np.ndarray:
    packed = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    camera_rgb = np.stack(
        [packed[..., 0], 0.5 * (packed[..., 1] + packed[..., 2]), packed[..., 3]],
        axis=-1,
    )
    display_linear = reinhard_tone_map(camera_rgb, white_point=4.0)
    return np.clip(linear_to_srgb(display_linear) * 255.0, 0.0, 255.0).astype(np.uint8)


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


def _title_panel(rgb: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font_size = 21
    font = _font(font_size)
    while font_size > 12 and draw.textbbox((0, 0), title, font=font)[2] > image.width - 20:
        font_size -= 1
        font = _font(font_size)
    draw.rectangle((0, 0, image.width, 40), fill=(0, 0, 0))
    draw.text((10, 6), title, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def _write_preview(input_tensor: torch.Tensor, target_tensor: torch.Tensor, path: Path) -> None:
    input_rgb = _packed_preview(input_tensor)
    target_rgb = _packed_preview(target_tensor)
    absolute_difference = torch.mean(torch.abs(input_tensor - target_tensor), dim=0).numpy()
    scale = max(float(np.percentile(absolute_difference, 99)), 1e-6)
    heatmap = cv2.applyColorMap(
        np.clip(absolute_difference / scale * 255.0, 0.0, 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    comparison = np.hstack(
        [
            _title_panel(input_rgb, "短曝光输入（曝光补偿后）"),
            _title_panel(target_rgb, "长曝光 RAW 监督目标"),
            _title_panel(heatmap, "绝对误差热力图"),
        ]
    )
    if not cv2.imwrite(str(path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入预览图: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Phase 4 SID RAW patch 数据管线")
    parser.add_argument("--train-list", type=Path, default=ROOT / "data/raw/SID/Sony_train_list.txt")
    parser.add_argument("--val-list", type=Path, default=ROOT / "data/raw/SID/Sony_val_list.txt")
    parser.add_argument("--test-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase4_learning" / "data_check",
    )
    args = parser.parse_args()

    split_scenes = {
        "train": _scene_set(args.train_list),
        "val": _scene_set(args.val_list),
        "test": _scene_set(args.test_list),
    }
    overlaps = {
        "train_val": sorted(split_scenes["train"] & split_scenes["val"]),
        "train_test": sorted(split_scenes["train"] & split_scenes["test"]),
        "val_test": sorted(split_scenes["val"] & split_scenes["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"SID 官方划分存在场景泄漏: {overlaps}")

    dataset = SidRawPatchDataset(
        args.train_list,
        patch_size=args.patch_size,
        patches_per_pair=1,
        augment=False,
        max_exposure_gain=300.0,
        seed=args.seed,
    )
    if not 0 <= args.pair_index < len(dataset):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(dataset)})")
    sample = dataset[args.pair_index]
    loader = DataLoader(Subset(dataset, [args.pair_index]), batch_size=1, num_workers=0)
    batch = next(iter(loader))
    expected_shape = (1, 4, args.patch_size, args.patch_size)
    if tuple(batch["input"].shape) != expected_shape or tuple(batch["target"].shape) != expected_shape:
        raise RuntimeError(f"DataLoader shape 不正确，期望 {expected_shape}")

    input_summary = _tensor_summary(sample["input"])
    target_summary = _tensor_summary(sample["target"])
    if not input_summary["finite"] or not target_summary["finite"]:
        raise RuntimeError("训练 patch 包含 NaN 或 Inf")
    report = {
        "experiment_version": "Phase4-DataCheck-v0.1.0",
        "description_zh": "SID 学习型 RAW 降噪数据管线单 patch 检查",
        "split_scene_counts": {name: len(value) for name, value in split_scenes.items()},
        "split_scene_overlaps": overlaps,
        "pair_index": sample["pair_index"],
        "scene_id": sample["scene_id"],
        "iso": sample["iso"],
        "physical_exposure_gain": sample["physical_exposure_gain"],
        "applied_exposure_gain": sample["applied_exposure_gain"],
        "packed_patch_coordinates": {
            "top": sample["crop_top"],
            "left": sample["crop_left"],
            "size": args.patch_size,
        },
        "input": input_summary,
        "target": target_summary,
        "dataloader_batch_shape": list(batch["input"].shape),
        "short_path": sample["short_path"],
        "long_path": sample["long_path"],
        "preview_note_zh": "预览直接组合 packed R/G/B，仅用于检查裁剪对齐，不代表最终 ISP 颜色",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{args.pair_index:04d}_patch_{args.patch_size}"
    preview_path = args.output_dir / f"{stem}_preview.png"
    report_path = args.output_dir / f"{stem}_report.json"
    _write_preview(sample["input"], sample["target"], preview_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"官方场景数: train={len(split_scenes['train'])}, "
        f"val={len(split_scenes['val'])}, test={len(split_scenes['test'])}"
    )
    print("场景交集: train/val=0, train/test=0, val/test=0")
    print(
        f"样本: pair={sample['pair_index']}, scene={sample['scene_id']}, "
        f"ISO={sample['iso']}, gain={sample['applied_exposure_gain']:.3f}"
    )
    print(f"DataLoader: input={tuple(batch['input'].shape)}, target={tuple(batch['target'].shape)}")
    print(
        f"输入范围=[{input_summary['minimum']:.6f}, {input_summary['maximum']:.6f}], "
        f"均值={input_summary['mean']:.6f}, 近黑={input_summary['near_black_ratio']:.2%}"
    )
    print(
        f"目标范围=[{target_summary['minimum']:.6f}, {target_summary['maximum']:.6f}], "
        f"均值={target_summary['mean']:.6f}, 近黑={target_summary['near_black_ratio']:.2%}"
    )
    print(f"中文对齐预览: {preview_path}")
    print(f"检查报告: {report_path}")
    print("Phase 4 数据管线检查: PASS")


if __name__ == "__main__":
    main()
