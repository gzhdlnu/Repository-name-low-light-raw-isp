#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 1 Lens Shading Correction Check
# Timestamp: 2026-09-28 | Author: Guozh | Version: Phase1-LSC-Check-v0.1.0
# =============================================================================
"""Validate LSC interpolation, flat-field uniformity, noise cost and identity bypass."""

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
from isp import (  # noqa: E402
    LscConfig,
    apply_lens_shading_correction_packed,
    identity_lsc_gain_map,
    linear_to_srgb,
    reinhard_tone_map,
)
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
    raise RuntimeError("未找到中文字体，请安装fonts-noto-cjk")


def _radial_grid(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = 2.0 * xx / max(width - 1, 1) - 1.0
    yy = 2.0 * yy / max(height - 1, 1) - 1.0
    return np.clip((xx * xx + yy * yy) / 2.0, 0.0, 1.0)


def _synthetic_flat_field(
    height: int,
    width: int,
    grid_height: int,
    grid_width: int,
    seed: int,
) -> dict[str, np.ndarray]:
    channel_levels = np.asarray((0.24, 0.22, 0.22, 0.20), np.float32)
    clean = np.broadcast_to(channel_levels, (height, width, 4)).copy()

    radial = _radial_grid(height, width)
    corner_losses = np.asarray((0.52, 0.44, 0.46, 0.56), np.float32)
    attenuation = 1.0 - radial[..., None] * corner_losses.reshape(1, 1, 4)
    noiseless_observed = clean * attenuation
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.0012, clean.shape).astype(np.float32)
    observed = np.clip(noiseless_observed + noise, 0.0, 1.0)

    coarse_radial = _radial_grid(grid_height, grid_width)
    coarse_attenuation = 1.0 - coarse_radial[..., None] * corner_losses.reshape(1, 1, 4)
    coarse_gains = 1.0 / coarse_attenuation
    return {
        "clean": clean,
        "attenuation": attenuation,
        "noiseless_observed": noiseless_observed,
        "observed": observed,
        "coarse_gains": coarse_gains.astype(np.float32),
    }


def _region_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    radial = _radial_grid(height, width)
    return radial <= 0.08, radial >= 0.72


def _flatness_metrics(packed: np.ndarray) -> dict[str, object]:
    center, corners = _region_masks(*packed.shape[:2])
    channel_metrics = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        plane = packed[..., channel]
        center_mean = float(np.mean(plane[center]))
        corner_mean = float(np.mean(plane[corners]))
        channel_metrics[name] = {
            "center_mean": center_mean,
            "corner_mean": corner_mean,
            "corner_to_center_ratio": corner_mean / max(center_mean, 1e-12),
            "spatial_cv": float(np.std(plane) / max(np.mean(plane), 1e-12)),
        }
    return channel_metrics


def _noise_metrics(noisy: np.ndarray, noiseless: np.ndarray) -> dict[str, object]:
    center, corners = _region_masks(*noisy.shape[:2])
    residual = noisy - noiseless
    channel_metrics = {}
    for channel, name in enumerate(CHANNEL_NAMES):
        plane = residual[..., channel]
        center_std = float(np.std(plane[center]))
        corner_std = float(np.std(plane[corners]))
        channel_metrics[name] = {
            "center_noise_std": center_std,
            "corner_noise_std": corner_std,
            "corner_noise_amplification": corner_std / max(center_std, 1e-12),
        }
    return channel_metrics


def _packed_preview(packed: np.ndarray, display_gain: float = 2.5) -> np.ndarray:
    camera_rgb = np.stack(
        [packed[..., 0], 0.5 * (packed[..., 1] + packed[..., 2]), packed[..., 3]],
        axis=-1,
    )
    mapped = reinhard_tone_map(np.maximum(camera_rgb * display_gain, 0.0), white_point=4.0)
    return np.clip(linear_to_srgb(mapped) * 255.0, 0.0, 255.0).astype(np.uint8)


def _gain_preview(gain_map: np.ndarray) -> np.ndarray:
    mean_gain = np.mean(gain_map, axis=-1)
    normalized = (mean_gain - mean_gain.min()) / max(float(np.ptp(mean_gain)), 1e-12)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _panel(rgb: np.ndarray, title: str, width: int = 560) -> np.ndarray:
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 44), fill=(0, 0, 0))
    draw.text((10, 7), title, font=_font(19), fill=(255, 255, 255))
    return np.asarray(image)


def _write_preview(
    observed: np.ndarray,
    corrected: np.ndarray,
    gain_map: np.ndarray,
    real_raw: np.ndarray,
    path: Path,
) -> None:
    synthetic_row = np.hstack(
        [
            _panel(_packed_preview(observed), "合成平场：带径向暗角"),
            _panel(_packed_preview(corrected), "合成平场：LSC校正后"),
            _panel(_gain_preview(gain_map), "LSC增益图：角落增益更高"),
        ]
    )
    real_preview = _packed_preview(real_raw, display_gain=250.0)
    real_row = np.hstack(
        [
            _panel(real_preview, "真实SID：identity LSC前"),
            _panel(real_preview.copy(), "真实SID：identity LSC后"),
            _panel(np.zeros_like(real_preview), "无真实平场标定：不执行实际LSC"),
        ]
    )
    canvas = np.vstack((synthetic_row, real_row))
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入LSC预览图: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 LSC接口、均匀性、噪声代价与identity检查")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--grid-height", type=int, default=13)
    parser.add_argument("--grid-width", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase1_raw_frontend/lsc",
    )
    args = parser.parse_args()

    synthetic = _synthetic_flat_field(
        args.height,
        args.width,
        args.grid_height,
        args.grid_width,
        args.seed,
    )
    config = LscConfig(minimum_gain=0.5, maximum_gain=4.0)
    noisy_result = apply_lens_shading_correction_packed(
        synthetic["observed"], synthetic["coarse_gains"], config
    )
    noiseless_result = apply_lens_shading_correction_packed(
        synthetic["noiseless_observed"], synthetic["coarse_gains"], config
    )

    before_flatness = _flatness_metrics(synthetic["observed"])
    after_flatness = _flatness_metrics(noisy_result.corrected)
    before_noise = _noise_metrics(synthetic["observed"], synthetic["noiseless_observed"])
    after_noise = _noise_metrics(noisy_result.corrected, noiseless_result.corrected)
    correction_mae = float(np.mean(np.abs(noiseless_result.corrected - synthetic["clean"])))

    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]
    frame = load_normalized_raw(pair.short_path)
    identity_result = apply_lens_shading_correction_packed(
        frame.packed,
        identity_lsc_gain_map(),
        config,
    )
    identity_max_abs_error = float(np.max(np.abs(identity_result.corrected - frame.packed)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / "lsc_synthetic_and_identity_check.png"
    report_path = args.output_dir / "lsc_synthetic_and_identity_report.json"
    _write_preview(
        synthetic["observed"],
        noisy_result.corrected,
        noisy_result.full_resolution_gain_map,
        frame.packed,
        preview_path,
    )

    before_ratios = [before_flatness[name]["corner_to_center_ratio"] for name in CHANNEL_NAMES]
    after_ratios = [after_flatness[name]["corner_to_center_ratio"] for name in CHANNEL_NAMES]
    before_cvs = [before_flatness[name]["spatial_cv"] for name in CHANNEL_NAMES]
    after_cvs = [after_flatness[name]["spatial_cv"] for name in CHANNEL_NAMES]
    report = {
        "experiment_version": "Phase1-LSC-Check-v0.1.0",
        "synthetic": {
            "packed_shape": list(synthetic["observed"].shape),
            "calibration_grid_shape": list(synthetic["coarse_gains"].shape),
            "gain_min": float(noisy_result.full_resolution_gain_map.min()),
            "gain_max": float(noisy_result.full_resolution_gain_map.max()),
            "before_flatness": before_flatness,
            "after_flatness": after_flatness,
            "before_noise": before_noise,
            "after_noise": after_noise,
            "mean_corner_to_center_before": float(np.mean(before_ratios)),
            "mean_corner_to_center_after": float(np.mean(after_ratios)),
            "mean_spatial_cv_before": float(np.mean(before_cvs)),
            "mean_spatial_cv_after": float(np.mean(after_cvs)),
            "noiseless_correction_mae": correction_mae,
        },
        "real_sid_identity": {
            "pair_index": args.pair_index,
            "input": str(pair.short_path),
            "packed_shape": list(frame.packed.shape),
            "maximum_absolute_error": identity_max_abs_error,
        },
        "preview": str(preview_path.resolve()),
        "limitation": "合成gain map只验收接口；没有目标镜头平场RAW，不能宣称完成真实LSC标定。",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("Phase 1 LSC合成平场验收")
    print(
        f"角落/中心均值比: {np.mean(before_ratios):.5f} -> {np.mean(after_ratios):.5f} "
        "（理想值接近1）"
    )
    print(
        f"空间CV: {np.mean(before_cvs):.5f} -> {np.mean(after_cvs):.5f} "
        "（越低越均匀）"
    )
    print(f"无噪声校正MAE: {correction_mae:.8f}")
    print(f"增益范围: [{report['synthetic']['gain_min']:.4f}, {report['synthetic']['gain_max']:.4f}]")
    print("角落噪声放大（校正后）")
    for name in CHANNEL_NAMES:
        print(f"{name}: {after_noise[name]['corner_noise_amplification']:.4f}x")
    print(f"真实SID identity最大绝对误差: {identity_max_abs_error:.8g}")
    print(f"中文对比图: {preview_path.resolve()}")
    print(f"指标报告: {report_path.resolve()}")

    mean_after_ratio = float(np.mean(after_ratios))
    mean_after_cv = float(np.mean(after_cvs))
    if not 0.98 <= mean_after_ratio <= 1.02:
        raise SystemExit("LSC合成验收未通过：校正后角落/中心比偏离1超过2%")
    if mean_after_cv >= float(np.mean(before_cvs)) * 0.15:
        raise SystemExit("LSC合成验收未通过：空间CV下降不足")
    if identity_max_abs_error != 0.0:
        raise SystemExit("LSC identity验收未通过：真实RAW发生数值变化")
    print("Phase 1 LSC接口验收: PASS（真实镜头标定仍未完成）")


if __name__ == "__main__":
    main()
