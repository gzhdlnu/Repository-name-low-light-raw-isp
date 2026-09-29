#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 4 Learned RAW Denoiser Evaluation
# Timestamp: 2026-09-22 | Author: Guozh | Version: Phase4-Eval-v0.2.0
# =============================================================================
"""Compare a learned RAW denoiser with traditional baselines on one SID pair."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import (  # noqa: E402
    gaussian_denoise_packed,
    NormalizedRawFrame,
    noise_aware_wiener_packed,
    parse_noise_models,
    render_packed_raw,
    load_normalized_raw,
)
from metrics import compare_display_rgb, display_luma_summary  # noqa: E402
from models import RawDenoisingUNet, tiled_predict  # noqa: E402


METHOD_LABELS_ZH = {
    "no_denoise": "原始短曝光（无降噪）",
    "fixed_gaussian": "固定 Gaussian",
    "noise_aware_wiener": "噪声感知 Wiener",
    "learned_unet": "学习型残差 U-Net",
    "reference": "真实长曝光参考",
}


_load_frame = load_normalized_raw


def _render(
    packed: np.ndarray,
    frame: NormalizedRawFrame,
    tone_map: str,
    white_point: float,
) -> np.ndarray:
    rgb, _ = render_packed_raw(
        packed,
        frame.pattern,
        frame.white_balance_gains,
        tone_map=tone_map,
        white_point=white_point,
    )
    return rgb


def _raw_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference * difference))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-12))),
    }


def _color_summary(rgb: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Quantify saturation/chroma so visual grayness is not judged by eye alone."""
    prediction_float = rgb.astype(np.float32) / 255.0
    reference_float = reference.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = rgb2lab(prediction_float[::4, ::4])
    ref_lab = rgb2lab(reference_float[::4, ::4])
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    ref_chroma = np.sqrt(ref_lab[..., 1] ** 2 + ref_lab[..., 2] ** 2)
    channel_mean = np.mean(prediction_float, axis=(0, 1))
    channel_ratio = channel_mean / max(float(channel_mean[1]), 1e-6)
    return {
        "mean_hsv_saturation": float(np.mean(hsv[..., 1]) / 255.0),
        "reference_mean_hsv_saturation": float(np.mean(ref_hsv[..., 1]) / 255.0),
        "mean_lab_chroma_cstar": float(np.mean(chroma)),
        "reference_mean_lab_chroma_cstar": float(np.mean(ref_chroma)),
        "mean_lab_chroma_abs_error": float(np.mean(np.abs(chroma - ref_chroma))),
        "mean_r_over_g": float(channel_ratio[0]),
        "mean_b_over_g": float(channel_ratio[2]),
    }


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError("未找到中文字体，请安装 fonts-noto-cjk")


def _panel(rgb: np.ndarray, title: str, width: int) -> np.ndarray:
    height = int(rgb.shape[0] * width / rgb.shape[1])
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 48), fill=(0, 0, 0))
    font_size = 24
    font = _font(font_size)
    while font_size > 14 and draw.textbbox((0, 0), title, font=font)[2] > width - 20:
        font_size -= 1
        font = _font(font_size)
    draw.text((10, 8), title, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def _summary_panel(methods: dict[str, object], width: int, height: int) -> np.ndarray:
    """Render metrics in the sixth tile instead of a misleading black tile."""
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    title_font = _font(26)
    body_font = _font(18)
    draw.text((22, 18), "指标摘要（相对长曝光参考）", font=title_font, fill=(20, 25, 35))
    names = ("fixed_gaussian", "noise_aware_wiener", "learned_unet")
    labels = ("Gaussian", "Wiener", "学习型 U-Net")
    y = 72
    for header, x in (("方法", 22), ("PSNR", 190), ("SSIM", 300), ("DeltaE", 410), ("梯度", 520)):
        draw.text((x, y), header, font=body_font, fill=(70, 75, 85))
    y += 42
    for name, label in zip(names, labels):
        metrics = methods[name]["display_metrics"]
        values = (label, f"{metrics['psnr_db']:.2f}", f"{metrics['ssim']:.3f}",
                  f"{metrics['delta_e2000_mean']:.2f}", f"{metrics['gradient_similarity']:.3f}")
        for value, x in zip(values, (22, 190, 300, 410, 520)):
            draw.text((x, y), value, font=body_font, fill=(20, 25, 35))
        y += 40
    draw.text((22, y + 20), "↑ PSNR / SSIM / 梯度越高越好", font=body_font, fill=(70, 75, 85))
    draw.text((22, y + 52), "↓ DeltaE 越低越好；仍需结合局部细节观察", font=body_font, fill=(70, 75, 85))
    return np.asarray(image)


def _comparison_grid(images: list[tuple[str, np.ndarray]], methods: dict[str, object]) -> np.ndarray:
    panel_width = 960
    panels = [_panel(rgb, label, panel_width) for label, rgb in images]
    summary = _summary_panel(methods, panel_width, panels[0].shape[0])
    return np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:] + [summary])])


def _select_rois(reference_rgb: np.ndarray, crop_size: int = 512) -> dict[str, tuple[int, int, int, int]]:
    """Select deterministic dark and detailed regions from the reference."""
    height, width = reference_rgb.shape[:2]
    size = min(crop_size, height, width)
    x = reference_rgb.astype(np.float32) / 255.0
    luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    dx = cv2.Scharr(luma, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(luma, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(dx, dy)
    step = max(size // 4, 64)
    candidates: list[tuple[float, float, int, int]] = []
    ys = list(range(0, max(height - size + 1, 1), step))
    xs = list(range(0, max(width - size + 1, 1), step))
    if not ys or ys[-1] != height - size:
        ys.append(height - size)
    if not xs or xs[-1] != width - size:
        xs.append(width - size)
    for top in ys:
        for left in xs:
            region = np.s_[top : top + size, left : left + size]
            candidates.append((float(luma[region].mean()), float(gradient[region].mean()), left, top))
    visible = [item for item in candidates if 0.03 <= item[0] <= 0.35] or candidates
    detailed = [item for item in candidates if 0.08 <= item[0] <= 0.85] or candidates
    dark = min(visible, key=lambda item: item[0] + 0.35 * item[1])
    detail = max(detailed, key=lambda item: item[1])
    return {
        "暗部噪声区域": (dark[2], dark[3], size, size),
        "高频细节区域": (detail[2], detail[3], size, size),
    }


def _crop_grid(
    images: list[tuple[str, np.ndarray]],
    rois: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    labels = ("原始", "Gaussian", "Wiener", "学习型 U-Net", "长曝光参考")
    rows = []
    for roi_name, (left, top, width, height) in rois.items():
        panels = []
        for label, (_, rgb) in zip(labels, images):
            crop = rgb[top : top + height, left : left + width]
            panels.append(_panel(crop, f"{roi_name}｜{label}", 560))
        rows.append(np.hstack(panels))
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Phase 4 学习型 RAW 降噪模型")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt",
    )
    parser.add_argument("--noise-report", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase4_learning/evaluation",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不能执行学习模型推理")
    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]
    noise_report = json.loads(args.noise_report.read_text(encoding="utf-8"))
    if int(noise_report["iso"]) != pair.iso:
        raise ValueError("噪声模型 ISO 与测试 pair 不一致")
    calibration_gains = {
        round(float(profile["exposure_gain"]), 6) for profile in noise_report["pair_profiles"]
    }
    if calibration_gains != {round(pair.exposure_gain, 6)}:
        raise ValueError("噪声模型曝光增益与测试 pair 不一致")
    slopes, intercepts = parse_noise_models(noise_report["channels"])

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint["config"]["base_channels"])
    max_gain = float(checkpoint["config"]["max_exposure_gain"])
    device = torch.device("cuda:0")
    model = RawDenoisingUNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    short = _load_frame(pair.short_path)
    reference_frame = _load_frame(pair.long_path)
    if short.packed.shape != reference_frame.packed.shape:
        raise ValueError("短长曝光 packed RAW shape 不一致")
    applied_gain = min(pair.exposure_gain, max_gain)
    exposure_corrected = np.clip(short.packed * applied_gain, 0.0, 1.0).astype(np.float32)
    input_tensor = torch.from_numpy(exposure_corrected.transpose(2, 0, 1)).unsqueeze(0).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    learned_tensor = tiled_predict(
        model, input_tensor, tile_size=args.tile_size, halo=args.halo, use_amp=True
    )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start
    learned = np.clip(
        learned_tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0
    )
    candidates = {
        "no_denoise": exposure_corrected,
        "fixed_gaussian": gaussian_denoise_packed(exposure_corrected, sigma=1.0),
        "noise_aware_wiener": noise_aware_wiener_packed(
            exposure_corrected, slopes, intercepts, window_size=7, strength=1.0
        ),
        "learned_unet": learned,
    }

    reference_rgb = _render(
        reference_frame.packed, reference_frame, args.tone_map, args.white_point
    )
    rendered: dict[str, np.ndarray] = {}
    methods: dict[str, object] = {}
    for name, packed in candidates.items():
        rgb = _render(packed, short, args.tone_map, args.white_point)
        rendered[name] = rgb
        methods[name] = {
            "display_name_zh": METHOD_LABELS_ZH[name],
            "raw_metrics": _raw_metrics(packed, reference_frame.packed),
            "display_metrics": compare_display_rgb(rgb, reference_rgb),
            "display_summary": display_luma_summary(rgb),
            "color_summary": _color_summary(rgb, reference_rgb),
        }

    report = {
        "experiment_version": "Phase4-Eval-v0.2.0",
        "description_zh": "官方 SID test 场景上的学习型与传统 RAW 降噪统一评估",
        "pair_index": args.pair_index,
        "short_path": str(pair.short_path),
        "long_path": str(pair.long_path),
        "iso": pair.iso,
        "physical_exposure_gain": pair.exposure_gain,
        "applied_exposure_gain": applied_gain,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_channels": base_channels,
        "tile_size": args.tile_size,
        "halo": args.halo,
        "inference_seconds": inference_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "tone_map": args.tone_map,
        "white_point": args.white_point,
        "color_matrix_applied": False,
        "methods": methods,
        "reference_display_summary": display_luma_summary(reference_rgb),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"test_pair_{args.pair_index:04d}_learned_denoisers"
    comparison_path = args.output_dir / f"{stem}_comparison.png"
    crops_path = args.output_dir / f"{stem}_crops.png"
    report_path = args.output_dir / f"{stem}_report.json"
    labeled_images = [
            (METHOD_LABELS_ZH["no_denoise"], rendered["no_denoise"]),
            (METHOD_LABELS_ZH["fixed_gaussian"], rendered["fixed_gaussian"]),
            (METHOD_LABELS_ZH["noise_aware_wiener"], rendered["noise_aware_wiener"]),
            (METHOD_LABELS_ZH["learned_unet"], rendered["learned_unet"]),
            (METHOD_LABELS_ZH["reference"], reference_rgb),
        ]
    rois = _select_rois(reference_rgb)
    report["analysis_rois"] = {
        name: {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
        for name, roi in rois.items()
    }
    grid = _comparison_grid(labeled_images, methods)
    crops = _crop_grid(labeled_images, rois)
    if not cv2.imwrite(str(comparison_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入 {comparison_path}")
    if not cv2.imwrite(str(crops_path), cv2.cvtColor(crops, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入 {crops_path}")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("方法                 RAW PSNR    RAW MAE    显示 PSNR    SSIM    DeltaE    梯度相似度")
    for name, result in methods.items():
        raw = result["raw_metrics"]
        display = result["display_metrics"]
        print(
            f"{METHOD_LABELS_ZH[name]:<18} {raw['psnr_db']:>8.3f}  {raw['mae']:>8.5f}"
            f"  {display['psnr_db']:>9.3f}  {display['ssim']:>7.5f}"
            f"  {display['delta_e2000_mean']:>7.3f}  {display['gradient_similarity']:>10.5f}"
        )
    print("\n颜色指标（用于判断是否偏灰）")
    print("方法                 HSV饱和度    Lab C*    色度误差    R/G      B/G")
    for name, result in methods.items():
        color = result["color_summary"]
        print(
            f"{METHOD_LABELS_ZH[name]:<18} {color['mean_hsv_saturation']:>9.4f}"
            f"  {color['mean_lab_chroma_cstar']:>7.3f}"
            f"  {color['mean_lab_chroma_abs_error']:>8.3f}"
            f"  {color['mean_r_over_g']:>6.3f}  {color['mean_b_over_g']:>6.3f}"
        )
    print(f"分块推理耗时: {inference_seconds:.3f}s，峰值显存: {report['peak_allocated_gib']:.3f} GiB")
    print(f"中文对比图: {comparison_path}")
    print(f"局部裁剪图: {crops_path}")
    print(f"评估报告: {report_path}")


if __name__ == "__main__":
    main()
