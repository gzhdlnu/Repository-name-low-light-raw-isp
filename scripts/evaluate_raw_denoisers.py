#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Traditional RAW Denoiser Evaluation
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-DenoiseEval-v0.2.0
# =============================================================================
"""Compare fixed and noise-aware RAW denoising on one official SID pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import (  # noqa: E402
    camera_rgb_to_srgb_matrix,
    gaussian_denoise_packed,
    noise_aware_wiener_packed,
    normalize_raw,
    pack_bayer,
    parse_noise_models,
    render_packed_raw,
)
from metrics import compare_display_rgb, display_luma_summary  # noqa: E402


@dataclass(frozen=True)
class RawFrame:
    packed: np.ndarray
    pattern: str
    white_balance_gains: tuple[float, float, float, float]
    color_matrix: np.ndarray


METHOD_LABELS_ZH = {
    "no_denoise": "原始短曝光（曝光补偿，无降噪）",
    "fixed_gaussian": "固定 Gaussian 降噪",
    "noise_aware_wiener": "噪声感知 Wiener 降噪",
    "reference": "真实长曝光参考（Ground Truth）",
}


def _load_chinese_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    raise RuntimeError("No Chinese font found; install fonts-noto-cjk")


def _load_frame(path: Path) -> RawFrame:
    with rawpy.imread(str(path)) as image:
        raw = image.raw_image_visible.astype(np.float32)
        black = float(min(image.black_level_per_channel))
        white = float(image.white_level)
        pattern = "".join("RGBG"[int(value)] for value in image.raw_pattern.flat)
        wb_rgbg = np.asarray(image.camera_whitebalance[:4], dtype=np.float32)
        color_matrix = camera_rgb_to_srgb_matrix(image.rgb_xyz_matrix)

    white_balance_gains = np.array(
        [wb_rgbg[0], wb_rgbg[1], wb_rgbg[3], wb_rgbg[2]], dtype=np.float32
    )
    white_balance_gains /= max(
        float(white_balance_gains[1]), float(white_balance_gains[2]), 1e-6
    )
    normalized = normalize_raw(raw, black, white)
    return RawFrame(
        packed=pack_bayer(normalized, pattern),
        pattern=pattern,
        white_balance_gains=tuple(float(value) for value in white_balance_gains),
        color_matrix=color_matrix,
    )


def _render(
    packed: np.ndarray,
    frame: RawFrame,
    tone_map: str,
    white_point: float,
    use_color_matrix: bool,
) -> tuple[np.ndarray, float]:
    return render_packed_raw(
        packed,
        frame.pattern,
        frame.white_balance_gains,
        tone_map=tone_map,
        white_point=white_point,
        color_matrix=frame.color_matrix if use_color_matrix else None,
    )


def _draw_title(rgb: np.ndarray, label: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 58), fill=(0, 0, 0))
    draw.text((18, 10), label, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def _comparison_grid(
    images: list[tuple[str, np.ndarray]], unamplified_short: np.ndarray
) -> np.ndarray:
    font = _load_chinese_font(28)
    panels = []
    for index, (label, rgb) in enumerate(images):
        panel = cv2.resize(
            rgb,
            dsize=None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA,
        )
        panel = _draw_title(panel, label, font).copy()
        if index == 0:
            inset_width = panel.shape[1] // 3
            inset_height = int(unamplified_short.shape[0] * inset_width / unamplified_short.shape[1])
            inset = cv2.resize(
                unamplified_short,
                (inset_width, inset_height),
                interpolation=cv2.INTER_AREA,
            )
            inset = _draw_title(inset, "未提亮的原始短曝光", _load_chinese_font(20))
            top = 76
            left = panel.shape[1] - inset_width - 18
            cv2.rectangle(
                panel,
                (left - 3, top - 3),
                (left + inset_width + 3, top + inset_height + 3),
                (255, 255, 255),
                thickness=3,
            )
            panel[top : top + inset_height, left : left + inset_width] = inset
        panels.append(panel)
    return cv2.vconcat([cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])])


def _select_analysis_rois(reference_rgb: np.ndarray, crop_size: int = 512) -> dict[str, tuple[int, int, int, int]]:
    """Select one dark/flat crop and one detailed crop from the reference."""
    height, width = reference_rgb.shape[:2]
    size = min(crop_size, height, width)
    luma = (
        0.2126 * reference_rgb[..., 0]
        + 0.7152 * reference_rgb[..., 1]
        + 0.0722 * reference_rgb[..., 2]
    ).astype(np.float32) / 255.0
    grad_x = cv2.Scharr(luma, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(luma, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(grad_x, grad_y)

    step = max(size // 4, 64)
    candidates: list[tuple[float, float, int, int]] = []
    y_positions = list(range(0, max(height - size + 1, 1), step))
    x_positions = list(range(0, max(width - size + 1, 1), step))
    if not y_positions or y_positions[-1] != height - size:
        y_positions.append(height - size)
    if not x_positions or x_positions[-1] != width - size:
        x_positions.append(width - size)
    for y in y_positions:
        for x in x_positions:
            patch_luma = luma[y : y + size, x : x + size]
            patch_gradient = gradient[y : y + size, x : x + size]
            candidates.append((float(np.mean(patch_luma)), float(np.mean(patch_gradient)), x, y))

    visible = [item for item in candidates if 0.03 <= item[0] <= 0.35]
    if not visible:
        visible = candidates
    dark = min(visible, key=lambda item: item[0] + 0.35 * item[1])
    detail_candidates = [item for item in candidates if 0.08 <= item[0] <= 0.85]
    if not detail_candidates:
        detail_candidates = candidates
    detail = max(detail_candidates, key=lambda item: item[1])
    return {
        "dark_noise": (dark[2], dark[3], size, size),
        "fine_detail": (detail[2], detail[3], size, size),
    }


def _crop_comparison(
    images: list[tuple[str, np.ndarray]],
    rois: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    font = _load_chinese_font(25)
    short_labels = ("原始（无降噪）", "固定 Gaussian", "噪声感知 Wiener", "长曝光参考")
    rows = []
    for roi_name, (x, y, width, height) in rois.items():
        purpose = "暗部噪声区域" if roi_name == "dark_noise" else "高频细节区域"
        panels = []
        for short_label, (_, rgb) in zip(short_labels, images):
            crop = rgb[y : y + height, x : x + width]
            panels.append(_draw_title(crop, f"{purpose}｜{short_label}", font))
        rows.append(cv2.hconcat(panels))
    return cv2.vconcat(rows)


def _validate_noise_report(
    report: dict[str, object], pair_index: int, short_path: Path, iso: int
) -> None:
    calibration_indices = report.get("calibration_pair_indices")
    if isinstance(calibration_indices, list):
        if pair_index in {int(index) for index in calibration_indices}:
            raise ValueError("evaluation pair belongs to the noise-model calibration set")
        if int(report.get("iso", -1)) != iso:
            raise ValueError("calibrated noise-model ISO does not match evaluation pair ISO")
        return
    if int(report.get("pair_index", -1)) != pair_index:
        raise ValueError("noise report pair_index does not match requested SID pair")
    report_short = Path(str(report.get("short_path", ""))).resolve()
    if report_short != short_path.resolve():
        raise ValueError("noise report short_path does not match requested SID pair")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traditional RAW denoisers on one SID pair")
    parser.add_argument("--sid-list", type=Path, required=True)
    parser.add_argument("--noise-report", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--adaptive-window", type=int, default=7)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument("--use-color-matrix", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "phase3_noise")
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]
    noise_report = json.loads(args.noise_report.read_text(encoding="utf-8"))
    _validate_noise_report(noise_report, args.pair_index, pair.short_path, pair.iso)
    slopes, intercepts = parse_noise_models(noise_report["channels"])

    short = _load_frame(pair.short_path)
    reference_frame = _load_frame(pair.long_path)
    exposure_corrected = short.packed * np.float32(pair.exposure_gain)
    candidates = {
        "no_denoise": exposure_corrected,
        "fixed_gaussian": gaussian_denoise_packed(exposure_corrected, args.gaussian_sigma),
        "noise_aware_wiener": noise_aware_wiener_packed(
            exposure_corrected,
            slopes,
            intercepts,
            window_size=args.adaptive_window,
            strength=args.noise_strength,
        ),
    }

    reference_rgb, reference_over_white = _render(
        reference_frame.packed,
        reference_frame,
        args.tone_map,
        args.white_point,
        args.use_color_matrix,
    )
    unamplified_short_rgb, unamplified_over_white = _render(
        short.packed,
        short,
        args.tone_map,
        args.white_point,
        args.use_color_matrix,
    )
    rendered: dict[str, np.ndarray] = {}
    methods: dict[str, object] = {}
    for name, packed in candidates.items():
        rgb, over_white = _render(
            packed, short, args.tone_map, args.white_point, args.use_color_matrix
        )
        rendered[name] = rgb
        methods[name] = {
            "display_name_zh": METHOD_LABELS_ZH[name],
            "metrics": compare_display_rgb(rgb, reference_rgb),
            "display_summary": display_luma_summary(rgb),
            "pre_tone_over_white_ratio": over_white,
        }

    report = {
        "experiment_version": "Phase3-DenoiseEval-v0.2.0",
        "pair_index": args.pair_index,
        "short_path": str(pair.short_path),
        "long_path": str(pair.long_path),
        "noise_report": str(args.noise_report.resolve()),
        "exposure_gain": pair.exposure_gain,
        "iso": pair.iso,
        "configuration": {
            "gaussian_sigma": args.gaussian_sigma,
            "adaptive_window": args.adaptive_window,
            "noise_strength": args.noise_strength,
            "tone_map": args.tone_map,
            "white_point": args.white_point,
            "color_matrix_applied": args.use_color_matrix,
        },
        "noise_model": {
            name: {"slope_a": float(slopes[index]), "intercept_b": float(intercepts[index])}
            for index, name in enumerate(("R", "G1", "G2", "B"))
        },
        "reference": {
            "display_summary": display_luma_summary(reference_rgb),
            "pre_tone_over_white_ratio": reference_over_white,
        },
        "unamplified_short": {
            "description_zh": "未乘曝光比的原始短曝光 RAW 经相同 ISP 渲染，仅用于展示采集亮度",
            "display_summary": display_luma_summary(unamplified_short_rgb),
            "pre_tone_over_white_ratio": unamplified_over_white,
        },
        "methods": methods,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{args.pair_index:04d}_raw_denoisers"
    comparison_path = args.output_dir / f"{stem}_comparison.png"
    crops_path = args.output_dir / f"{stem}_crops.png"
    report_path = args.output_dir / f"{stem}_report.json"
    labeled_images = [
        (METHOD_LABELS_ZH["no_denoise"], rendered["no_denoise"]),
        (METHOD_LABELS_ZH["fixed_gaussian"], rendered["fixed_gaussian"]),
        (METHOD_LABELS_ZH["noise_aware_wiener"], rendered["noise_aware_wiener"]),
        (METHOD_LABELS_ZH["reference"], reference_rgb),
    ]
    rois = _select_analysis_rois(reference_rgb)
    report["analysis_rois"] = {
        name: {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
        for name, roi in rois.items()
    }
    grid = _comparison_grid(labeled_images, unamplified_short_rgb)
    crops = _crop_comparison(labeled_images, rois)
    if not cv2.imwrite(str(comparison_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {comparison_path}")
    if not cv2.imwrite(str(crops_path), cv2.cvtColor(crops, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {crops_path}")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("方法                     PSNR(dB)    SSIM    DeltaE均值    梯度相似度")
    for name, result in methods.items():
        metrics = result["metrics"]
        print(
            f"{METHOD_LABELS_ZH[name]:<22} {metrics['psnr_db']:>8.3f}  {metrics['ssim']:>7.5f}"
            f"  {metrics['delta_e2000_mean']:>11.3f}  {metrics['gradient_similarity']:>19.5f}"
        )
    print(f"全图对比: {comparison_path}")
    print(f"局部放大: {crops_path}")
    print(f"指标报告: {report_path}")


if __name__ == "__main__":
    main()
