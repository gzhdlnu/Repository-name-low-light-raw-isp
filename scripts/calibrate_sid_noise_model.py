#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Multi-Scene SID Noise Calibration
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-NoiseCal-v0.1.0
# =============================================================================
"""Fit one robust RAW noise model from multiple same-ISO SID scenes."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import rawpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import estimate_residual_noise_profile, fit_nonnegative_variance_model  # noqa: E402
from data import load_sid_list  # noqa: E402
from isp import normalize_raw, pack_bayer  # noqa: E402


CHANNEL_NAMES = ("R", "G1", "G2", "B")


def _parse_pair_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair indices must be comma-separated integers") from exc
    if len(indices) < 3:
        raise argparse.ArgumentTypeError("at least three calibration pairs are required")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("pair indices must not contain duplicates")
    return indices


def _scene_id(path: Path) -> str:
    return path.stem.split("_")[0]


def _load_packed(path: Path) -> np.ndarray:
    with rawpy.imread(str(path)) as image:
        raw = image.raw_image_visible.astype(np.float32)
        black = float(min(image.black_level_per_channel))
        white = float(image.white_level)
        pattern = "".join("RGBG"[int(value)] for value in image.raw_pattern.flat)
    return pack_bayer(normalize_raw(raw, black, white, clip=False), pattern)


def _median_and_mad(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(finite))
    return median, float(np.median(np.abs(finite - median)))


def main() -> None:
    parser = argparse.ArgumentParser(description="从多个同 ISO SID 场景标定 RAW 噪声模型")
    parser.add_argument("--sid-list", type=Path, required=True)
    parser.add_argument("--pair-indices", type=_parse_pair_indices, required=True)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--max-signal", type=float, default=0.8)
    parser.add_argument("--flat-fraction", type=float, default=0.4)
    parser.add_argument("--minimum-samples", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase3_noise" / "calibration",
    )
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    if any(index < 0 or index >= len(pairs) for index in args.pair_indices):
        raise IndexError(f"pair index outside [0, {len(pairs)})")
    selected = [(index, pairs[index]) for index in args.pair_indices]
    iso_values = {pair.iso for _, pair in selected}
    scene_ids = {_scene_id(pair.short_path) for _, pair in selected}
    if len(iso_values) != 1:
        raise ValueError(f"all calibration pairs must share one ISO, got {sorted(iso_values)}")
    if len(scene_ids) != len(selected):
        raise ValueError("calibration pairs must come from distinct SID scenes")

    edges = np.linspace(0.005, args.max_signal, args.bins + 1, dtype=np.float32)
    pair_profiles: list[dict[str, object]] = []
    for index, pair in selected:
        short = _load_packed(pair.short_path)
        long = _load_packed(pair.long_path)
        profile = estimate_residual_noise_profile(
            short,
            long,
            pair.exposure_gain,
            edges,
            flat_fraction=args.flat_fraction,
            minimum_samples=args.minimum_samples,
        )
        pair_profiles.append(
            {
                "pair_index": index,
                "scene_id": _scene_id(pair.short_path),
                "short_path": str(pair.short_path),
                "long_path": str(pair.long_path),
                "exposure_gain": pair.exposure_gain,
                "channels": profile,
            }
        )

    calibrated_channels: dict[str, object] = {}
    for channel_name in CHANNEL_NAMES:
        bins = [
            item
            for profile in pair_profiles
            for item in profile["channels"][channel_name]["bins"]
        ]
        signal = np.asarray([item["signal"] for item in bins], dtype=np.float64)
        variance = np.asarray([item["variance"] for item in bins], dtype=np.float64)
        fitted = fit_nonnegative_variance_model(signal, variance)
        pair_slopes = np.asarray(
            [
                profile["channels"][channel_name]["variance_model"]["slope_a"]
                for profile in pair_profiles
            ],
            dtype=np.float64,
        )
        pair_intercepts = np.asarray(
            [
                profile["channels"][channel_name]["variance_model"]["intercept_b"]
                for profile in pair_profiles
            ],
            dtype=np.float64,
        )
        slope_median, slope_mad = _median_and_mad(pair_slopes)
        intercept_median, intercept_mad = _median_and_mad(pair_intercepts)
        fitted["cross_scene_spread"] = {
            "slope_median": slope_median,
            "slope_mad": slope_mad,
            "intercept_median": intercept_median,
            "intercept_mad": intercept_mad,
        }
        fitted["observations"] = bins
        calibrated_channels[channel_name] = fitted

    iso = next(iter(iso_values))
    report = {
        "experiment_version": "Phase3-NoiseCal-v0.1.0",
        "measurement_scope": (
            "多场景、同 ISO 的增益归一化短长曝光残差模型；不是实验室 photon-transfer 标定"
        ),
        "intended_use": "供未参与标定的相同 ISO SID 样本进行传统 RAW 去噪验证",
        "iso": iso,
        "calibration_pair_indices": args.pair_indices,
        "calibration_scene_ids": sorted(scene_ids),
        "signal_range": [float(edges[0]), float(edges[-1])],
        "flat_fraction": args.flat_fraction,
        "minimum_samples": args.minimum_samples,
        "channels": calibrated_channels,
        "pair_profiles": pair_profiles,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iso_{iso:05d}_multiscene_noise_model"
    report_path = args.output_dir / f"{stem}.json"
    plot_path = args.output_dir / f"{stem}.png"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(pair_profiles)))
    for axis, channel_name in zip(axes.flat, CHANNEL_NAMES):
        for color, profile in zip(colors, pair_profiles):
            bins = profile["channels"][channel_name]["bins"]
            axis.scatter(
                [item["signal"] for item in bins],
                [item["variance"] for item in bins],
                s=16,
                alpha=0.55,
                color=color,
                label=f"场景 {profile['scene_id']}",
            )
        model = calibrated_channels[channel_name]["variance_model"]
        grid = np.linspace(float(edges[0]), float(edges[-1]), 200)
        axis.plot(
            grid,
            model["slope_a"] * grid + model["intercept_b"],
            color="black",
            linewidth=2.2,
            label="多场景鲁棒拟合",
        )
        quality = calibrated_channels[channel_name]["fit_quality"]
        axis.set_title(f"{channel_name} 通道｜R²={quality['r_squared']:.3f}")
        axis.set_xlabel("长曝光参考的线性 RAW 信号")
        axis.set_ylabel("增益归一化残差方差")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"SID Sony ISO {iso} 多场景 RAW 噪声标定", fontsize=15)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"标定 ISO: {iso}，独立场景数: {len(pair_profiles)}")
    print("通道       a（信号相关项）       b（噪声底）       R²       RMSE")
    for name in CHANNEL_NAMES:
        channel = calibrated_channels[name]
        model = channel["variance_model"]
        quality = channel["fit_quality"]
        print(
            f"{name:<4} {model['slope_a']:>17.8g} {model['intercept_b']:>16.8g}"
            f" {quality['r_squared']:>8.4f} {quality['rmse']:>10.6g}"
        )
    print(f"标定曲线: {plot_path}")
    print(f"标定报告: {report_path}")


if __name__ == "__main__":
    main()
