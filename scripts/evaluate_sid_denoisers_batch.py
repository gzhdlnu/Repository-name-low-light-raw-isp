#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Held-Out SID Denoiser Benchmark
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-BatchEval-v0.1.0
# =============================================================================
"""Evaluate traditional RAW denoisers on multiple held-out SID scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import rawpy

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
from metrics import compare_display_rgb  # noqa: E402


METHOD_NAMES = ("no_denoise", "fixed_gaussian", "noise_aware_wiener")
METHOD_LABELS_ZH = {
    "no_denoise": "原始短曝光",
    "fixed_gaussian": "固定 Gaussian",
    "noise_aware_wiener": "噪声感知 Wiener",
}
METRICS = {
    "psnr_db": ("PSNR（dB，越高越好）", True),
    "ssim": ("SSIM（越高越好）", True),
    "delta_e2000_mean": ("DeltaE2000 均值（越低越好）", False),
    "gradient_similarity": ("梯度相似度（越高越好）", True),
}


@dataclass(frozen=True)
class RawFrame:
    packed: np.ndarray
    pattern: str
    white_balance_gains: tuple[float, float, float, float]
    color_matrix: np.ndarray


def _parse_pair_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair indices must be comma-separated integers") from exc
    if len(indices) < 3:
        raise argparse.ArgumentTypeError("at least three held-out pairs are required")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("pair indices must not contain duplicates")
    return indices


def _scene_id(path: Path) -> str:
    return path.stem.split("_")[0]


def _load_frame(path: Path) -> RawFrame:
    with rawpy.imread(str(path)) as image:
        raw = image.raw_image_visible.astype(np.float32)
        black = float(min(image.black_level_per_channel))
        white = float(image.white_level)
        pattern = "".join("RGBG"[int(value)] for value in image.raw_pattern.flat)
        wb_rgbg = np.asarray(image.camera_whitebalance[:4], dtype=np.float32)
        color_matrix = camera_rgb_to_srgb_matrix(image.rgb_xyz_matrix)
    gains = np.asarray([wb_rgbg[0], wb_rgbg[1], wb_rgbg[3], wb_rgbg[2]], dtype=np.float32)
    gains /= max(float(gains[1]), float(gains[2]), 1e-6)
    return RawFrame(
        packed=pack_bayer(normalize_raw(raw, black, white), pattern),
        pattern=pattern,
        white_balance_gains=tuple(float(value) for value in gains),
        color_matrix=color_matrix,
    )


def _render(
    packed: np.ndarray,
    frame: RawFrame,
    tone_map: str,
    white_point: float,
    use_color_matrix: bool,
) -> np.ndarray:
    rgb, _ = render_packed_raw(
        packed,
        frame.pattern,
        frame.white_balance_gains,
        tone_map=tone_map,
        white_point=white_point,
        color_matrix=frame.color_matrix if use_color_matrix else None,
    )
    return rgb


def _aggregate(per_pair: list[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {}
    for method in METHOD_NAMES:
        aggregate[method] = {}
        for metric in METRICS:
            values = np.asarray(
                [item["methods"][method][metric] for item in per_pair], dtype=np.float64
            )
            aggregate[method][metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
            }

    win_counts: dict[str, int] = {}
    for metric, (_, higher_is_better) in METRICS.items():
        wins = 0
        for item in per_pair:
            adaptive = item["methods"]["noise_aware_wiener"][metric]
            gaussian = item["methods"]["fixed_gaussian"][metric]
            wins += int(adaptive > gaussian if higher_is_better else adaptive < gaussian)
        win_counts[metric] = wins
    return {
        "method_statistics": aggregate,
        "wiener_vs_gaussian_win_counts": win_counts,
        "total_pairs": len(per_pair),
    }


def _plot_metrics(per_pair: list[dict[str, object]], output_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(len(per_pair))
    labels = [f"pair {item['pair_index']}" for item in per_pair]
    styles = {
        "no_denoise": ("#6b7280", "o"),
        "fixed_gaussian": ("#2563eb", "s"),
        "noise_aware_wiener": ("#dc2626", "^"),
    }
    for axis, (metric, (title, _)) in zip(axes.flat, METRICS.items()):
        for method in METHOD_NAMES:
            color, marker = styles[method]
            values = [item["methods"][method][metric] for item in per_pair]
            axis.plot(
                x,
                values,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=6,
                label=METHOD_LABELS_ZH[method],
            )
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    fig.suptitle("SID ISO 200 独立场景传统 RAW 降噪验证", fontsize=16)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量评估独立 SID 场景上的传统 RAW 降噪")
    parser.add_argument("--sid-list", type=Path, required=True)
    parser.add_argument("--noise-report", type=Path, required=True)
    parser.add_argument("--pair-indices", type=_parse_pair_indices, required=True)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--adaptive-window", type=int, default=7)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument("--use-color-matrix", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase3_noise" / "validation",
    )
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    if any(index < 0 or index >= len(pairs) for index in args.pair_indices):
        raise IndexError(f"pair index outside [0, {len(pairs)})")
    selected = [(index, pairs[index]) for index in args.pair_indices]
    scenes = {_scene_id(pair.short_path) for _, pair in selected}
    if len(scenes) != len(selected):
        raise ValueError("evaluation pairs must come from distinct SID scenes")

    noise_report = json.loads(args.noise_report.read_text(encoding="utf-8"))
    calibration_indices = {int(index) for index in noise_report["calibration_pair_indices"]}
    overlap = calibration_indices.intersection(args.pair_indices)
    if overlap:
        raise ValueError(f"evaluation pairs overlap calibration set: {sorted(overlap)}")
    model_iso = int(noise_report["iso"])
    if any(pair.iso != model_iso for _, pair in selected):
        raise ValueError("all evaluation pairs must match the calibrated noise-model ISO")
    calibration_gains = {
        round(float(profile["exposure_gain"]), 6) for profile in noise_report["pair_profiles"]
    }
    evaluation_gains = {round(float(pair.exposure_gain), 6) for _, pair in selected}
    if len(calibration_gains) != 1 or evaluation_gains != calibration_gains:
        raise ValueError(
            "evaluation exposure gain must exactly match the calibration gain; "
            f"calibration={sorted(calibration_gains)}, evaluation={sorted(evaluation_gains)}"
        )
    slopes, intercepts = parse_noise_models(noise_report["channels"])

    per_pair: list[dict[str, object]] = []
    for index, pair in selected:
        short = _load_frame(pair.short_path)
        reference = _load_frame(pair.long_path)
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
        reference_rgb = _render(
            reference.packed,
            reference,
            args.tone_map,
            args.white_point,
            args.use_color_matrix,
        )
        method_metrics = {
            name: compare_display_rgb(
                _render(
                    packed,
                    short,
                    args.tone_map,
                    args.white_point,
                    args.use_color_matrix,
                ),
                reference_rgb,
            )
            for name, packed in candidates.items()
        }
        per_pair.append(
            {
                "pair_index": index,
                "scene_id": _scene_id(pair.short_path),
                "iso": pair.iso,
                "exposure_gain": pair.exposure_gain,
                "methods": method_metrics,
            }
        )
        adaptive = method_metrics["noise_aware_wiener"]
        print(
            f"完成 pair {index}（场景 {_scene_id(pair.short_path)}）: "
            f"Wiener PSNR={adaptive['psnr_db']:.3f} dB, SSIM={adaptive['ssim']:.5f}"
        )

    aggregate = _aggregate(per_pair)
    report = {
        "experiment_version": "Phase3-BatchEval-v0.1.0",
        "description_zh": "同 ISO、同曝光增益、与标定集场景隔离的传统 RAW 降噪验证",
        "noise_report": str(args.noise_report.resolve()),
        "pair_indices": args.pair_indices,
        "scene_ids": sorted(scenes),
        "iso": model_iso,
        "configuration": {
            "gaussian_sigma": args.gaussian_sigma,
            "adaptive_window": args.adaptive_window,
            "noise_strength": args.noise_strength,
            "tone_map": args.tone_map,
            "white_point": args.white_point,
            "color_matrix_applied": args.use_color_matrix,
        },
        "per_pair": per_pair,
        "aggregate": aggregate,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"iso_{model_iso:05d}_heldout_denoiser_benchmark"
    report_path = args.output_dir / f"{stem}.json"
    plot_path = args.output_dir / f"{stem}.png"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_metrics(per_pair, plot_path)

    print("\n独立场景均值")
    print("方法                 PSNR(dB)    SSIM    DeltaE均值    梯度相似度")
    statistics = aggregate["method_statistics"]
    for method in METHOD_NAMES:
        values = statistics[method]
        print(
            f"{METHOD_LABELS_ZH[method]:<19} {values['psnr_db']['mean']:>8.3f}"
            f"  {values['ssim']['mean']:>7.5f}"
            f"  {values['delta_e2000_mean']['mean']:>11.3f}"
            f"  {values['gradient_similarity']['mean']:>12.5f}"
        )
    wins = aggregate["wiener_vs_gaussian_win_counts"]
    total = aggregate["total_pairs"]
    print(
        "Wiener 相对 Gaussian 胜场: "
        f"PSNR {wins['psnr_db']}/{total}, SSIM {wins['ssim']}/{total}, "
        f"DeltaE {wins['delta_e2000_mean']}/{total}, "
        f"梯度相似度 {wins['gradient_similarity']}/{total}"
    )
    print(f"中文指标图: {plot_path}")
    print(f"汇总报告: {report_path}")


if __name__ == "__main__":
    main()
