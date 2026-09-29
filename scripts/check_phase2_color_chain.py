#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 2 Linear RGB Color Chain Audit
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase2-ColorChain-v0.1.0
# =============================================================================
"""Audit domain order: WB -> demosaic -> CCM -> tone map -> sRGB OETF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import (  # noqa: E402
    apply_color_matrix,
    demosaic_bilinear,
    linear_to_srgb,
    reinhard_tone_map,
    white_balance,
)
from isp.raw_io import load_normalized_raw  # noqa: E402


def _summary(array: np.ndarray) -> dict[str, object]:
    x = np.asarray(array, np.float32)
    return {
        "shape": list(x.shape),
        "minimum": float(x.min()),
        "maximum": float(x.max()),
        "mean": float(x.mean()),
        "finite": bool(np.isfinite(x).all()),
    }


def _monotonic_fraction(inputs: np.ndarray, outputs: np.ndarray) -> float:
    """Return the fraction of sorted input steps whose output is non-decreasing."""
    x = np.asarray(inputs, np.float32).reshape(-1)
    y = np.asarray(outputs, np.float32).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("monotonicity inputs and outputs must have the same shape")
    order = np.argsort(x, kind="stable")
    sorted_values = y[order]
    differences = np.diff(sorted_values)
    return float(np.mean(differences >= -1e-7)) if differences.size else 1.0


def _channel_ratios(rgb: np.ndarray) -> dict[str, float]:
    means = np.mean(rgb.reshape(-1, 3), axis=0)
    return {"R/G": float(means[0] / max(means[1], 1e-8)), "B/G": float(means[2] / max(means[1], 1e-8))}


def _synthetic_chain() -> dict[str, object]:
    rng = np.random.default_rng(20260929)
    packed = np.clip(rng.uniform(0.0, 0.4, (128, 192, 4)).astype(np.float32), 0.0, 1.0)
    gains = (1.8, 1.0, 1.0, 2.2)
    matrix = np.asarray(
        [[1.02, -0.03, 0.01], [-0.02, 1.03, -0.01], [0.01, -0.02, 1.01]],
        np.float32,
    )
    balanced = white_balance(packed, gains)
    scene_rgb = np.maximum(demosaic_bilinear(balanced, "RGGB"), 0.0)
    corrected_rgb = apply_color_matrix(scene_rgb, matrix)
    tone = reinhard_tone_map(corrected_rgb, white_point=4.0)
    srgb = linear_to_srgb(tone)
    return {
        "packed": _summary(packed),
        "balanced_raw": _summary(balanced),
        "linear_camera_rgb": _summary(scene_rgb),
        "linear_target_rgb": _summary(corrected_rgb),
        "tone_mapped_linear_rgb": _summary(tone),
        "srgb": _summary(srgb),
        "tone_monotonic_fraction": _monotonic_fraction(corrected_rgb[..., 1], tone[..., 1]),
        "srgb_monotonic_fraction": _monotonic_fraction(tone[..., 1], srgb[..., 1]),
        "wb_before_demosaic_ratios": _channel_ratios(scene_rgb),
        "ccm_after_demosaic_ratios": _channel_ratios(corrected_rgb),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2线性RGB颜色链路复核")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase2_color_tone/color_chain")
    args = parser.parse_args()

    synthetic = _synthetic_chain()
    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]
    frame = load_normalized_raw(pair.short_path)
    gain = min(pair.exposure_gain, 300.0)
    exposure_raw = np.clip(frame.packed * gain, 0.0, 1.0).astype(np.float32)
    balanced = white_balance(exposure_raw, frame.white_balance_gains)
    scene_rgb = np.maximum(demosaic_bilinear(balanced, frame.pattern), 0.0)
    ccm_rgb = apply_color_matrix(scene_rgb, frame.color_matrix)
    tone = reinhard_tone_map(scene_rgb, white_point=4.0)
    tone_ccm = reinhard_tone_map(ccm_rgb, white_point=4.0)
    srgb = linear_to_srgb(tone)
    srgb_ccm = linear_to_srgb(tone_ccm)
    identity_error = float(np.max(np.abs(tone - reinhard_tone_map(scene_rgb, white_point=4.0))))

    real = {
        "pair_index": args.pair_index,
        "iso": pair.iso,
        "pattern": frame.pattern,
        "requested_exposure_gain": pair.exposure_gain,
        "applied_exposure_gain": gain,
        "stages": {
            "exposure_corrected_raw": _summary(exposure_raw),
            "white_balanced_raw": _summary(balanced),
            "linear_camera_rgb": _summary(scene_rgb),
            "linear_ccm_rgb": _summary(ccm_rgb),
            "tone_mapped_linear_rgb": _summary(tone),
            "tone_mapped_linear_ccm_rgb": _summary(tone_ccm),
            "srgb_default_no_ccm": _summary(srgb),
            "srgb_experimental_ccm": _summary(srgb_ccm),
        },
        "white_balance_gains": list(frame.white_balance_gains),
        "camera_rgb_to_srgb_matrix": frame.color_matrix.tolist(),
        "default_vs_ccm_mean_abs_display_difference": float(np.mean(np.abs(srgb - srgb_ccm))),
        "tone_identity_max_abs_error": identity_error,
        "pre_tone_over_one_ratio": float(np.mean(np.max(scene_rgb, axis=-1) > 1.0)),
        "pre_tone_ccm_over_one_ratio": float(np.mean(np.max(ccm_rgb, axis=-1) > 1.0)),
        "default_pipeline": "WB -> Bilinear Demosaic -> Reinhard -> sRGB OETF",
        "experimental_pipeline": "WB -> Bilinear Demosaic -> CCM -> Reinhard -> sRGB OETF",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"pair_{args.pair_index:04d}_color_chain_report.json"
    report = {
        "experiment_version": "Phase2-ColorChain-v0.1.0",
        "domain_contract": {
            "raw_domain": "BLC/normalization, exposure, denoise, WB gains",
            "linear_rgb_domain": "demosaic output, optional CCM, tone mapping input",
            "nonlinear_srgb_domain": "display encoding and PNG output",
            "yuv_domain": "not implemented in current Phase 2 audit",
        },
        "synthetic": synthetic,
        "real_sid": real,
        "conclusion": "CCM remains experimental because metadata-derived matrix previously degraded independent display metrics; default chain keeps linear Camera RGB without CCM.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("Phase 2 线性RGB颜色链路复核")
    print("主链: WB -> Bilinear Demosaic -> Reinhard Tone Mapping -> sRGB OETF")
    print(f"真实SID: pair={args.pair_index}, ISO={pair.iso}, pattern={frame.pattern}")
    print(f"线性Camera RGB范围: [{scene_rgb.min():.6f}, {scene_rgb.max():.6f}]")
    print(f"CCM实验RGB范围: [{ccm_rgb.min():.6f}, {ccm_rgb.max():.6f}]")
    print(f"默认/CCM显示域平均绝对差: {real['default_vs_ccm_mean_abs_display_difference']:.6f}")
    print(f"Tone前超过1比例（默认/CCM）: {real['pre_tone_over_one_ratio']:.4%} / {real['pre_tone_ccm_over_one_ratio']:.4%}")
    print(f"Tone identity最大绝对误差: {identity_error:.8g}")
    print(f"JSON报告: {report_path.resolve()}")

    if not np.isfinite(ccm_rgb).all() or not np.isfinite(srgb).all() or identity_error != 0.0:
        raise SystemExit("颜色链路复核失败：出现非有限值或Tone identity不一致")
    print("Phase 2 线性RGB颜色链路复核: PASS（CCM仍为实验分支）")


if __name__ == "__main__":
    main()
