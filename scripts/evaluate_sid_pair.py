#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: SID Pair Exposure/Tone Evaluation
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase2-Eval-v0.2.0
# =============================================================================
"""Compare an exposure-corrected SID short RAW with its long reference."""

from pathlib import Path
import argparse
import json
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import process_sony_raw  # noqa: E402
from metrics import compare_display_rgb, display_luma_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one aligned SID short/long pair")
    parser.add_argument("--sid-list", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument("--use-color-matrix", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "phase2_color_tone")
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]

    short = process_sony_raw(
        pair.short_path, pair.exposure_gain, args.tone_map, args.white_point, args.use_color_matrix
    )
    reference = process_sony_raw(
        pair.long_path, 1.0, args.tone_map, args.white_point, args.use_color_matrix
    )
    metrics = compare_display_rgb(short.display_rgb, reference.display_rgb)
    report = {
        "pipeline_version": "Phase2-Eval-v0.2.0",
        "pair_index": args.pair_index,
        "short_path": str(pair.short_path),
        "long_path": str(pair.long_path),
        "short_exposure_s": pair.short_exposure_s,
        "long_exposure_s": pair.long_exposure_s,
        "exposure_gain": pair.exposure_gain,
        "iso": pair.iso,
        "aperture": pair.aperture,
        "tone_map": args.tone_map,
        "white_point": args.white_point,
        "color_matrix_applied": args.use_color_matrix,
        "short_pre_tone_over_white_ratio": short.pre_tone_over_white_ratio,
        "reference_pre_tone_over_white_ratio": reference.pre_tone_over_white_ratio,
        "full_reference_metrics": metrics,
        "short_display_summary": display_luma_summary(short.display_rgb),
        "reference_display_summary": display_luma_summary(reference.display_rgb),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    color_tag = "ccm" if args.use_color_matrix else "camera_rgb"
    stem = f"pair_{args.pair_index:04d}_{args.tone_map}_wp{args.white_point:g}_{color_tag}"
    comparison_path = args.output_dir / f"{stem}_comparison.png"
    report_path = args.output_dir / f"{stem}_report.json"
    comparison = cv2.hconcat(
        [cv2.cvtColor(short.display_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(reference.display_rgb, cv2.COLOR_RGB2BGR)]
    )
    cv2.putText(comparison, "SHORT x exposure gain", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(
        comparison,
        "LONG reference",
        (short.display_rgb.shape[1] + 30, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )
    if not cv2.imwrite(str(comparison_path), comparison):
        raise RuntimeError(f"Failed to write {comparison_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"PSNR: {metrics['psnr_db']:.3f} dB")
    print(f"SSIM: {metrics['ssim']:.5f}")
    print(f"DeltaE2000 mean/p50/p95: {metrics['delta_e2000_mean']:.3f} / {metrics['delta_e2000_p50']:.3f} / {metrics['delta_e2000_p95']:.3f}")
    print(f"comparison: {comparison_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
