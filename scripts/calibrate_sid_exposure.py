#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: SID Effective Exposure Calibration
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase2-ExposureCal-v0.1.1
# =============================================================================
"""Calibrate display-effective gain against one SID long-exposure reference.

This is an offline calibration tool. The long reference must never be used by
the deployed/inference pipeline to choose gain.
"""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import process_sony_raw  # noqa: E402
from metrics import compare_display_rgb, display_luma_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SID short-exposure effective gain")
    parser.add_argument("--sid-list", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument(
        "--gain-scales",
        type=float,
        nargs="+",
        default=(0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4),
        help="Multipliers around the physical exposure ratio",
    )
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase2_color_tone" / "calibration" / "pair_0000_exposure.json",
    )
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    if any(scale <= 0 for scale in args.gain_scales):
        raise ValueError("all gain scales must be positive")
    pair = pairs[args.pair_index]

    reference = process_sony_raw(pair.long_path, 1.0, args.tone_map, args.white_point)
    reference_summary = display_luma_summary(reference.display_rgb)
    candidates = []
    for scale in args.gain_scales:
        effective_gain = pair.exposure_gain * scale
        short = process_sony_raw(pair.short_path, effective_gain, args.tone_map, args.white_point)
        summary = display_luma_summary(short.display_rgb)
        quality = compare_display_rgb(short.display_rgb, reference.display_rgb)
        candidates.append(
            {
                "gain_scale": scale,
                "effective_gain": effective_gain,
                "median_luma": summary["luma_p50"],
                "median_luma_absolute_error": abs(summary["luma_p50"] - reference_summary["luma_p50"]),
                "psnr_db": quality["psnr_db"],
                "ssim": quality["ssim"],
                "pre_tone_over_white_ratio": short.pre_tone_over_white_ratio,
            }
        )

    # Exposure is selected using robust median luminance, not PSNR/SSIM. This
    # prevents noise texture from steering an exposure-control decision.
    best = min(candidates, key=lambda item: item["median_luma_absolute_error"])
    selected_index = candidates.index(best)
    selection_status = "boundary_not_converged" if selected_index in (0, len(candidates) - 1) else "interior_minimum"
    report = {
        "pipeline_version": "Phase2-ExposureCal-v0.1.1",
        "calibration_scope": "offline only; long reference unavailable during deployment",
        "pair_index": args.pair_index,
        "physical_exposure_gain": pair.exposure_gain,
        "reference_median_luma": reference_summary["luma_p50"],
        "tone_map": args.tone_map,
        "white_point": args.white_point,
        "selection_metric": "minimum absolute median display-luma error",
        "selection_status": selection_status,
        "selected": best,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"physical gain: {pair.exposure_gain:g}")
    print(f"reference median luma: {reference_summary['luma_p50']:.6f}")
    print("scale  gain       median     abs-error  PSNR(dB)  SSIM")
    for item in candidates:
        print(
            f"{item['gain_scale']:>5.2f}  {item['effective_gain']:>8.3f}  "
            f"{item['median_luma']:>9.6f}  {item['median_luma_absolute_error']:>9.6f}  "
            f"{item['psnr_db']:>8.3f}  {item['ssim']:.5f}"
        )
    print(f"selected effective gain: {best['effective_gain']:.3f}")
    if selection_status == "boundary_not_converged":
        print("WARNING: selected gain is on the search boundary; calibration has not converged")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
