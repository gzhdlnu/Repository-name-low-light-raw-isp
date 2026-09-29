#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: SID Real RAW Baseline Conversion
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase2-Pipeline-v0.4.0
# =============================================================================
"""Convert one SID Sony ARW with the project's explicit ISP baseline."""

from pathlib import Path
import argparse
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isp import process_sony_raw  # noqa: E402
from data import load_sid_list  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one SID Sony RAW using the baseline ISP")
    parser.add_argument("raw", type=Path, nargs="?", help="Direct path to an .ARW file")
    parser.add_argument("--sid-list", type=Path, help="Official SID list file")
    parser.add_argument("--pair-index", type=int, default=0, help="Zero-based pair index in --sid-list")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase2_color_tone" / "sid_sample_baseline.png",
    )
    parser.add_argument(
        "--exposure-gain",
        type=float,
        default=None,
        help="Override the linear exposure multiplier; otherwise read it from --sid-list",
    )
    parser.add_argument(
        "--use-color-matrix",
        action="store_true",
        help="Apply the experimental camera-to-sRGB matrix; disabled by default until calibrated",
    )
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument(
        "--white-point",
        type=float,
        default=4.0,
        help="Scene-linear white point for Reinhard tone mapping",
    )
    args = parser.parse_args()

    if args.sid_list:
        pairs = load_sid_list(args.sid_list)
        if not 0 <= args.pair_index < len(pairs):
            raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
        pair = pairs[args.pair_index]
        raw_path = pair.short_path
        exposure_gain = pair.exposure_gain if args.exposure_gain is None else args.exposure_gain
        pair_info = f"SID pair index={args.pair_index}, exposure={pair.short_exposure_s}s->{pair.long_exposure_s}s"
    elif args.raw:
        raw_path = args.raw.resolve()
        if args.exposure_gain is None:
            raise ValueError("--exposure-gain is required when --sid-list is not provided")
        exposure_gain = args.exposure_gain
        pair_info = "direct RAW input"
    else:
        raise ValueError("provide a RAW path or --sid-list")

    result = process_sony_raw(
        raw_path,
        exposure_gain=exposure_gain,
        tone_map=args.tone_map,
        white_point=args.white_point,
        use_color_matrix=args.use_color_matrix,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(result.display_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {args.output}")
    print(f"input: {raw_path}")
    print(f"pair: {pair_info}")
    print(
        f"pattern={result.pattern}, black={result.black_level:.1f}, "
        f"white={result.white_level:.1f}, wb={list(result.white_balance_gains)}, "
        f"exposure_gain={result.exposure_gain:.4f}"
    )
    print(f"color_matrix_applied={args.use_color_matrix}")
    print(f"tone_map={args.tone_map}, white_point={args.white_point:g}")
    print(f"pre-tone pixels above display white: {result.pre_tone_over_white_ratio:.2%}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
