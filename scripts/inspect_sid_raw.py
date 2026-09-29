#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: SID RAW Metadata Inspection
# Timestamp: 2026-09-20 | Author: Guozh | Version: Data-SID-Inspect-v0.1.0
# =============================================================================
"""Inspect one SID Sony ARW without writing a converted image."""

from pathlib import Path
import argparse

import rawpy


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SID Sony RAW metadata")
    parser.add_argument("raw", type=Path, help="Path to an .ARW file")
    args = parser.parse_args()
    if not args.raw.is_file():
        raise FileNotFoundError(args.raw)

    with rawpy.imread(str(args.raw)) as image:
        pattern = "".join("RGBG"[int(v)] for v in image.raw_pattern.flat)
        print(f"path: {args.raw}")
        print(f"raw_image shape: {image.raw_image.shape}, dtype: {image.raw_image.dtype}")
        print(f"visible shape: {image.raw_image_visible.shape}")
        print(f"raw pattern: {pattern}")
        print(f"black level per channel: {image.black_level_per_channel}")
        print(f"white level: {image.white_level}")
        print(f"camera white balance: {image.camera_whitebalance}")
        print(f"daylight white balance: {image.daylight_whitebalance}")
        print("exposure metadata: read from the official SID *_list.txt entry")


if __name__ == "__main__":
    main()
