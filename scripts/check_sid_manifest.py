#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: SID Manifest Validation
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase2-Data-v0.1.0
# =============================================================================
"""Validate and summarize one official SID list. Run manually."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an official SID list")
    parser.add_argument("list_path", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    pairs = load_sid_list(args.list_path)
    missing = []
    if args.check_files:
        missing = [p for pair in pairs for p in (pair.short_path, pair.long_path) if not p.is_file()]

    print(f"list: {args.list_path}")
    print(f"pairs: {len(pairs)}")
    if pairs:
        first = pairs[0]
        print(f"first short: {first.short_path}")
        print(f"first long: {first.long_path}")
        print(f"exposure: {first.short_exposure_s}s -> {first.long_exposure_s}s")
        print(f"exposure gain: {first.exposure_gain:g}")
        print(f"ISO: {first.iso}, aperture: F{first.aperture:g}")
    if args.check_files:
        print(f"missing files: {len(missing)}")
        if missing:
            raise SystemExit("SID manifest contains missing RAW files")


if __name__ == "__main__":
    main()
