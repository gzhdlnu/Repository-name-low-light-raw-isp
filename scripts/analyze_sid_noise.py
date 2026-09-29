#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: SID RAW Noise Profile
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase3-Noise-v0.1.0
# =============================================================================
"""Analyze gain-normalized RAW residual noise for one SID pair."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import rawpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import estimate_residual_noise_profile  # noqa: E402
from data import load_sid_list  # noqa: E402
from isp import normalize_raw, pack_bayer  # noqa: E402


def _load_packed(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with rawpy.imread(str(path)) as image:
        raw = image.raw_image_visible.astype(np.float32)
        black = float(min(image.black_level_per_channel))
        white = float(image.white_level)
        pattern = "".join("RGBG"[int(v)] for v in image.raw_pattern.flat)
    normalized = normalize_raw(raw, black, white, clip=False)
    return pack_bayer(normalized, pattern), {"black_level": black, "white_level": white, "pattern": pattern}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RAW residual noise for one SID pair")
    parser.add_argument("--sid-list", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--max-signal", type=float, default=0.8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase3_noise",
    )
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    if not 0 <= args.pair_index < len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs)})")
    pair = pairs[args.pair_index]
    short, short_meta = _load_packed(pair.short_path)
    long, long_meta = _load_packed(pair.long_path)
    edges = np.linspace(0.005, args.max_signal, args.bins + 1, dtype=np.float32)
    channels = estimate_residual_noise_profile(short, long, pair.exposure_gain, edges)

    report = {
        "pipeline_version": "Phase3-Noise-v0.1.0",
        "measurement_scope": "gain-normalized short-vs-long residual; not a laboratory sensor noise profile",
        "pair_index": args.pair_index,
        "short_path": str(pair.short_path),
        "long_path": str(pair.long_path),
        "exposure_gain": pair.exposure_gain,
        "iso": pair.iso,
        "short_metadata": short_meta,
        "long_metadata": long_meta,
        "flat_fraction": 0.4,
        "signal_range": [float(edges[0]), float(edges[-1])],
        "channels": channels,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{args.pair_index:04d}_raw_noise"
    report_path = args.output_dir / f"{stem}.json"
    plot_path = args.output_dir / f"{stem}.png"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (name, channel) in zip(axes.flat, channels.items()):
        bins = channel["bins"]
        signal = np.array([item["signal"] for item in bins])
        variance = np.array([item["variance"] for item in bins])
        axis.scatter(signal, variance, s=28, label="robust measured variance")
        model = channel["variance_model"]
        if signal.size:
            grid = np.linspace(signal.min(), signal.max(), 100)
            axis.plot(grid, model["slope_a"] * grid + model["intercept_b"], label="a*signal+b")
        axis.set_title(name)
        axis.set_xlabel("reference linear RAW signal")
        axis.set_ylabel("gain-normalized residual variance")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"pair: {args.pair_index}, exposure gain: {pair.exposure_gain:g}, ISO: {pair.iso}")
    for name, channel in channels.items():
        model = channel["variance_model"]
        print(
            f"{name}: bins={len(channel['bins'])}, "
            f"variance = {model['slope_a']:.6g} * signal + {model['intercept_b']:.6g}"
        )
    print(f"plot: {plot_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
