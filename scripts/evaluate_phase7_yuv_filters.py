#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 7 YUV Filter Ablation
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase7-YUVFilter-v0.1.0
# =============================================================================
"""Compare conservative YNR/CNR/sharpen variants against SID long exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_sid_list  # noqa: E402
from isp import load_normalized_raw, render_packed_raw  # noqa: E402
from isp.yuv_filters import apply_yuv_filters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 YUV后端滤波消融")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=None)
    parser.add_argument("--pair-indices", type=str, default=None, help="逗号分隔的多个pair；与--pair-index二选一")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase7_deployment/yuv/yuv_filter_ablation.json")
    args = parser.parse_args()
    if args.pair_indices:
        pair_indices = [int(value) for value in args.pair_indices.split(",") if value.strip()]
    elif args.pair_index is not None:
        pair_indices = [args.pair_index]
    else:
        pair_indices = [0]
    variants = {
        "baseline": {},
        "YNR": {"ynr_sigma": 1.0},
        "CNR": {"cnr_sigma": 1.0},
        "YNR+CNR": {"ynr_sigma": 1.0, "cnr_sigma": 1.0},
        "YNR+CNR+Sharpen": {"ynr_sigma": 1.0, "cnr_sigma": 1.0, "sharpen_amount": 0.35},
    }
    per_pair = {}
    for pair_index in pair_indices:
        pair = load_sid_list(args.sid_list)[pair_index]
        short = load_normalized_raw(pair.short_path)
        long = load_normalized_raw(pair.long_path)
        short_rgb8, _ = render_packed_raw(short.packed, short.pattern, short.white_balance_gains, exposure_gain=min(pair.exposure_gain, 300.0))
        long_rgb8, _ = render_packed_raw(long.packed, long.pattern, long.white_balance_gains, exposure_gain=1.0)
        short_rgb = short_rgb8.astype(np.float32) / 255.0
        reference = long_rgb8.astype(np.float32) / 255.0
        per_pair[str(pair_index)] = {}
        for name, kwargs in variants.items():
            output = apply_yuv_filters(short_rgb, **kwargs)
            per_pair[str(pair_index)][name] = {
                "PSNR": float(peak_signal_noise_ratio(reference, output, data_range=1.0)),
                "SSIM": float(structural_similarity(reference, output, channel_axis=-1, data_range=1.0)),
                "mean_abs": float(np.mean(np.abs(reference - output))),
            }
    report = {"pair_indices": pair_indices, "variants": {}}
    for name in variants:
        report["variants"][name] = {
            metric: float(np.mean([per_pair[str(index)][name][metric] for index in pair_indices]))
            for metric in ("PSNR", "SSIM", "mean_abs")
        }
    report["per_pair"] = per_pair
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Phase 7 YUV后端滤波消融")
    print(f"场景数: {len(pair_indices)}, pair={','.join(map(str, pair_indices))}")
    print("方法                  PSNR(dB)   SSIM      MAE")
    for name, item in report["variants"].items():
        print(f"{name:<20} {item['PSNR']:8.3f} {item['SSIM']:8.5f} {item['mean_abs']:8.5f}")
    print(f"JSON报告: {args.output.resolve()}")


if __name__ == "__main__":
    main()
