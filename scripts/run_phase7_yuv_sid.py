#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 7 SID RGB-to-NV12 Smoke Test
# Timestamp: 2026-09-29 | Author: Guozh | Version: Phase7-YUV-v0.2.0
# =============================================================================
"""Render one SID short exposure and write RGB plus full/limited NV12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_sid_list  # noqa: E402
from isp import load_normalized_raw, render_packed_raw  # noqa: E402
from isp.yuv import rgb_to_ycbcr, ycbcr_to_nv12, ycbcr_to_rgb  # noqa: E402


def _write_nv12(path: Path, y: np.ndarray, uv: np.ndarray) -> None:
    path.write_bytes(y.tobytes(order="C") + uv.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description="真实SID RGB到NV12输出验收")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase7_deployment/yuv/sid_smoke")
    args = parser.parse_args()

    pairs = load_sid_list(args.sid_list)
    pair = pairs[args.pair_index]
    frame = load_normalized_raw(pair.short_path)
    rgb8, over_white = render_packed_raw(
        frame.packed,
        frame.pattern,
        frame.white_balance_gains,
        exposure_gain=min(pair.exposure_gain, 300.0),
        tone_map="reinhard",
        white_point=4.0,
    )
    rgb = rgb8.astype(np.float32) / 255.0
    if rgb.shape[0] % 2 or rgb.shape[1] % 2:
        raise ValueError("SID RGB尺寸不是偶数，无法输出4:2:0 NV12")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / f"pair_{args.pair_index:04d}_rgb.png"), cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))
    report: dict[str, object] = {"pair_index": args.pair_index, "shape": list(rgb.shape), "over_white_ratio": float(over_white)}
    for limited in (False, True):
        name = "limited" if limited else "full"
        yuv = rgb_to_ycbcr(rgb, limited_range=limited)
        restored = ycbcr_to_rgb(yuv, limited_range=limited)
        y, uv = ycbcr_to_nv12(rgb, limited_range=limited)
        _write_nv12(args.output_dir / f"pair_{args.pair_index:04d}_{name}.nv12", y, uv)
        report[name] = {
            "round_trip_max_abs_error": float(np.max(np.abs(rgb - restored))),
            "round_trip_mean_abs_error": float(np.mean(np.abs(rgb - restored))),
            "y_range": [int(y.min()), int(y.max())],
            "uv_range": [int(uv.min()), int(uv.max())],
            "y_shape": list(y.shape),
            "uv_shape": list(uv.shape),
            "bytes": int(y.nbytes + uv.nbytes),
        }
    report_path = args.output_dir / f"pair_{args.pair_index:04d}_yuv_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Phase 7 真实SID RGB/NV12输出验收")
    print(f"pair={args.pair_index}, RGB={rgb.shape}, over-white={over_white:.4%}")
    for name in ("full", "limited"):
        item = report[name]
        print(f"{name}: round-trip max={item['round_trip_max_abs_error']:.7f}, Y range={item['y_range']}, UV range={item['uv_range']}, bytes={item['bytes']}")
    print(f"RGB预览: {args.output_dir / f'pair_{args.pair_index:04d}_rgb.png'}")
    print(f"JSON报告: {report_path.resolve()}")
    if any(report[name]["round_trip_max_abs_error"] > 2.0 / 255.0 for name in ("full", "limited")):
        raise SystemExit("真实SID YCbCr round-trip误差超过阈值")
    print("Phase 7 真实SID RGB/NV12输出验收: PASS")


if __name__ == "__main__":
    main()
