#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 6 Sensor-Parameter Robustness
# Timestamp: 2026-09-24 | Author: Guozh | Version: Phase6-Robust-v0.1.0
# =============================================================================
"""Evaluate frozen RAW U-Net under controlled sensor/input perturbations."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import load_normalized_raw, render_packed_raw  # noqa: E402
from metrics import compare_display_rgb  # noqa: E402
from models import RawDenoisingUNet, tiled_predict  # noqa: E402


def _indices(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("pair-indices 必须为非负整数")
    return values


def _perturbations(
    packed: np.ndarray,
    gain: float,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    base = np.clip(packed * gain, 0.0, 1.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    signal = np.clip(base, 0.0, 1.0)
    noise_std = np.sqrt(np.maximum(slopes.reshape(1, 1, 4) * signal + intercepts.reshape(1, 1, 4), 0.0))
    return {
        "基线": base,
        "曝光增益-5%": np.clip(packed * gain * 0.95, 0.0, 1.0),
        "曝光增益+5%": np.clip(packed * gain * 1.05, 0.0, 1.0),
        "R白平衡+5%": np.clip(base * np.array([1.05, 1.0, 1.0, 1.0], np.float32), 0.0, 1.0),
        "B白平衡-5%": np.clip(base * np.array([1.0, 1.0, 1.0, 0.95], np.float32), 0.0, 1.0),
        "黑电平偏高1%": np.clip(base - 0.01, 0.0, 1.0),
        "黑电平偏低1%": np.clip(base + 0.01, 0.0, 1.0),
        "额外噪声1.5倍": np.clip(base + rng.normal(0.0, noise_std * 1.5).astype(np.float32), 0.0, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 SID 传感器参数扰动鲁棒性评估")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-indices", type=_indices, default=[0, 14, 74])
    parser.add_argument("--noise-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt")
    parser.add_argument("--tile-batch-size", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase6_generalization/robustness_report.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    pairs = load_sid_list(args.sid_list)
    if max(args.pair_indices) >= len(pairs):
        raise IndexError("pair-index 超出 SID list 范围")
    noise = json.loads(args.noise_report.read_text(encoding="utf-8"))
    if int(noise["iso"]) != pairs[args.pair_indices[0]].iso:
        raise ValueError("噪声模型 ISO 与测试 pair 不一致")
    slopes = np.asarray(
        [noise["channels"][key]["variance_model"]["slope_a"] for key in ("R", "G1", "G2", "B")],
        np.float32,
    )
    intercepts = np.asarray(
        [noise["channels"][key]["variance_model"]["intercept_b"] for key in ("R", "G1", "G2", "B")],
        np.float32,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = RawDenoisingUNet(base_channels=int(checkpoint["config"]["base_channels"])).cuda().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    records: list[dict[str, object]] = []
    for pair_index in args.pair_indices:
        pair = pairs[pair_index]
        if pair.iso != pairs[args.pair_indices[0]].iso or round(pair.exposure_gain, 6) != round(pairs[args.pair_indices[0]].exposure_gain, 6):
            raise ValueError("一次评估内 pair 必须具有相同 ISO 和曝光增益")
        short = load_normalized_raw(pair.short_path)
        long = load_normalized_raw(pair.long_path)
        reference_rgb = render_packed_raw(long.packed, long.pattern, long.white_balance_gains, tone_map="reinhard", white_point=4.0)[0]
        variants = _perturbations(short.packed, pair.exposure_gain, slopes, intercepts, 20260924 + pair_index)
        for name, variant in variants.items():
            tensor = torch.from_numpy(variant.transpose(2, 0, 1)).unsqueeze(0).cuda()
            prediction = tiled_predict(model, tensor, tile_size=512, halo=32, use_amp=True, tile_batch_size=args.tile_batch_size)
            restored = np.clip(prediction.squeeze(0).permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0)
            rgb = render_packed_raw(restored, short.pattern, short.white_balance_gains, tone_map="reinhard", white_point=4.0)[0]
            metrics = compare_display_rgb(rgb, reference_rgb)
            records.append({"pair_index": pair_index, "scene_id": pair.short_path.stem.split("_")[0], "perturbation": name, **metrics})
        print(f"完成 pair {pair_index}（场景 {pair.short_path.stem.split('_')[0]}）")
    summary: dict[str, dict[str, float]] = {}
    for name in sorted({item["perturbation"] for item in records}):
        subset = [item for item in records if item["perturbation"] == name]
        summary[name] = {key: float(np.mean([item[key] for item in subset])) for key in ("psnr_db", "ssim", "delta_e2000_mean", "gradient_similarity")}
    report = {"experiment_version": "Phase6-Robust-v0.1.0", "description_zh": "SID 参数扰动鲁棒性，不等同于跨相机泛化", "pair_indices": args.pair_indices, "iso": pairs[args.pair_indices[0]].iso, "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n扰动均值（相对长曝光参考）")
    print("扰动                 PSNR      SSIM    DeltaE   梯度")
    for name, values in summary.items():
        print(f"{name:<18} {values['psnr_db']:>7.3f}  {values['ssim']:>7.5f}  {values['delta_e2000_mean']:>7.3f}  {values['gradient_similarity']:>7.5f}")
    print(f"报告: {args.output.resolve()}")
    print("Phase 6 参数扰动鲁棒性评估: PASS")


if __name__ == "__main__":
    main()
