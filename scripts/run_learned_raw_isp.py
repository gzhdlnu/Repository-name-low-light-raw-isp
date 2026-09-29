#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Production-Style Learned RAW ISP Inference
# Timestamp: 2026-09-23 | Author: Guozh | Version: Phase7-Infer-v0.1.0
# =============================================================================
"""Run the frozen learned RAW denoising and display ISP pipeline on one ARW."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import load_normalized_raw, render_packed_raw  # noqa: E402
from metrics import display_luma_summary  # noqa: E402
from models import RawDenoisingUNet, tiled_predict  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_input(args: argparse.Namespace) -> tuple[Path, float, int | None, str]:
    if args.sid_list is not None:
        if args.raw is not None:
            raise ValueError("RAW 路径和 --sid-list 只能选择一种输入方式")
        pairs = load_sid_list(args.sid_list)
        if not 0 <= args.pair_index < len(pairs):
            raise IndexError(f"pair-index {args.pair_index} 超出 [0, {len(pairs)})")
        pair = pairs[args.pair_index]
        gain = pair.exposure_gain if args.exposure_gain is None else args.exposure_gain
        source = (
            f"SID pair {args.pair_index}: "
            f"{pair.short_exposure_s:g}s -> {pair.long_exposure_s:g}s"
        )
        return pair.short_path, float(gain), pair.iso, source
    if args.raw is None:
        raise ValueError("请提供 RAW 路径，或使用 --sid-list 与 --pair-index")
    if args.exposure_gain is None:
        raise ValueError("直接处理 RAW 时必须明确提供 --exposure-gain")
    return args.raw.resolve(), float(args.exposure_gain), args.iso, "direct RAW input"


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="单图学习型 RAW ISP 推理")
    parser.add_argument("raw", type=Path, nargs="?", help="直接输入的 Sony ARW")
    parser.add_argument("--sid-list", type=Path, help="SID 官方 list；与直接 RAW 输入二选一")
    parser.add_argument("--pair-index", type=int, default=0, help="--sid-list 中的零基 pair 索引")
    parser.add_argument("--exposure-gain", type=float, help="曝光补偿倍率；可覆盖 SID 物理曝光比")
    parser.add_argument("--iso", type=int, help="直接 RAW 输入的 ISO，仅用于报告记录")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, help="输出 PNG；默认写入 Phase 7 推理目录")
    parser.add_argument("--report", type=Path, help="输出 JSON；默认与 PNG 同名")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA FP16 自动混合精度")
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument(
        "--use-color-matrix",
        action="store_true",
        help="启用未通过默认验收的实验性 CCM",
    )
    args = parser.parse_args()

    if args.exposure_gain is not None and args.exposure_gain <= 0:
        raise ValueError("exposure-gain 必须为正数")
    if args.iso is not None and args.iso <= 0:
        raise ValueError("ISO 必须为正整数")
    if args.white_point <= 0:
        raise ValueError("white-point 必须为正数")
    if args.tile_size <= 0 or args.tile_size % 4:
        raise ValueError("tile-size 必须为正数且能被 4 整除")
    if args.halo < 0 or args.halo % 4:
        raise ValueError("halo 必须非负且能被 4 整除")
    if args.tile_batch_size <= 0:
        raise ValueError("tile-batch-size 必须为正数")

    started = time.perf_counter()
    raw_path, requested_gain, iso, input_source = _resolve_input(args)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_path = args.output or (
        ROOT / "outputs/phase7_deployment/inference" / f"{raw_path.stem}_learned_isp.png"
    )
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output 必须使用 .png 扩展名")
    report_path = (args.report or output_path.with_suffix(".json")).resolve()
    device = _select_device(args.device)

    decode_started = time.perf_counter()
    frame = load_normalized_raw(raw_path)
    decode_seconds = time.perf_counter() - decode_started

    model_started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    base_channels = int(config["base_channels"])
    max_gain = float(config["max_exposure_gain"])
    model = RawDenoisingUNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model_load_seconds = time.perf_counter() - model_started

    applied_gain = min(requested_gain, max_gain)
    exposure_corrected = np.clip(frame.packed * applied_gain, 0.0, 1.0).astype(np.float32)
    input_tensor = (
        torch.from_numpy(exposure_corrected.transpose(2, 0, 1)).unsqueeze(0).to(device)
    )
    baseline_allocated = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    inference_started = time.perf_counter()
    prediction = tiled_predict(
        model,
        input_tensor,
        tile_size=args.tile_size,
        halo=args.halo,
        use_amp=not args.no_amp,
        tile_batch_size=args.tile_batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    restored = np.clip(
        prediction.squeeze(0).permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0
    )

    render_started = time.perf_counter()
    display_rgb, over_white = render_packed_raw(
        restored,
        frame.pattern,
        frame.white_balance_gains,
        tone_map=args.tone_map,
        white_point=args.white_point,
        color_matrix=frame.color_matrix if args.use_color_matrix else None,
    )
    render_seconds = time.perf_counter() - render_started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"无法写入输出图像: {output_path}")

    peak_allocated = 0
    incremental_peak = 0
    gpu_name = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        incremental_peak = max(0, peak_allocated - baseline_allocated)
        gpu_name = torch.cuda.get_device_name(device)
    report = {
        "experiment_version": "Phase7-Infer-v0.1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(raw_path.resolve()),
            "source": input_source,
            "iso": iso,
            "pattern": frame.pattern,
            "packed_shape": list(frame.packed.shape),
            "black_levels_r_g1_g2_b": list(frame.black_levels),
            "white_level": frame.white_level,
            "white_balance_gains_r_g1_g2_b": list(frame.white_balance_gains),
            "requested_exposure_gain": requested_gain,
            "applied_exposure_gain": applied_gain,
            "exposure_gain_was_clamped": applied_gain != requested_gain,
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "base_channels": base_channels,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "max_exposure_gain": max_gain,
        },
        "pipeline": {
            "tile_size": args.tile_size,
            "halo": args.halo,
            "tile_batch_size": args.tile_batch_size,
            "amp_enabled": not args.no_amp and device.type == "cuda",
            "tone_map": args.tone_map,
            "white_point": args.white_point,
            "color_matrix_applied": args.use_color_matrix,
            "pre_tone_over_white_ratio": over_white,
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "shape": list(display_rgb.shape),
            "luma_summary": display_luma_summary(display_rgb),
        },
        "performance": {
            "decode_seconds": decode_seconds,
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "render_seconds": render_seconds,
            "total_seconds": time.perf_counter() - started,
            "peak_allocated_gib": peak_allocated / 1024**3,
            "incremental_inference_peak_gib": incremental_peak / 1024**3,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": gpu_name,
            "cuda": torch.version.cuda,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("学习型 RAW ISP 单图推理完成")
    print(f"输入: {raw_path.resolve()}")
    print(f"输入来源: {input_source}")
    print(f"ISO: {iso if iso is not None else '未提供'}")
    print(f"曝光增益: 请求={requested_gain:.4f}, 实际={applied_gain:.4f}")
    print(f"模型: epoch={checkpoint['epoch']}, base_channels={base_channels}")
    print(f"推理: {inference_seconds:.3f}s, device={device}, AMP={report['pipeline']['amp_enabled']}")
    if device.type == "cuda":
        print(f"峰值显存: {report['performance']['peak_allocated_gib']:.3f} GiB")
    print(f"CCM: {'实验性启用' if args.use_color_matrix else '关闭（正式默认）'}")
    print(f"输出图像: {output_path}")
    print(f"推理报告: {report_path}")


if __name__ == "__main__":
    main()
