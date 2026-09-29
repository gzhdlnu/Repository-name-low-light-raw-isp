#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 7 Backend Benchmark
# Timestamp: 2026-09-24 | Author: Guozh | Version: Phase7-Benchmark-v0.1.0
# =============================================================================
"""Benchmark PyTorch CPU and ONNX Runtime backends on the same RAW tensor."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import RawDenoisingUNet  # noqa: E402


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _measure(callback, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        callback()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        values.append(time.perf_counter() - started)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 PyTorch/ONNX Runtime 后端基准")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ROOT / "outputs/phase7_deployment/onnx/raw_denoiser_b48.onnx",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/phase7_deployment/onnx/backend_benchmark.json",
    )
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 4 or args.width % 4:
        raise ValueError("height/width 必须为正数且能被 4 整除")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup 不能为负数，repeats 必须为正数")
    checkpoint_path = args.checkpoint.resolve()
    onnx_path = args.onnx.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    import onnxruntime as ort

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RawDenoisingUNet(base_channels=int(checkpoint["config"]["base_channels"])).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.manual_seed(args.seed)
    input_tensor = torch.rand(1, 4, args.height, args.width, dtype=torch.float32)
    with torch.inference_mode():
        torch_output = model(input_tensor).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(["restored_packed"], {input_name: input_tensor.numpy()})[0]
    difference = np.abs(torch_output - onnx_output)
    max_abs_error = float(np.max(difference))
    mean_abs_error = float(np.mean(difference))
    if max_abs_error > 1e-4:
        raise RuntimeError(f"后端输出误差过大: max={max_abs_error:.8g}")

    def run_torch() -> None:
        with torch.inference_mode():
            model(input_tensor)

    def run_onnx() -> None:
        session.run(["restored_packed"], {input_name: input_tensor.numpy()})

    torch_times = _measure(run_torch, args.warmup, args.repeats)
    onnx_times = _measure(run_onnx, args.warmup, args.repeats)
    report = {
        "experiment_version": "Phase7-Benchmark-v0.1.0",
        "shape": list(input_tensor.shape),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "providers": session.get_providers(),
        "pytorch_cpu": {
            "mean_seconds": float(np.mean(torch_times)),
            "p50_seconds": _percentile(torch_times, 50),
            "p95_seconds": _percentile(torch_times, 95),
        },
        "onnxruntime_cpu": {
            "mean_seconds": float(np.mean(onnx_times)),
            "p50_seconds": _percentile(onnx_times, 50),
            "p95_seconds": _percentile(onnx_times, 95),
        },
        "onnx_speedup_mean": float(np.mean(torch_times) / np.mean(onnx_times)),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Phase 7 PyTorch/ONNX Runtime 后端基准")
    print(f"输入: {tuple(input_tensor.shape)}，provider: {session.get_providers()}")
    print(f"输出最大/平均绝对误差: {max_abs_error:.8g} / {mean_abs_error:.8g}")
    print(
        f"PyTorch CPU mean/p50/p95: {report['pytorch_cpu']['mean_seconds']:.6f} / "
        f"{report['pytorch_cpu']['p50_seconds']:.6f} / {report['pytorch_cpu']['p95_seconds']:.6f}s"
    )
    print(
        f"ONNX CPU mean/p50/p95: {report['onnxruntime_cpu']['mean_seconds']:.6f} / "
        f"{report['onnxruntime_cpu']['p50_seconds']:.6f} / {report['onnxruntime_cpu']['p95_seconds']:.6f}s"
    )
    print(f"ONNX 平均加速比: {report['onnx_speedup_mean']:.3f}x")
    print(f"基准报告: {args.output.resolve()}")
    print("Phase 7 后端基准: PASS")


if __name__ == "__main__":
    main()
