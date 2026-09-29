#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 5 Perceptual Metrics Environment Check
# Timestamp: 2026-09-23 | Author: Guozh | Version: Phase5-Env-v0.1.0
# =============================================================================
"""Check the optional LPIPS environment before running Phase 5 evaluations."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        import lpips  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "未安装 LPIPS，请先运行: python -m pip install lpips==0.1.4"
        ) from error

    try:
        import torchvision
    except ImportError as error:
        raise RuntimeError("LPIPS 依赖 torchvision，但当前环境无法导入 torchvision") from error

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Random backbone avoids downloading AlexNet weights during the environment check.
    metric = lpips.LPIPS(net="alex", pnet_rand=True, verbose=False).to(device).eval()
    prediction = torch.rand((1, 3, 64, 64), device=device) * 2.0 - 1.0
    reference = torch.rand((1, 3, 64, 64), device=device) * 2.0 - 1.0
    with torch.inference_mode():
        value = metric(prediction, reference)
    if value.ndim != 4 or value.shape != (1, 1, 1, 1):
        raise RuntimeError(f"LPIPS 输出 shape 异常: {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise RuntimeError("LPIPS 输出包含 NaN 或 Inf")

    report = {
        "experiment_version": "Phase5-Env-v0.1.0",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "lpips": getattr(lpips, "__version__", "unknown"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "test_input_shape": [1, 3, 64, 64],
        "test_output_shape": list(value.shape),
        "test_value": float(value.mean().item()),
        "pretrained_weights_checked": False,
    }
    output_dir = ROOT / "outputs/phase5_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "phase5_environment_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Phase 5 感知指标环境检查")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"LPIPS: {getattr(lpips, '__version__', 'unknown')}")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"随机输入: (1, 3, 64, 64)，LPIPS 输出: {tuple(value.shape)}")
    print(f"随机检查值: {float(value.mean().item()):.6f}")
    print("注意：本检查使用随机 backbone，不代表官方预训练 LPIPS 权重已经缓存")
    print(f"检查报告: {report_path}")
    print("Phase 5 环境检查: PASS")


if __name__ == "__main__":
    main()
