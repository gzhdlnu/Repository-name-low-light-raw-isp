#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 5 Pretrained LPIPS Check
# Timestamp: 2026-09-23 | Author: Guozh | Version: Phase5-LPIPS-v0.1.0
# =============================================================================
"""Verify that official pretrained LPIPS weights can be loaded and evaluated."""

from __future__ import annotations

from pathlib import Path
import json

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        import lpips  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "未安装 LPIPS，请先运行: python -m pip install lpips==0.1.4"
        ) from error

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Phase 5 LPIPS 官方预训练权重检查")
    print("首次运行可能下载 AlexNet 权重，请等待下载完成。")
    metric = lpips.LPIPS(net="alex", pnet_rand=False, verbose=True).to(device).eval()
    prediction = torch.zeros((1, 3, 64, 64), device=device)
    reference = torch.ones((1, 3, 64, 64), device=device) * 0.25
    with torch.inference_mode():
        value = metric(prediction, reference)
    if value.shape != (1, 1, 1, 1) or not torch.isfinite(value).all():
        raise RuntimeError(f"LPIPS 正式权重输出异常: shape={tuple(value.shape)}")

    report = {
        "experiment_version": "Phase5-LPIPS-v0.1.0",
        "device": str(device),
        "network": "alex",
        "pretrained": True,
        "test_output_shape": list(value.shape),
        "test_value": float(value.item()),
    }
    output_dir = ROOT / "outputs/phase5_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "phase5_pretrained_lpips_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"设备: {device}")
    print(f"LPIPS(AlexNet) 测试输出: {float(value.item()):.6f}")
    print(f"检查报告: {report_path}")
    print("Phase 5 LPIPS 预训练权重检查: PASS")


if __name__ == "__main__":
    main()
