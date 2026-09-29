#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 4 PyTorch CUDA Environment Check
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase4-Env-v0.1.0
# =============================================================================
"""Verify that the Phase 4 training environment can execute CUDA and AMP."""

from __future__ import annotations

import platform

import cv2
import numpy as np
import rawpy
import torch


def _gib(value: int) -> float:
    return value / 1024**3


def main() -> None:
    print("Phase 4 学习型 RAW 降噪环境检查")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch 编译 CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"rawpy: {rawpy.__version__}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 无法使用 CUDA；不能进入 Phase 4 GPU 训练")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(f"GPU: {properties.name}")
    print(f"CUDA capability: {properties.major}.{properties.minor}")
    print(f"显存总量: {_gib(properties.total_memory):.2f} GiB")
    print(f"当前空闲显存: {_gib(free_bytes):.2f} / {_gib(total_bytes):.2f} GiB")

    torch.cuda.reset_peak_memory_stats(device)
    left = torch.randn((2048, 2048), device=device, dtype=torch.float32)
    right = torch.randn((2048, 2048), device=device, dtype=torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = left @ right
        checksum = result.float().mean()
    torch.cuda.synchronize(device)
    if not torch.isfinite(checksum):
        raise RuntimeError("GPU AMP 运算产生 NaN 或 Inf")

    print(f"AMP 输出 dtype: {result.dtype}")
    print(f"测试峰值显存: {_gib(torch.cuda.max_memory_allocated(device)):.3f} GiB")
    print(f"数值检查: mean={checksum.item():.6g}")
    print("环境检查: PASS")


if __name__ == "__main__":
    main()
