#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 7 ONNX Export and Numerical Verification
# Timestamp: 2026-09-24 | Author: Guozh | Version: Phase7-ONNX-v0.1.0
# =============================================================================
"""Export the frozen RAW U-Net to ONNX and compare it with PyTorch on CPU."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="导出并验证 Phase 7 RAW U-Net ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/phase7_deployment/onnx/raw_denoiser_b48.onnx",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--verify-height", type=int, default=512)
    parser.add_argument("--verify-width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    for name, value in (
        ("height", args.height), ("width", args.width),
        ("verify-height", args.verify_height), ("verify-width", args.verify_width),
    ):
        if value <= 0 or value % 4:
            raise ValueError(f"{name} 必须为正数且能被 4 整除")
    if args.opset < 17:
        raise ValueError("opset 至少为 17")
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = RawDenoisingUNet(base_channels=int(config["base_channels"])).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.manual_seed(args.seed)
    dummy = torch.rand(1, 4, args.height, args.width, dtype=torch.float32)
    with torch.inference_mode():
        torch_output = model(dummy).numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_started = time.perf_counter()
    torch.onnx.export(
        model,
        (dummy,),
        str(args.output),
        input_names=["raw_packed"],
        output_names=["restored_packed"],
        dynamic_axes={
            "raw_packed": {0: "batch", 2: "height", 3: "width"},
            "restored_packed": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    export_seconds = time.perf_counter() - export_started

    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    ort_input = {session.get_inputs()[0].name: dummy.numpy()}
    ort_started = time.perf_counter()
    ort_output = session.run(["restored_packed"], ort_input)[0]
    ort_seconds = time.perf_counter() - ort_started
    difference = np.abs(torch_output - ort_output)
    max_abs_error = float(np.max(difference))
    mean_abs_error = float(np.mean(difference))
    if max_abs_error > 1e-4:
        raise RuntimeError(
            f"ONNX 数值误差过大: max_abs_error={max_abs_error:.8g}, "
            f"mean_abs_error={mean_abs_error:.8g}"
        )

    torch.manual_seed(args.seed + 1)
    dynamic_dummy = torch.rand(1, 4, args.verify_height, args.verify_width, dtype=torch.float32)
    with torch.inference_mode():
        dynamic_torch_output = model(dynamic_dummy).numpy()
    dynamic_ort_output = session.run(
        ["restored_packed"], {session.get_inputs()[0].name: dynamic_dummy.numpy()}
    )[0]
    dynamic_difference = np.abs(dynamic_torch_output - dynamic_ort_output)
    dynamic_max_abs_error = float(np.max(dynamic_difference))
    dynamic_mean_abs_error = float(np.mean(dynamic_difference))
    if dynamic_max_abs_error > 1e-4:
        raise RuntimeError(
            f"ONNX 动态尺寸误差过大: max_abs_error={dynamic_max_abs_error:.8g}, "
            f"mean_abs_error={dynamic_mean_abs_error:.8g}"
        )

    report = {
        "experiment_version": "Phase7-ONNX-v0.1.0",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_channels": int(config["base_channels"]),
        "opset": args.opset,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "providers": session.get_providers(),
        "test_shape": list(dummy.shape),
        "dynamic_test_shape": list(dynamic_dummy.shape),
        "dynamic_axes": True,
        "export_seconds": export_seconds,
        "onnxruntime_cpu_seconds": ort_seconds,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "dynamic_max_abs_error": dynamic_max_abs_error,
        "dynamic_mean_abs_error": dynamic_mean_abs_error,
        "output": str(args.output.resolve()),
        "status": "PASS",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Phase 7 ONNX 导出与数值检查")
    print(f"模型: base_channels={config['base_channels']}, epoch={checkpoint['epoch']}")
    print(f"输入 shape: {tuple(dummy.shape)}，opset={args.opset}")
    print(f"ONNX: {onnx.__version__}，ONNX Runtime: {ort.__version__}")
    print(f"PyTorch 输出 shape: {tuple(torch_output.shape)}")
    print(f"最大绝对误差: {max_abs_error:.8g}")
    print(f"平均绝对误差: {mean_abs_error:.8g}")
    print(
        f"动态尺寸 {tuple(dynamic_dummy.shape)} 最大/平均误差: "
        f"{dynamic_max_abs_error:.8g} / {dynamic_mean_abs_error:.8g}"
    )
    print(f"导出耗时: {export_seconds:.3f}s，CPU ORT 推理: {ort_seconds:.3f}s")
    print(f"ONNX 文件: {args.output.resolve()}")
    print(f"验证报告: {report_path.resolve()}")
    print("ONNX 导出与数值检查: PASS")


if __name__ == "__main__":
    main()
