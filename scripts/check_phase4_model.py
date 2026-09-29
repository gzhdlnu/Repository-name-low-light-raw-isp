#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 4 Model Forward/Backward Check
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase4-ModelCheck-v0.1.0
# =============================================================================
"""Run one AMP optimization step through the RAW denoising U-Net."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.sid_torch_dataset import SidRawPatchDataset  # noqa: E402
from models import RawDenoisingUNet, charbonnier_loss  # noqa: E402


def _gib(value: int) -> float:
    return value / 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 Phase 4 RAW U-Net 前向、反向和显存")
    parser.add_argument(
        "--train-list",
        type=Path,
        default=ROOT / "data" / "raw" / "SID" / "Sony_train_list.txt",
    )
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase4_learning" / "model_check",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不能执行 Phase 4 模型检查")
    device = torch.device("cuda:0")
    dataset = SidRawPatchDataset(
        args.train_list,
        patch_size=args.patch_size,
        patches_per_pair=1,
        augment=False,
        max_exposure_gain=300.0,
    )
    if not 0 <= args.pair_index < len(dataset):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(dataset)})")
    loader = DataLoader(
        Subset(dataset, [args.pair_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(loader))
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)

    model = RawDenoisingUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    activation_dtypes: dict[str, str] = {}
    output_hook = model.output.register_forward_hook(
        lambda _module, _inputs, output: activation_dtypes.update(output_head=str(output.dtype))
    )

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(inputs)
        loss_before_step = charbonnier_loss(prediction, targets)
    output_hook.remove()
    scaler.scale(loss_before_step).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("梯度包含 NaN 或 Inf")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction_after_step = model(inputs)
        loss_after_step = charbonnier_loss(prediction_after_step, targets)
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss_after_step):
        raise RuntimeError("模型输出或损失包含 NaN/Inf")

    report = {
        "experiment_version": "Phase4-ModelCheck-v0.1.0",
        "model": "RawDenoisingUNet",
        "base_channels": args.base_channels,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "input_shape": list(inputs.shape),
        "output_shape": list(prediction_after_step.shape),
        "input_dtype": str(inputs.dtype),
        "amp_output_head_dtype": activation_dtypes["output_head"],
        "final_prediction_dtype": str(prediction.dtype),
        "loss": "Charbonnier",
        "loss_before_step": float(loss_before_step.detach()),
        "loss_after_step": float(loss_after_step.detach()),
        "gradient_norm_before_clipping": float(gradient_norm),
        "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "status": "PASS",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "raw_unet_forward_backward_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"模型: RawDenoisingUNet，基础通道数: {args.base_channels}")
    print(f"参数量: {parameter_count:,}（可训练: {trainable_count:,}）")
    print(f"输入/输出: {tuple(inputs.shape)} -> {tuple(prediction_after_step.shape)}")
    print(f"AMP 卷积 head dtype: {activation_dtypes['output_head']}")
    print(f"残差相加后的最终 dtype: {prediction.dtype}")
    print(
        f"Charbonnier loss: {loss_before_step.item():.6f} -> "
        f"{loss_after_step.item():.6f}"
    )
    print(f"裁剪前梯度范数: {gradient_norm.item():.6f}")
    print(
        f"峰值显存: allocated={report['peak_allocated_gib']:.3f} GiB, "
        f"reserved={report['peak_reserved_gib']:.3f} GiB"
    )
    print(f"检查报告: {report_path}")
    print("Phase 4 模型前向/反向检查: PASS")


if __name__ == "__main__":
    main()
