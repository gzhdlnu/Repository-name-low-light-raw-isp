#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 4 Single-Patch Overfit Check
# Timestamp: 2026-09-21 | Author: Guozh | Version: Phase4-Overfit-v0.1.0
# =============================================================================
"""Overfit one fixed SID RAW patch to validate the complete training loop."""

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
    parser = argparse.ArgumentParser(description="单 patch 过拟合检查 Phase 4 训练闭环")
    parser.add_argument(
        "--train-list",
        type=Path,
        default=ROOT / "data" / "raw" / "SID" / "Sony_train_list.txt",
    )
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase4_learning" / "overfit_check",
    )
    args = parser.parse_args()

    if args.steps <= 0 or args.log_every <= 0:
        raise ValueError("steps and log-every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不能执行 GPU 过拟合检查")

    torch.manual_seed(20260921)
    torch.cuda.manual_seed_all(20260921)
    device = torch.device("cuda:0")
    dataset = SidRawPatchDataset(
        args.train_list,
        patch_size=args.patch_size,
        patches_per_pair=1,
        augment=False,
        max_exposure_gain=300.0,
        seed=20260920,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    best_loss = float("inf")
    best_step = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "single_patch_overfit_best.pt"

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(inputs)
            loss = charbonnier_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"step {step}: loss is NaN or Inf")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"step {step}: gradient is NaN or Inf")
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "loss": loss_value,
                    "pair_index": args.pair_index,
                    "patch_size": args.patch_size,
                    "base_channels": args.base_channels,
                },
                checkpoint_path,
            )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step {step:04d}/{args.steps}: Charbonnier loss={loss_value:.7f}")

    first_loss = losses[0]
    last_loss = losses[-1]
    improvement_ratio = (first_loss - last_loss) / max(abs(first_loss), 1e-12)
    if last_loss >= first_loss * 0.85:
        raise RuntimeError(
            f"过拟合检查未达到预期下降：first={first_loss:.7f}, last={last_loss:.7f}"
        )

    report = {
        "experiment_version": "Phase4-Overfit-v0.1.0",
        "description_zh": "固定单 patch 过拟合，用于验证训练闭环而非评估泛化能力",
        "pair_index": args.pair_index,
        "scene_id": batch["scene_id"][0],
        "patch_size": args.patch_size,
        "base_channels": args.base_channels,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "improvement_ratio": improvement_ratio,
        "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
        "gpu": torch.cuda.get_device_name(device),
        "status": "PASS",
    }
    report_path = args.output_dir / "single_patch_overfit_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"首步 loss: {first_loss:.7f}")
    print(f"末步 loss: {last_loss:.7f}，下降比例: {improvement_ratio:.2%}")
    print(f"最佳 checkpoint: {checkpoint_path}")
    print(f"检查报告: {report_path}")
    print("Phase 4 单 patch 过拟合检查: PASS")


if __name__ == "__main__":
    main()
