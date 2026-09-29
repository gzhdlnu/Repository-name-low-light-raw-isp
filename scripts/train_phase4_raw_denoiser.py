#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 4 Learned RAW Denoising Training
# Timestamp: 2026-09-21 | Author: Guozh | Version: Phase4-Train-v0.1.0
# =============================================================================
"""Train a compact packed-RAW denoiser with scene-isolated SID validation."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.sid_torch_dataset import SidRawPatchDataset  # noqa: E402
from models import RawDenoisingUNet, charbonnier_loss, raw_chroma_loss, raw_edge_loss  # noqa: E402


def _gib(value: int) -> float:
    return value / 1024**3


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    absolute_errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(inputs)
                loss = charbonnier_loss(predictions, targets)
            losses.append(float(loss.detach().cpu()))
            absolute_errors.append(float(torch.mean(torch.abs(predictions - targets)).cpu()))
    model.train()
    return {
        "charbonnier_loss": float(np.mean(losses)),
        "mae": float(np.mean(absolute_errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Phase 4 packed RAW 降噪模型")
    parser.add_argument("--train-list", type=Path, default=ROOT / "data/raw/SID/Sony_train_list.txt")
    parser.add_argument("--val-list", type=Path, default=ROOT / "data/raw/SID/Sony_val_list.txt")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-pair", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-loss-weight", type=float, default=0.0)
    parser.add_argument("--chroma-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-exposure-gain", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase4_learning" / "training",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "phase4_learning",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不能开始 Phase 4 训练")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs、batch-size 必须为正，num-workers 不能为负")
    if args.edge_loss_weight < 0 or args.chroma_loss_weight < 0:
        raise ValueError("edge-loss-weight and chroma-loss-weight must be nonnegative")
    _seed_everything(args.seed)
    device = torch.device("cuda:0")

    train_dataset = SidRawPatchDataset(
        args.train_list,
        patch_size=args.patch_size,
        patches_per_pair=args.patches_per_pair,
        augment=True,
        max_exposure_gain=args.max_exposure_gain,
        seed=args.seed,
    )
    val_dataset = SidRawPatchDataset(
        args.val_list,
        patch_size=args.patch_size,
        patches_per_pair=1,
        augment=False,
        max_exposure_gain=args.max_exposure_gain,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = RawDenoisingUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    best_val = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val_charbonnier", best_val))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    global_step = 0
    start_time = time.time()

    print(
        f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, "
        f"batch={args.batch_size}, patch={args.patch_size}, epochs={args.epochs}"
    )
    print(f"GPU: {torch.cuda.get_device_name(device)}，AMP=FP16")
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(inputs)
                loss = charbonnier_loss(predictions, targets)
                if args.edge_loss_weight:
                    loss = loss + args.edge_loss_weight * raw_edge_loss(predictions, targets)
                if args.chroma_loss_weight:
                    loss = loss + args.chroma_loss_weight * raw_chroma_loss(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch}, step {global_step}: loss is NaN or Inf")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            value = float(loss.detach().cpu())
            epoch_losses.append(value)
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch {epoch:03d}, step {global_step:06d}: train loss={value:.7f}")

        val_metrics = _validate(model, val_loader, device)
        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_charbonnier_loss": train_loss,
            "val_charbonnier_loss": val_metrics["charbonnier_loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}: train={train_loss:.7f}, "
            f"val={val_metrics['charbonnier_loss']:.7f}, val MAE={val_metrics['mae']:.7f}"
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_charbonnier": min(best_val, val_metrics["charbonnier_loss"]),
            "config": vars(args),
        }
        latest_path = args.checkpoint_dir / "raw_denoiser_latest.pt"
        torch.save(checkpoint, latest_path)
        if val_metrics["charbonnier_loss"] < best_val:
            best_val = val_metrics["charbonnier_loss"]
            torch.save(checkpoint, args.checkpoint_dir / "raw_denoiser_best.pt")

    elapsed = time.time() - start_time
    report = {
        "experiment_version": "Phase4-Train-v0.1.0",
        "description_zh": "SID packed RAW 学习型降噪训练记录；验证集按官方场景隔离",
        "config": vars(args),
        "train_pairs": len(train_dataset.pairs),
        "val_pairs": len(val_dataset.pairs),
        "train_samples_per_epoch": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_charbonnier_loss": best_val,
        "global_steps": global_step,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(device)),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "history": history,
    }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"最佳验证 loss: {best_val:.7f}")
    print(f"峰值显存: {_gib(torch.cuda.max_memory_allocated(device)):.3f} GiB")
    print(f"训练报告: {report_path}")
    print(f"最佳权重: {args.checkpoint_dir / 'raw_denoiser_best.pt'}")


if __name__ == "__main__":
    main()
