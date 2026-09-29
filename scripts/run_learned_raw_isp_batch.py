#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 7 Batch RAW ISP Inference
# Timestamp: 2026-09-24 | Author: Guozh | Version: Phase7-BatchInfer-v0.1.0
# =============================================================================
"""Run the frozen learned RAW ISP on multiple files with one model load."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import json
import sys
import time
import traceback

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from isp import load_normalized_raw, render_packed_raw  # noqa: E402
from models import RawDenoisingUNet, tiled_predict  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT / "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt"
)


@dataclass(frozen=True)
class InputItem:
    index: int
    path: Path
    exposure_gain: float
    iso: int | None
    source_id: str


def _parse_indices(text: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair-indices 必须为逗号分隔的整数") from error
    if not values or any(value < 0 for value in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("pair-indices 不能为空、不能为负数或重复")
    return values


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _collect_inputs(args: argparse.Namespace) -> list[InputItem]:
    if (args.sid_list is None) == (args.input_dir is None):
        raise ValueError("--sid-list 和 --input-dir 必须且只能指定一个")
    if args.sid_list is not None:
        if args.pair_indices is None:
            raise ValueError("SID 批量推理必须明确提供 --pair-indices，避免误跑完整数据集")
        pairs = load_sid_list(args.sid_list)
        if max(args.pair_indices) >= len(pairs):
            raise IndexError(f"pair-index 超出范围，list 共 {len(pairs)} 个 pair")
        return [
            InputItem(
                index=index,
                path=pairs[index].short_path,
                exposure_gain=float(pairs[index].exposure_gain),
                iso=int(pairs[index].iso),
                source_id=f"pair_{index:04d}",
            )
            for index in args.pair_indices
        ]
    if args.exposure_gain is None or args.exposure_gain <= 0:
        raise ValueError("目录输入必须提供正数 --exposure-gain")
    paths = sorted(
        path for path in args.input_dir.resolve().iterdir()
        if path.is_file() and path.suffix.lower() in {".arw", ".dng", ".nef", ".cr2", ".cr3"}
    )
    if not paths:
        raise FileNotFoundError(f"输入目录中没有支持的 RAW 文件: {args.input_dir}")
    return [
        InputItem(
            index=position,
            path=path,
            exposure_gain=float(args.exposure_gain),
            iso=args.iso,
            source_id=f"file_{position:04d}",
        )
        for position, path in enumerate(paths)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="目录级学习型 RAW ISP 批量推理")
    parser.add_argument("--sid-list", type=Path)
    parser.add_argument("--pair-indices", type=_parse_indices)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--exposure-gain", type=float)
    parser.add_argument("--iso", type=int, help="目录输入的 ISO，仅用于报告")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase7_deployment/batch_inference",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1, help="不计入统计的 GPU 预热次数")
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument("--fail-fast", action="store_true", help="遇到首个失败文件立即退出")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；正式批量基线要求 PyTorch CUDA")
    if args.tile_size <= 0 or args.tile_size % 4 or args.halo < 0 or args.halo % 4:
        raise ValueError("tile-size 必须为正且能被 4 整除；halo 必须非负且能被 4 整除")
    if args.tile_batch_size <= 0:
        raise ValueError("tile-batch-size 必须为正数")
    if args.warmup < 0 or args.white_point <= 0:
        raise ValueError("warmup 不能为负数，white-point 必须为正数")
    items = _collect_inputs(args)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda:0")
    load_started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = RawDenoisingUNet(base_channels=int(config["base_channels"])).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    model_load_seconds = time.perf_counter() - load_started
    max_gain = float(config["max_exposure_gain"])

    # Warm up kernels using a tile-sized tensor; warmup is excluded from latency statistics.
    warmup_tensor = torch.zeros(1, 4, args.tile_size, args.tile_size, device=device)
    for _ in range(args.warmup):
        tiled_predict(
            model, warmup_tensor, args.tile_size, args.halo,
            use_amp=True, tile_batch_size=args.tile_batch_size,
        )
    torch.cuda.synchronize(device)
    del warmup_tensor
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    batch_started = time.perf_counter()
    for sequence, item in enumerate(items, start=1):
        try:
            item_started = time.perf_counter()
            decode_started = time.perf_counter()
            frame = load_normalized_raw(item.path)
            decode_seconds = time.perf_counter() - decode_started
            applied_gain = min(item.exposure_gain, max_gain)
            exposure_corrected = np.clip(
                frame.packed * applied_gain, 0.0, 1.0
            ).astype(np.float32)
            input_tensor = torch.from_numpy(
                exposure_corrected.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            inference_started = time.perf_counter()
            prediction = tiled_predict(
                model, input_tensor, args.tile_size, args.halo,
                use_amp=True, tile_batch_size=args.tile_batch_size,
            )
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
                color_matrix=None,
            )
            render_seconds = time.perf_counter() - render_started
            output_path = args.output_dir / f"{item.source_id}_{item.path.stem}_learned_isp.png"
            write_started = time.perf_counter()
            if not cv2.imwrite(str(output_path), cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"无法写入 {output_path}")
            write_seconds = time.perf_counter() - write_started
            total_seconds = time.perf_counter() - item_started
            successes.append(
                {
                    "sequence": sequence,
                    "source_id": item.source_id,
                    "input": str(item.path.resolve()),
                    "output": str(output_path.resolve()),
                    "iso": item.iso,
                    "requested_exposure_gain": item.exposure_gain,
                    "applied_exposure_gain": applied_gain,
                    "gain_clamped": applied_gain != item.exposure_gain,
                    "packed_shape": list(frame.packed.shape),
                    "decode_seconds": decode_seconds,
                    "inference_seconds": inference_seconds,
                    "render_seconds": render_seconds,
                    "write_seconds": write_seconds,
                    "total_seconds": total_seconds,
                    "pre_tone_over_white_ratio": over_white,
                }
            )
            print(
                f"完成 {sequence}/{len(items)}: {item.path.name}，"
                f"推理={inference_seconds:.3f}s，总计={total_seconds:.3f}s"
            )
        except Exception as error:
            failure = {
                "sequence": sequence,
                "source_id": item.source_id,
                "input": str(item.path.resolve()),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"失败 {sequence}/{len(items)}: {item.path.name}: {type(error).__name__}: {error}")
            if args.fail_fast:
                raise

    batch_seconds = time.perf_counter() - batch_started
    total_times = [float(item["total_seconds"]) for item in successes]
    inference_times = [float(item["inference_seconds"]) for item in successes]
    summary = {
        "experiment_version": "Phase7-BatchInfer-v0.1.0",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_channels": int(config["base_channels"]),
        "model_load_seconds": model_load_seconds,
        "tile_size": args.tile_size,
        "halo": args.halo,
        "tile_batch_size": args.tile_batch_size,
        "warmup": args.warmup,
        "requested_items": len(items),
        "successful_items": len(successes),
        "failed_items": len(failures),
        "batch_seconds": batch_seconds,
        "throughput_images_per_second": len(successes) / batch_seconds if batch_seconds else None,
        "total_latency_seconds": {
            "mean": float(np.mean(total_times)) if total_times else None,
            "p50": _percentile(total_times, 50),
            "p95": _percentile(total_times, 95),
        },
        "model_inference_seconds": {
            "mean": float(np.mean(inference_times)) if inference_times else None,
            "p50": _percentile(inference_times, 50),
            "p95": _percentile(inference_times, 95),
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "gpu": torch.cuda.get_device_name(device),
        "successes": successes,
        "failures": failures,
    }
    report_path = args.output_dir / "batch_inference_report.json"
    csv_path = args.output_dir / "batch_inference_index.csv"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "序号", "输入", "输出", "ISO", "请求曝光增益", "实际曝光增益",
                "解码秒", "模型推理秒", "渲染秒", "写盘秒", "总耗时秒",
            ]
        )
        for item in successes:
            writer.writerow(
                [
                    item["sequence"], item["input"], item["output"], item["iso"],
                    item["requested_exposure_gain"], item["applied_exposure_gain"],
                    item["decode_seconds"], item["inference_seconds"], item["render_seconds"],
                    item["write_seconds"], item["total_seconds"],
                ]
            )

    print("\n批量推理汇总")
    print(f"成功/失败/总数: {len(successes)}/{len(failures)}/{len(items)}")
    if successes:
        print(
            f"总延迟 mean/p50/p95: {summary['total_latency_seconds']['mean']:.3f} / "
            f"{summary['total_latency_seconds']['p50']:.3f} / "
            f"{summary['total_latency_seconds']['p95']:.3f}s"
        )
        print(
            f"模型推理 mean/p50/p95: {summary['model_inference_seconds']['mean']:.3f} / "
            f"{summary['model_inference_seconds']['p50']:.3f} / "
            f"{summary['model_inference_seconds']['p95']:.3f}s"
        )
        print(f"吞吐率: {summary['throughput_images_per_second']:.4f} images/s")
    print(f"峰值显存: {summary['peak_allocated_gib']:.3f} GiB")
    print(f"JSON 报告: {report_path.resolve()}")
    print(f"中文索引: {csv_path.resolve()}")
    if failures:
        print("批量推理: PARTIAL FAILURE")
        raise SystemExit(2)
    print("批量推理: PASS")


if __name__ == "__main__":
    main()
