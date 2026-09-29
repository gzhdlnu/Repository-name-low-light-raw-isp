#!/usr/bin/env python
# =============================================================================
# Project: RAW ISP | Task: Phase 4 Multi-Scene Model Evaluation
# Timestamp: 2026-09-23 | Author: Guozh | Version: Phase5-FailureEval-v0.4.0
# =============================================================================
"""Evaluate learned and traditional RAW denoisers on multiple official SID scenes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import numpy as np
import torch
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_sid_list  # noqa: E402
from data.sid_torch_dataset import sid_scene_id  # noqa: E402
from evaluate_phase4_model import (  # noqa: E402
    METHOD_LABELS_ZH,
    _color_summary,
    _raw_metrics,
    _render,
    _select_rois,
)
from isp import (  # noqa: E402
    gaussian_denoise_packed,
    noise_aware_wiener_packed,
    parse_noise_models,
    render_packed_raw,
    load_normalized_raw,
)
from metrics import compare_display_rgb  # noqa: E402
from models import RawDenoisingUNet, tiled_predict  # noqa: E402


DEFAULT_CHECKPOINTS = (
    (
        "原最佳 U-Net",
        ROOT / "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt",
    ),
    (
        "Detail U-Net",
        ROOT / "checkpoints/phase4_learning/b48_bs12_detail/raw_denoiser_best.pt",
    ),
)


def _parse_indices(value: str) -> list[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair-indices 必须是逗号分隔的整数") from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("pair-indices 不能为空且不能包含负数")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("pair-indices 不能包含重复值")
    return indices


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint 格式应为 中文名称=权重路径")
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label or not path_text.strip():
        raise argparse.ArgumentTypeError("checkpoint 名称和路径均不能为空")
    return label, Path(path_text.strip())


def _load_models(
    checkpoint_specs: list[tuple[str, Path]], device: torch.device
) -> tuple[list[dict[str, object]], float]:
    loaded: list[dict[str, object]] = []
    max_gains: set[float] = set()
    for position, (label, path) in enumerate(checkpoint_specs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        base_channels = int(checkpoint["config"]["base_channels"])
        max_gain = float(checkpoint["config"]["max_exposure_gain"])
        model = RawDenoisingUNet(base_channels=base_channels).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        loaded.append(
            {
                "key": f"learned_{position}",
                "label": label,
                "path": str(path.resolve()),
                "epoch": int(checkpoint["epoch"]),
                "base_channels": base_channels,
                "max_exposure_gain": max_gain,
                "model": model,
            }
        )
        max_gains.add(max_gain)
    if len(max_gains) != 1:
        raise ValueError("待比较 checkpoint 的 max_exposure_gain 不一致，不能公平比较")
    return loaded, max_gains.pop()


def _crop(rgb: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    left, top, width, height = roi
    return rgb[top : top + height, left : left + width]


def _failure_metrics(rgb: np.ndarray, roi: tuple[int, int, int, int]) -> dict[str, float]:
    """Measure dark-region residual noise, chroma fluctuation, and display clipping."""
    full = rgb.astype(np.float32) / 255.0
    crop = _crop(rgb, roi).astype(np.float32) / 255.0
    gray = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    highpass = gray - smooth
    chroma_rg = crop[..., 0] - crop[..., 1]
    chroma_bg = crop[..., 2] - crop[..., 1]
    return {
        "dark_luma_highpass_std": float(np.std(highpass)),
        "dark_chroma_rg_std": float(np.std(chroma_rg)),
        "dark_chroma_bg_std": float(np.std(chroma_bg)),
        "near_black_ratio": float(np.mean(np.max(full, axis=-1) <= 0.01)),
        "near_white_ratio": float(np.mean(np.max(full, axis=-1) >= 0.99)),
    }


def _render_for_eval(
    packed: np.ndarray,
    frame: object,
    tone_map: str,
    white_point: float,
    use_color_matrix: bool,
) -> np.ndarray:
    """Render packed RAW, optionally applying the frame's camera-to-sRGB CCM."""
    return render_packed_raw(
        packed,
        frame.pattern,
        frame.white_balance_gains,
        tone_map=tone_map,
        white_point=white_point,
        color_matrix=frame.color_matrix if use_color_matrix else None,
    )[0]


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _lpips_score(metric: object, prediction: np.ndarray, reference: np.ndarray, device: torch.device) -> float:
    """Compute LPIPS on resized display RGB in the network's [-1, 1] range."""
    import cv2

    height, width = prediction.shape[:2]
    scale = min(1.0, 768.0 / max(height, width))
    target_size = (max(32, int(round(width * scale))), max(32, int(round(height * scale))))
    pred_small = cv2.resize(prediction, target_size, interpolation=cv2.INTER_AREA)
    ref_small = cv2.resize(reference, target_size, interpolation=cv2.INTER_AREA)
    pred_tensor = torch.from_numpy(pred_small.transpose(2, 0, 1)).float().div(127.5).sub(1.0)
    ref_tensor = torch.from_numpy(ref_small.transpose(2, 0, 1)).float().div(127.5).sub(1.0)
    with torch.inference_mode():
        value = metric(pred_tensor.unsqueeze(0).to(device), ref_tensor.unsqueeze(0).to(device))
    return float(value.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="多场景评估 Phase 4 RAW 降噪模型")
    parser.add_argument("--sid-list", type=Path, default=ROOT / "data/raw/SID/Sony_test_list.txt")
    parser.add_argument("--pair-indices", type=_parse_indices, required=True)
    parser.add_argument("--noise-report", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=_parse_checkpoint,
        action="append",
        help="可重复使用，格式为 中文名称=权重路径；省略时比较原最佳与 Detail 模型",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--tone-map", choices=("reinhard", "clip"), default="reinhard")
    parser.add_argument("--white-point", type=float, default=4.0)
    parser.add_argument(
        "--use-color-matrix",
        action="store_true",
        help="在白平衡和去马赛克后应用相机 CCM，再进行 tone mapping",
    )
    parser.add_argument(
        "--with-lpips",
        action="store_true",
        help="启用官方预训练 AlexNet-LPIPS；首次运行可能下载权重，且会增加推理时间",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/phase4_learning/batch_evaluation",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，不能执行学习模型推理")
    pairs = load_sid_list(args.sid_list)
    if max(args.pair_indices) >= len(pairs):
        raise IndexError(f"pair-index 超出范围，test list 共 {len(pairs)} 个 pair")

    selected_pairs = [(index, pairs[index]) for index in args.pair_indices]
    selected_isos = {pair.iso for _, pair in selected_pairs}
    selected_gains = {round(pair.exposure_gain, 6) for _, pair in selected_pairs}
    if len(selected_isos) != 1 or len(selected_gains) != 1:
        raise ValueError("一次批量评估中的 pair 必须具有相同 ISO 和曝光增益")

    noise_report = json.loads(args.noise_report.read_text(encoding="utf-8"))
    expected_iso = next(iter(selected_isos))
    expected_gain = next(iter(selected_gains))
    if int(noise_report["iso"]) != expected_iso:
        raise ValueError(f"噪声模型 ISO={noise_report['iso']}，测试数据 ISO={expected_iso}")
    calibration_gains = {
        round(float(profile["exposure_gain"]), 6) for profile in noise_report["pair_profiles"]
    }
    if calibration_gains != {expected_gain}:
        raise ValueError("噪声模型曝光增益与测试 pair 不一致")
    slopes, intercepts = parse_noise_models(noise_report["channels"])

    device = torch.device("cuda:0")
    checkpoint_specs = args.checkpoint or list(DEFAULT_CHECKPOINTS)
    labels = [label for label, _ in checkpoint_specs]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint 中文名称不能重复")
    learned_models, max_gain = _load_models(checkpoint_specs, device)
    lpips_metric = None
    if args.with_lpips:
        try:
            import lpips  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "未安装 LPIPS，请先运行: python -m pip install lpips==0.1.4"
            ) from error
        lpips_metric = lpips.LPIPS(net="alex", pnet_rand=False, verbose=False).to(device).eval()
    method_labels = {
        "no_denoise": METHOD_LABELS_ZH["no_denoise"],
        "fixed_gaussian": METHOD_LABELS_ZH["fixed_gaussian"],
        "noise_aware_wiener": METHOD_LABELS_ZH["noise_aware_wiener"],
        **{str(item["key"]): str(item["label"]) for item in learned_models},
    }

    scene_results: list[dict[str, object]] = []
    total_inference_seconds: defaultdict[str, float] = defaultdict(float)
    torch.cuda.reset_peak_memory_stats(device)
    for sequence, (pair_index, pair) in enumerate(selected_pairs, start=1):
        short = load_normalized_raw(pair.short_path)
        reference_frame = load_normalized_raw(pair.long_path)
        if short.packed.shape != reference_frame.packed.shape:
            raise ValueError(f"pair {pair_index} 的短长曝光 RAW shape 不一致")
        applied_gain = min(pair.exposure_gain, max_gain)
        exposure_corrected = np.clip(short.packed * applied_gain, 0.0, 1.0).astype(np.float32)
        input_tensor = torch.from_numpy(exposure_corrected.transpose(2, 0, 1)).unsqueeze(0).to(device)

        packed_candidates = {
            "no_denoise": exposure_corrected,
            "fixed_gaussian": gaussian_denoise_packed(exposure_corrected, sigma=1.0),
            "noise_aware_wiener": noise_aware_wiener_packed(
                exposure_corrected, slopes, intercepts, window_size=7, strength=1.0
            ),
        }
        for model_info in learned_models:
            start = time.perf_counter()
            prediction = tiled_predict(
                model_info["model"],
                input_tensor,
                tile_size=args.tile_size,
                halo=args.halo,
                use_amp=True,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            key = str(model_info["key"])
            total_inference_seconds[key] += elapsed
            packed_candidates[key] = np.clip(
                prediction.squeeze(0).permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0
            )

        reference_rgb = _render_for_eval(
            reference_frame.packed, reference_frame, args.tone_map, args.white_point,
            args.use_color_matrix,
        )
        rois = _select_rois(reference_rgb)
        scene_methods: dict[str, object] = {}
        for key, packed in packed_candidates.items():
            rgb = _render_for_eval(
                packed, short, args.tone_map, args.white_point, args.use_color_matrix
            )
            roi_metrics = {
                roi_name: compare_display_rgb(_crop(rgb, roi), _crop(reference_rgb, roi))
                for roi_name, roi in rois.items()
            }
            scene_method = {
                "display_name_zh": method_labels[key],
                "raw_metrics": _raw_metrics(packed, reference_frame.packed),
                "display_metrics": compare_display_rgb(rgb, reference_rgb),
                "color_summary": _color_summary(rgb, reference_rgb),
                "roi_metrics": roi_metrics,
                "failure_metrics": _failure_metrics(rgb, rois["暗部噪声区域"]),
            }
            if lpips_metric is not None:
                scene_method["lpips"] = _lpips_score(lpips_metric, rgb, reference_rgb, device)
            scene_methods[key] = scene_method
        reference_color = _color_summary(reference_rgb, reference_rgb)
        reference_failure = _failure_metrics(reference_rgb, rois["暗部噪声区域"])
        scene_results.append(
            {
                "pair_index": pair_index,
                "scene_id": sid_scene_id(pair.short_path),
                "iso": pair.iso,
                "physical_exposure_gain": pair.exposure_gain,
                "applied_exposure_gain": applied_gain,
                "reference_color": {
                    "mean_hsv_saturation": reference_color["mean_hsv_saturation"],
                    "mean_lab_chroma_cstar": reference_color["mean_lab_chroma_cstar"],
                },
                "reference_failure_metrics": reference_failure,
                "analysis_rois": {
                    name: {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
                    for name, roi in rois.items()
                },
                "methods": scene_methods,
            }
        )
        print(f"完成 {sequence}/{len(selected_pairs)}: pair {pair_index}（场景 {sid_scene_id(pair.short_path)}）")

    summary: dict[str, object] = {}
    for key, label in method_labels.items():
        results = [scene["methods"][key] for scene in scene_results]
        result = {
            "display_name_zh": label,
            "mean_raw_psnr_db": _mean([item["raw_metrics"]["psnr_db"] for item in results]),
            "mean_raw_mae": _mean([item["raw_metrics"]["mae"] for item in results]),
            "mean_display_psnr_db": _mean(
                [item["display_metrics"]["psnr_db"] for item in results]
            ),
            "mean_ssim": _mean([item["display_metrics"]["ssim"] for item in results]),
            "mean_delta_e2000": _mean(
                [item["display_metrics"]["delta_e2000_mean"] for item in results]
            ),
            "mean_gradient_similarity": _mean(
                [item["display_metrics"]["gradient_similarity"] for item in results]
            ),
            "mean_hsv_saturation": _mean(
                [item["color_summary"]["mean_hsv_saturation"] for item in results]
            ),
            "mean_lab_chroma_cstar": _mean(
                [item["color_summary"]["mean_lab_chroma_cstar"] for item in results]
            ),
            "mean_chroma_abs_error": _mean(
                [item["color_summary"]["mean_lab_chroma_abs_error"] for item in results]
            ),
            "mean_dark_roi_ssim": _mean(
                [item["roi_metrics"]["暗部噪声区域"]["ssim"] for item in results]
            ),
            "mean_detail_roi_ssim": _mean(
                [item["roi_metrics"]["高频细节区域"]["ssim"] for item in results]
            ),
            "mean_detail_roi_gradient_similarity": _mean(
                [
                    item["roi_metrics"]["高频细节区域"]["gradient_similarity"]
                    for item in results
                ]
            ),
            "mean_dark_luma_highpass_std": _mean(
                [item["failure_metrics"]["dark_luma_highpass_std"] for item in results]
            ),
            "mean_dark_chroma_rg_std": _mean(
                [item["failure_metrics"]["dark_chroma_rg_std"] for item in results]
            ),
            "mean_dark_chroma_bg_std": _mean(
                [item["failure_metrics"]["dark_chroma_bg_std"] for item in results]
            ),
            "mean_near_black_ratio": _mean(
                [item["failure_metrics"]["near_black_ratio"] for item in results]
            ),
            "mean_near_white_ratio": _mean(
                [item["failure_metrics"]["near_white_ratio"] for item in results]
            ),
        }
        if lpips_metric is not None:
            result["mean_lpips"] = _mean([item["lpips"] for item in results])
        summary[key] = result

    reference_summary = {
        "mean_hsv_saturation": _mean(
            [scene["reference_color"]["mean_hsv_saturation"] for scene in scene_results]
        ),
        "mean_lab_chroma_cstar": _mean(
            [scene["reference_color"]["mean_lab_chroma_cstar"] for scene in scene_results]
        ),
    }
    reference_failure_summary = {
        key: _mean([scene["reference_failure_metrics"][key] for scene in scene_results])
        for key in (
            "dark_luma_highpass_std",
            "dark_chroma_rg_std",
            "dark_chroma_bg_std",
            "near_black_ratio",
            "near_white_ratio",
        )
    }
    report = {
        "experiment_version": "Phase5-FailureEval-v0.4.0",
        "description_zh": "官方 SID test 多场景学习型与传统 RAW 降噪统一评估",
        "sid_list": str(args.sid_list.resolve()),
        "pair_indices": args.pair_indices,
        "iso": expected_iso,
        "physical_exposure_gain": next(iter(selected_gains)),
        "noise_report": str(args.noise_report.resolve()),
        "checkpoints": [
            {key: value for key, value in item.items() if key != "model"} for item in learned_models
        ],
        "tile_size": args.tile_size,
        "halo": args.halo,
        "with_lpips": args.with_lpips,
        "color_matrix_applied": args.use_color_matrix,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "total_inference_seconds": dict(total_inference_seconds),
        "reference_color_mean": reference_summary,
        "reference_failure_mean": reference_failure_summary,
        "summary": summary,
        "scenes": scene_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_lpips" if args.with_lpips else ""
    suffix += "_ccm" if args.use_color_matrix else ""
    stem = f"iso_{expected_iso:05d}_{len(scene_results)}scenes_model_comparison{suffix}"
    report_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        header = [
            "方法", "RAW PSNR(dB)", "RAW MAE", "显示 PSNR(dB)", "SSIM",
            "DeltaE2000", "全图梯度相似度", "HSV饱和度", "Lab C*", "色度误差",
            "暗部ROI SSIM", "细节ROI SSIM", "细节ROI梯度相似度",
            "暗部亮度高频STD", "暗部R-G STD", "暗部B-G STD", "近黑比例", "近白比例",
        ]
        if lpips_metric is not None:
            header.append("LPIPS（越低越好）")
        writer.writerow(header)
        for result in summary.values():
            row = [
                    result["display_name_zh"], result["mean_raw_psnr_db"], result["mean_raw_mae"],
                    result["mean_display_psnr_db"], result["mean_ssim"], result["mean_delta_e2000"],
                    result["mean_gradient_similarity"], result["mean_hsv_saturation"],
                    result["mean_lab_chroma_cstar"], result["mean_chroma_abs_error"],
                    result["mean_dark_roi_ssim"], result["mean_detail_roi_ssim"],
                    result["mean_detail_roi_gradient_similarity"],
                    result["mean_dark_luma_highpass_std"], result["mean_dark_chroma_rg_std"],
                    result["mean_dark_chroma_bg_std"], result["mean_near_black_ratio"],
                    result["mean_near_white_ratio"],
                ]
            if lpips_metric is not None:
                row.append(result["mean_lpips"])
            writer.writerow(row)

    print("\n多场景均值（所有指标均相对真实长曝光参考）")
    print("方法                 显示PSNR    SSIM   DeltaE  全图梯度  细节ROI梯度  HSV饱和度")
    for result in summary.values():
        print(
            f"{result['display_name_zh']:<18} {result['mean_display_psnr_db']:>8.3f}"
            f"  {result['mean_ssim']:>7.5f}  {result['mean_delta_e2000']:>7.3f}"
            f"  {result['mean_gradient_similarity']:>8.5f}"
            f"  {result['mean_detail_roi_gradient_similarity']:>11.5f}"
            f"  {result['mean_hsv_saturation']:>9.4f}"
        )
    if lpips_metric is not None:
        print("\nLPIPS（越低越好）")
        for result in summary.values():
            print(f"{result['display_name_zh']:<18} {result['mean_lpips']:.6f}")
    print("\n失败案例指标（暗部 ROI；噪声和彩噪越低越好）")
    print("方法                 亮度高频STD   R-G STD   B-G STD   近黑比例   近白比例")
    for result in summary.values():
        print(
            f"{result['display_name_zh']:<18} {result['mean_dark_luma_highpass_std']:>11.6f}"
            f"  {result['mean_dark_chroma_rg_std']:>8.6f}  {result['mean_dark_chroma_bg_std']:>8.6f}"
            f"  {result['mean_near_black_ratio']:>9.6f}  {result['mean_near_white_ratio']:>9.6f}"
        )
    print(
        f"{'真实长曝光参考':<18} {reference_failure_summary['dark_luma_highpass_std']:>11.6f}"
        f"  {reference_failure_summary['dark_chroma_rg_std']:>8.6f}"
        f"  {reference_failure_summary['dark_chroma_bg_std']:>8.6f}"
        f"  {reference_failure_summary['near_black_ratio']:>9.6f}"
        f"  {reference_failure_summary['near_white_ratio']:>9.6f}"
    )
    print(
        f"长曝光参考颜色均值: HSV饱和度={reference_summary['mean_hsv_saturation']:.4f}, "
        f"Lab C*={reference_summary['mean_lab_chroma_cstar']:.3f}"
    )
    print(f"JSON 报告: {report_path}")
    print(f"中文 CSV 汇总: {csv_path}")


if __name__ == "__main__":
    main()
