"""
Project: Low-light RAW ISP
Task: Static release readiness check
Timestamp: 2026-09-24
Author: Guozh
Version: Release-Check-v0.8.0
"""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = (
    "docs/项目总览与交付说明.md",
    "docs/工程主线与补全计划.md",
    "docs/Phase1_DPC结论.md",
    "docs/Phase1_LSC结论.md",
    "docs/Phase2_AWB结论.md",
    "docs/Phase2_Demosaic结论.md",
    "docs/面试答辩指南.md",
    "docs/Phase2_结论.md",
    "docs/Phase3_结论.md",
    "docs/Phase4_结论.md",
    "docs/Phase5_第一版结论.md",
    "docs/Phase6_结论.md",
    "docs/Phase7_部署基线.md",
    "scripts/run_learned_raw_isp.py",
    "scripts/run_learned_raw_isp_batch.py",
    "scripts/evaluate_phase6_robustness.py",
    "scripts/export_phase7_onnx.py",
    "checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt",
    "outputs/phase7_deployment/onnx/raw_denoiser_b48.onnx",
    "outputs/phase7_deployment/release_smoke/batch_inference_report.json",
    "outputs/phase7_deployment/release_smoke/batch_inference_index.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 RAW ISP 项目是否具备交付材料")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.project_root.resolve()
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    print("RAW ISP 项目发布前静态检查")
    print(f"项目目录: {root}")
    print(f"必需文件: {len(REQUIRED_FILES)}")
    print(f"缺失文件: {len(missing)}")
    if missing:
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)
    print("检查结果: PASS")
    print("说明: 这是材料和路径检查，不代替训练、画质评估或真实设备性能测试。")


if __name__ == "__main__":
    main()
