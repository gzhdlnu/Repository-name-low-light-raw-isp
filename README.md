# Low-light RAW ISP

基于 SID Sony 官方 RAW 数据构建的部署感知型低照度 RAW-to-sRGB ISP 原型。

## Pipeline

```text
ARW -> 黑电平归一化 -> Bayer packing -> 曝光补偿
    -> RAW U-Net 降噪 -> 白平衡 -> 去马赛克
    -> Reinhard tone mapping -> PNG
```

项目覆盖传统噪声建模与 Wiener 降噪、学习型 RAW 降噪、多维画质评价、参数鲁棒性、CUDA 分块批量推理和 ONNX 数值一致性验证。传统 ISP 前端、颜色重建和 YCbCr 输出的最终补全范围与顺序，以 `docs/工程主线与补全计划.md` 为准。

## 正式配置

```text
模型：RawDenoisingUNet，base_channels=48
权重：checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt
后端：PyTorch CUDA + AMP
推理：tile=512, halo=32, tile-batch-size=6
设备基线：NVIDIA GeForce RTX 3060 Ti 8 GB
```

## Release Smoke Test

SID test 中 3 个 ISO 200 和 3 个 ISO 8000 样本：

```text
成功/失败：6/0
模型推理 mean/p50/p95：0.541 / 0.543 / 0.552 s
端到端延迟 mean/p50/p95：2.622 / 2.466 / 3.189 s
吞吐率：0.3814 images/s
峰值显存：4.273 GiB
```

## 快速检查

```bash
conda activate rawisp
python scripts/check_project_release.py
```

完整阶段结论、指标解释、部署边界和面试表述见：

- `docs/工程主线与补全计划.md`（后续开发的唯一主路线图）
- `docs/Phase1_DPC结论.md`
- `docs/Phase1_LSC结论.md`
- `docs/Phase2_AWB结论.md`
- `docs/Phase2_Demosaic结论.md`
- `docs/项目总览与交付说明.md`
- `docs/面试答辩指南.md`
- `docs/Phase3_结论.md`
- `docs/Phase4_结论.md`
- `docs/Phase5_第一版结论.md`
- `docs/Phase6_结论.md`
- `docs/Phase7_部署基线.md`

## 边界

当前已完成 SID Sony 同相机测试集验证和参数扰动鲁棒性验证，尚未完成真实跨相机 RAW 泛化。当前性能是桌面 GPU 原型基线，不代表相机 SoC、车载 ECU 或量产 ISP 性能。
