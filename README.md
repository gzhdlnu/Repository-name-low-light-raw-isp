# Low-light RAW ISP / 低照度 RAW ISP

基于 SID Sony 官方 RAW 数据构建的部署感知型低照度 RAW-to-sRGB ISP 原型。
这是一个面向工程实践的 workstation prototype，不宣称是量产相机 ISP。

This is a deployment-aware low-light RAW-to-sRGB ISP prototype built on the official
SID Sony RAW dataset. It is an engineering workstation prototype, not a production
camera ISP implementation.

## Pipeline

```text
ARW -> 黑电平归一化 -> Bayer packing -> 曝光补偿
    -> RAW U-Net 降噪 -> 白平衡 -> 去马赛克
    -> Reinhard tone mapping -> PNG
```

中文说明：输入是相机 RAW/Bayer 数据，先进行黑电平归一化和曝光补偿，再在 RAW 域降噪；随后完成白平衡、去马赛克、线性 RGB 处理、Tone Mapping 和 sRGB 输出。

English: The pipeline normalizes sensor RAW/Bayer data, applies exposure compensation
and RAW-domain denoising, then performs white balance, demosaicing, linear RGB
processing, tone mapping, and sRGB output.

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

The release smoke test validates six SID test images on an RTX 3060 Ti workstation.
它验证的是工作站部署原型，不代表相机 SoC、车载 ECU 或量产 ISP 的实时性能。

## 快速检查

```bash
conda activate rawisp
python scripts/check_project_release.py
```

The command checks required source files and release materials. It does not download
the dataset, train the model, or claim cross-camera generalization.

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

## Dataset / 数据集

SID Sony RAW files are not included in this repository. Download the dataset from
the official source, then place the extracted files under `data/raw/SID/` according
to `docs/数据集说明.md`. Do not commit RAW archives, private credentials, or local
experiment dumps.

本仓库不包含 SID Sony 原始 RAW。请根据 `docs/数据集说明.md` 下载并解压到
`data/raw/SID/`。不要提交 RAW 压缩包、个人凭据或本地实验输出。

## Project status / 项目状态

- Phase 1–5: RAW frontend, noise modeling, learned denoising, and quality evaluation completed.
- Phase 6: SID internal robustness completed; real cross-camera validation remains open.
- Phase 7: CUDA tiled inference, ONNX numerical export, BT.709 YCbCr and NV12 output smoke tests completed.

- Phase 1–5：RAW 前端、噪声建模、学习型降噪和画质评价已完成。
- Phase 6：SID 内部参数鲁棒性已完成，真实跨相机验证仍未完成。
- Phase 7：CUDA 分块推理、ONNX 数值导出、BT.709 YCbCr 和 NV12 输出验收已完成。
