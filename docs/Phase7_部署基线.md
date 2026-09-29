# Phase 7 部署基线：PyTorch CUDA 与 ONNX

## 1. 当前完整推理入口

入口脚本：

`scripts/run_learned_raw_isp.py`

默认 pipeline：

```text
ARW
→ 每通道黑电平归一化
→ R/G1/G2/B packed RAW
→ 物理曝光补偿
→ 残差 U-Net 分块降噪
→ 白平衡
→ 双线性去马赛克
→ Reinhard tone mapping
→ PNG
```

默认模型：

`checkpoints/phase4_learning/b48_bs12_fair/raw_denoiser_best.pt`

默认 CCM 关闭，tile=512，halo=32，CUDA AMP 开启。

## 2. 单图 GPU 基线

测试输入：SID Sony test pair 0，ISO 200，12MP ARW。

| 项目 | 结果 |
|---|---:|
| U-Net 推理 | 0.914 s |
| RAW 解码 | 0.196 s |
| 模型加载 | 0.155 s |
| 白平衡/去马赛克/tone mapping | 2.029 s |
| 总耗时 | 5.185 s |
| 峰值显存 | 0.897 GiB |
| 高光溢出 | 0% |

当前瓶颈不是 U-Net，而是 CPU 端显示域渲染和文件读写。后续优化应优先分析 demosaic、tone mapping 和 PNG 编码。

## 3. ONNX 导出验收

导出脚本：

`scripts/export_phase7_onnx.py`

固定输入 `1×4×256×256` 的第一次验证结果：

- 最大绝对误差：`7.7486e-06`
- 平均绝对误差：`3.4409e-07`
- ONNX checker：通过

ONNX 文件：

`outputs/phase7_deployment/onnx/raw_denoiser_b48.onnx`

导出模型使用动态 batch、height、width 轴。导出 warning 来自旧版 TorchScript exporter 和模型内的 shape 检查，不影响当前数值验收；动态尺寸仍需单独测试。

## 4. CPU 后端基准

输入：`1×4×512×512`，预热 5 次，正式测量 20 次。

| 后端 | mean | P50 | P95 |
|---|---:|---:|---:|
| PyTorch CPU | 0.856892 s | 0.844371 s | 0.929313 s |
| ONNX Runtime CPU | 1.072732 s | 1.038634 s | 1.194332 s |

ONNX Runtime 平均加速比为 `0.799x`，即比 PyTorch CPU 慢约 25%。动态尺寸误差：

- 最大绝对误差：`8.3864e-05`
- 平均绝对误差：`3.0324e-06`

误差仍低于当前 `1e-4` 验收阈值，但已经接近阈值。正式部署前不能只看固定尺寸误差。

## 5. 后端决策

当前正式后端固定为：

```text
PyTorch CUDA + AMP + tiled inference
```

ONNX 保留用于跨平台验证和后续部署探索，不作为当前 CPU 加速后端。当前环境没有 TensorRT、`trtexec` 或 `torch_tensorrt`，暂不引入 GPU Runtime 依赖，避免 CUDA/cuDNN 版本冲突。

## 6. 后续优化方向

1. 单独优化 demosaic、tone mapping 与 PNG 编码；
2. 在确认部署目标后再评估 TensorRT FP16；
3. 保留 ONNX 数值一致性报告，作为后端替换的回归测试；
4. 在目标硬件上重新建立延迟、功耗和显存基线。

## 7. 批量推理基线

在 6 个 ISO 200 官方 test pair（`0,14,74,86,98,123`）上，模型只加载一次，使用 CUDA AMP 和 `tile-batch-size=6`：

| 项目 | 结果 |
|---|---:|
| 成功/失败 | 6/0 |
| 总延迟 mean/p50/p95 | 2.473 / 2.472 / 2.513 s |
| U-Net 推理 mean/p50/p95 | 0.562 / 0.557 / 0.606 s |
| 吞吐率 | 0.4043 images/s |
| 峰值显存 | 4.273 GiB |

相同输入下，`tile-batch-size=6` 相比串行 tile（batch=1）将模型推理均值从约 `0.745s` 降至 `0.562s`，端到端吞吐从约 `0.331` 提升至 `0.404 images/s`。6 个场景均无失败文件，P95 与均值接近，说明当前工作站基线稳定。

当前部署配置固定为：

```text
tile=512, halo=32, tile-batch-size=6, CUDA AMP, PyTorch CUDA
```

这是一台 RTX 3060 Ti 工作站上的单图 RAW-to-PNG 基线，不等同于嵌入式 ISP 量产吞吐。后续若面向视频或相机 SoC，需要在目标硬件上重新测量。

## 8. 最终 Release Smoke Test

最终交付测试同时覆盖 3 个 ISO 200 和 3 个 ISO 8000 官方 test pair：

| 项目 | 结果 |
|---|---:|
| 成功/失败 | 6/0 |
| 总延迟 mean/p50/p95 | 2.622 / 2.466 / 3.189 s |
| U-Net 推理 mean/p50/p95 | 0.541 / 0.543 / 0.552 s |
| 吞吐率 | 0.3814 images/s |
| 峰值显存 | 4.273 GiB |

报告：`outputs/phase7_deployment/release_smoke/batch_inference_report.json`

ISO 8000 样本的物理曝光比为 `303.03x`，部署入口受 `max_gain=300` 安全上限约束，实际应用 `300x`，报告中记录为 `gain_clamped=true`。该保护避免异常元数据导致数值溢出；本次差异约 1%，但在严格画质复现时必须同时记录请求增益和实际增益。
