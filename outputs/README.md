# Output Layout

实验输出按照项目阶段隔离：

```text
phase1_raw_baseline/       RAW 读取和传统 ISP 基线
phase2_color_tone/         曝光、白平衡、颜色和 tone mapping
phase3_noise/              噪声建模和传统降噪
phase4_learning/           学习型 RAW 降噪
phase5_evaluation/         指标、对比图和消融实验
phase6_generalization/     跨数据集泛化结果
phase7_deployment/         ONNX/TensorRT 性能结果
```

每个阶段的临时探索结果放入自己的 `debug/` 子目录。正式报告只引用阶段目录中的正式结果。
