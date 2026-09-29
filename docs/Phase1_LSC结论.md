# Phase 1 LSC 结论

**Author:** Guozh  
**Version:** Phase1-LSC-Conclusion-v0.1.0  
**Date:** 2026-09-28

## 验收结果

合成四通道径向暗角和粗粒度 gain map 验收通过：

| 指标 | 校正前 | 校正后 |
|---|---:|---:|
| 角落/中心均值比 | 0.61176 | 1.00489 |
| 空间 CV | 0.12592 | 0.00702 |
| 无噪声校正 MAE | - | 0.00071514 |

增益范围为 `1.0000–2.2727`。四通道角落噪声放大倍数为：R `1.7202x`、G1 `1.5436x`、G2 `1.5697x`、B `1.8112x`。

真实SID pair 0 使用 identity gain map 时最大绝对误差为 `0`，说明默认旁路不会改变已有RAW输入。

## 工程结论

LSC通过乘法gain map补偿镜头和传感器的空间响应不均匀，中心/角落亮度一致性明显改善。但角落gain同时会放大角落噪声；原始噪声方差为 `sigma^2` 时，乘以增益 `g` 后方差约为 `g^2 * sigma^2`。

当前没有Sony目标镜头的暗场/平场标定RAW，不能把合成径向gain map当作真实标定结果。正式配置保持：

```text
真实SID：LSC identity / bypass
合成实验：允许使用gain map验证接口和噪声代价
真实产品标定：等待目标相机平场数据
```

## 阶段状态

```text
LSC接口与插值：PASS
均匀性校正：PASS
噪声放大记录：PASS
真实镜头标定：未完成
正式U-Net输入改变：NO
```

下一步进入Phase 2 AWB估计比较，先把camera metadata white balance与统计AWB算法区分开。
