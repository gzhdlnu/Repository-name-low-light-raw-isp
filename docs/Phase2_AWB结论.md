# Phase 2 AWB 结论

**Author:** Guozh  
**Version:** Phase2-AWB-Conclusion-v0.1.0  
**Date:** 2026-09-29

## 1. 实现与输入定义

实现了三种统计AWB估计器：

- Gray World：有效亮度范围内的通道均值平衡；
- Shades-of-Gray：Minkowski `p=6` 通道统计；
- Gray Edge：通道梯度能量平衡。

AWB估计在黑电平归一化、物理曝光补偿后的线性packed RAW上进行；估计出来的是补偿增益，不是通道响应比。R/G1/G2/B四通道最后以G1/G2平均值作为unity参考。

## 2. 合成验收

对人工注入的R偏亮1.42倍、B偏暗0.72倍色偏：

| 方法 | 反向增益MAE |
|---|---:|
| Gray World | 0.000006 |
| Shades-of-Gray | 0.000003 |
| Gray Edge | 0.049139 |

说明三种实现方向正确；Gray Edge受纹理分布影响更明显。

## 3. SID真实场景审计

测试pair：`0,14,74,86,98,123,542,556,570`，覆盖ISO 200和ISO 8000。统计AWB与rawpy camera metadata WB的平均绝对差：

```text
Gray World       0.437940，最大 1.192384
Shades-of-Gray   0.459642，最大 0.681484
Gray Edge        0.424485，最大 0.593687
```

最大异常来自极暗/非中性场景，例如pair 86的Gray World B gain达到`6.86`，明显不适合作为无约束的自动增益。

## 4. 工程解释

统计AWB隐含假设：场景中不同颜色的平均反射接近中性，或图像中包含足够的灰/白参考。夜间球场、草地、墙面、标志牌、大片单色物体和多光源都会破坏该假设。ISO 8000的短曝光还会让通道均值受噪声和近黑裁剪影响。

因此“与metadata差异大”不是metadata绝对正确的证明，但足以说明统计AWB不能在当前SID夜景上无条件替换metadata WB。

## 5. 正式决策

```text
正式pipeline WB：使用camera metadata，作为稳定基线
Gray World/Shades-of-Gray/Gray Edge：保留实验分支
统计AWB默认启用：NO
统计AWB使用前提：有效像素足够、增益范围受限、场景置信度通过
```

工业AWB后续应加入候选区域筛选、饱和/近黑排除、色温先验、时序平滑和置信度回退；不能只靠单帧全图平均。

## 6. 阶段状态

```text
AWB实现：PASS
合成色偏回收：PASS
SID统计稳定性审计：完成
统计AWB替换metadata：FAIL/不启用
```

下一步进入Demosaic，对Bilinear、Malvar-He-Cutler和Edge-aware方法建立斜边、周期纹理、伪色和运行时间对比。
