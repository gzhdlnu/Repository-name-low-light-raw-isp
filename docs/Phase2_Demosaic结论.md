# Phase 2 Demosaic 结论

**Author:** Guozh  
**Version:** Phase2-Demosaic-Conclusion-v0.1.0  
**Date:** 2026-09-29

## 1. 对比范围

当前比较三个OpenCV工程后端：

```text
bilinear    双线性基线
vng         Variable Number of Gradients
edge_aware  Edge-Aware
```

VNG和Edge-Aware是OpenCV后端实现，不等同于严格复现论文版Malvar-He-Cutler。VNG后端只接受8-bit Bayer，因此封装层存在临时8-bit量化；Bilinear使用16-bit临时输入。

## 2. 合成真值结果

测试图包含高频纹理、斜边、强颜色边界和周期纹理：

| 方法 | PSNR | SSIM | 梯度MAE | 假色MAE | 单次耗时 |
|---|---:|---:|---:|---:|---:|
| Bilinear | 12.585 | 0.68675 | 0.005315 | 0.224056 | 0.00285 s |
| VNG | **12.616** | **0.69182** | **0.004226** | **0.223146** | 0.00455 s |
| Edge-Aware | 12.588 | 0.68861 | 0.005641 | 0.223926 | 0.00780 s |

VNG略好，但领先幅度很小；三者的假色误差接近，不能据此认为VNG已经解决摩尔纹、拉链或伪色问题。

## 3. 真实SID运行时间

pair 0，ISO 200，RGGB，当前工作站平均耗时：

```text
Bilinear    0.23579 s
VNG         0.24409 s
Edge-Aware  0.14829 s
```

三种输出范围均为 `[0, 1]`，没有NaN/Inf。真实SID没有逐像素RGB真值，因此这些耗时不能替代真实画质评价。

## 4. 工程决策

正式pipeline继续使用Bilinear，原因是：

1. 当前合成真值差距小，没有稳定的质量收益证明；
2. Bilinear最简单、最容易解释和复现；
3. VNG存在8-bit临时量化约束；
4. Edge-Aware虽然本次较快，但合成梯度误差并未更优；
5. 当前SID项目的主要困难仍是低照度噪声，不是Demosaic算法本身。

VNG和Edge-Aware保留为实验分支。后续只有在更多高频纹理、斜边和真实局部伪色案例上形成稳定收益，才考虑替换默认后端。

## 5. 阶段状态

```text
三种Demosaic实现：PASS
合成真值对比：PASS
真实SID范围/耗时检查：PASS
默认算法替换：NO
严格MHC论文复现：未完成
```

下一步补Phase 2的线性RGB颜色链路复核：明确WB→Demosaic→CCM→Tone/Gamma的域和顺序，并继续保持CCM默认关闭。
