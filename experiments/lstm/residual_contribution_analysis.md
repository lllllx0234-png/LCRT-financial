# Residual auxiliary LCT-Riesz contribution 诊断

## 1. 诊断对象

本诊断针对时间戳为 `20260625_163554` 的 residual auxiliary LCT-Riesz LSTM 正式实验。配置来自 `experiments/lstm/outputs/volatility_5/residual_lct_signal_features/20260625_163554/config.json`，checkpoint 为 `experiments/lstm/checkpoints/volatility_5/residual_lct_signal_features/20260625_163554/best_model.pth`。测试区间为 2024-06-04 至 2026-05-31，共 727 个样本；checkpoint 中的 `residual_scale` 为 0.01110094878822565。

历史结果复核以 `experiments/lstm/outputs/volatility_5/residual_lct_signal_features/20260625_163554/metrics.json` 和 `experiments/lstm/outputs/volatility_5/residual_lct_signal_features/20260625_163554/prediction_results.csv` 为准。

诊断直接载入已有 checkpoint，以评估模式和无梯度推理拆分预测，没有重新训练模型，也没有更新任何权重。

## 2. 数学分解与解释边界

```text
residual_correction = residual_scale × spectral_delta
final_pred = main_pred + residual_correction
```

本文的 **main-only** 是 residual 模型联合训练后的内部 main branch，不是独立训练的 Plain LSTM。因此，Full 与 main-only 的差异只说明联合模型中的 correction 改变了预测，不能据此声称 LCT-Riesz 优于独立 Plain LSTM。

## 3. Full、main-only 与常数对照

| 设置 | MAE | MSE | RMSE | R² |
| -- | --: | --: | --: | --: |
| Full | 0.008156298901 | 0.000113881330 | 0.010671519574 | 0.019591018311 |
| Main-only（联合训练内部主分支） | 0.010333114614 | 0.000154756065 | 0.012440099056 | -0.332301226428 |
| Test correction mean | 0.008170398448 | 0.000113065448 | 0.010633223805 | 0.026614975320 |
| Validation correction mean | 0.008298936066 | 0.000115031309 | 0.010725264993 | 0.009690801151 |
| Validation MSE bias | 0.008814758776 | 0.000123796019 | 0.011126365961 | -0.065764944923 |
| Validation MAE bias | 0.008051609707 | 0.000111332504 | 0.010551421885 | 0.041533966423 |

Test correction mean（-0.004071498680）和 validation correction mean（-0.003767744472）均只来自模型频域分支的 correction，不使用标签。Validation MSE bias（-0.002656448835）和 validation MAE bias（-0.004366565496）使用 validation target 进行后处理校准；没有使用 test target 拟合偏置。

Full 的四项指标与原正式实验完全一致：指标绝对差均为 0，target 与 prediction 的最大绝对差分别为 `9.8879238130678e-17` 和 `9.71445146547012e-17`。

## 4. 置换结果

固定总随机种子 42，诊断包含 1,000 次完全随机置换和全部 726 个非零循环位移。误差指标越低越好，R² 越高越好；“Full 优于比例”表示 Full 胜过相应置换的比例，“Full CDF 百分位”表示置换值小于等于 Full 的比例。

| 置换 | 指标 | 置换均值 | Full 优于比例 | Full CDF 百分位 | 经验 p 值 |
| -- | -- | --: | --: | --: | --: |
| Random（1,000 次） | MAE | 0.008184113803 | 90.40% | 9.60% | 0.096903096903 |
| Random（1,000 次） | MSE | 0.000113523845 | 24.10% | 75.90% | 0.759240759241 |
| Random（1,000 次） | RMSE | 0.010654729911 | 24.10% | 75.90% | 0.759240759241 |
| Random（1,000 次） | R² | 0.022668621459 | 24.10% | 24.10% | 0.759240759241 |
| Circular shift（726 次） | MAE | 0.008185622382 | 64.19% | 35.81% | 0.359009628611 |
| Circular shift（726 次） | MSE | 0.000113533619 | 45.59% | 54.41% | 0.544704264099 |
| Circular shift（726 次） | RMSE | 0.010655002512 | 45.59% | 54.41% | 0.544704264099 |
| Circular shift（726 次） | R² | 0.022584473570 | 45.59% | 45.59% | 0.544704264099 |

经验 p 值采用加一修正：MAE、MSE、RMSE 统计置换结果小于等于 Full 的次数，R² 统计置换结果大于等于 Full 的次数。由于 `volatility_5` 相邻目标窗口重叠，普通置换的样本交换性假设不成立；循环位移更能保留 correction 自身的时序结构，但两类结果都只能作为稳健性参考，不能解释为严格统计显著性。

## 5. Correction 的实际性质

727 个 correction 全为负值，范围为 -0.005166215356 至 -0.002059821272；均值为 -0.004071498680，标准差为 0.000684579525，绝对均值为 0.004071498680。去均值后的标准差仍为 0.000684579525，去均值 correction 与 `target - main_pred` 的相关系数仅为 -0.024922039415。

按单样本绝对误差计算，Full 改善 559 个样本（76.89%），恶化 168 个样本（23.11%）。分区结果如下：

| 波动状态 | 样本数 | Main MAE | Full MAE | Main RMSE | Full RMSE | 改善比例 | 去均值 correction 标准差 |
| -- | --: | --: | --: | --: | --: | --: | --: |
| Low | 182 | 0.016698839667 | 0.012560721704 | 0.016895490636 | 0.012835671308 | 100.00% | 0.000641905806 |
| Normal | 363 | 0.008832259512 | 0.005057369774 | 0.009571570164 | 0.006060177956 | 97.80% | 0.000674494216 |
| High | 182 | 0.006960853310 | 0.009932707269 | 0.012247032879 | 0.014727369646 | 12.09% | 0.000739163301 |

负向 correction 在低、中波动区间主要起整体下调作用，但在高波动区间明显恶化误差。较小的 `residual_scale` 只是乘法系数，不能脱离 `spectral_delta` 的尺度推断实际 correction 很小。

## 6. 最终结论

> Full 明显优于联合训练模型内部的 main branch，但没有明确优于常数修正或置换分布。当前改善主要来自整体负向校准，样本级频域信息证据不足。

本诊断属于结论分类 B。尤其是 test correction mean 在 MSE、RMSE 和 R² 上优于 Full，validation MAE bias 在四项指标上也优于 Full；同时，Full 并未处于置换分布的一致优异尾部。因此，当前结果不能证明 LCT-Riesz 优于独立 Plain LSTM，不能证明频域分支学到了稳定的样本级修正，也不能把置换结果作为严格统计显著性证据。

## 7. 后续决策

- 暂不对当前 residual 结构直接开展大规模多 seed 实验。
- 暂不将当前 residual 结果作为 LCT-Riesz 有效性的正面论文结论。
- 下一阶段应设计能够排除常数偏置作用的公平控制模型，或修改融合机制。
- 新结构应重点验证高波动区间、样本级 correction 对齐，以及相对独立 Plain LSTM 的真实增益。
