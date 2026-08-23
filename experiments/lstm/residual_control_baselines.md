# Residual auxiliary 控制实验记录

## 1. 控制目的

No-LCT signal auxiliary residual 是 existing residual LCT-Riesz 的机制控制。它保留相同的 main branch、4 个 signal features 的辅助输入、时间均值池化、residual 联合训练和可学习 residual scale，仅将 LCT-Riesz 模块替换为 Identity。该控制用于检验以下因素是否足以解释原 residual LCT-Riesz 的结果：

- 增加辅助分支本身；
- signal features 在辅助分支中的重复输入；
- 时间均值池化；
- main branch 与 auxiliary branch 的联合训练；
- correction 带来的整体偏置校准。

本次是固定 seed 42 的单次机制筛查，不是调参实验，也不构成统计等效性或显著性检验。

Parameter-matched temporal auxiliary residual 是第二个机制控制。它以一个共享的严格因果 `Conv1d` 替换 LCT-Riesz block：卷积核大小为 56、无 bias、仅左侧 padding 55，并在 4 个 signal channels 之间共享，因此恰好包含 56 个可训练参数。该控制用于排除普通时域建模能力和额外参数量对 Residual LCT-Riesz 结果的解释。

## 2. 公平条件

| 条件 | 统一设置 |
| -- | -- |
| 数据 | `data/processed/BTC_daily_train.csv` |
| 任务 | `volatility_5` |
| 输入 | OHLCV + 4 个 signal features，共 9 维 |
| Signal indices | `[5, 6, 7, 8]` |
| 时间划分 | train/validation/test = 0.70/0.15/0.15 |
| Sequence length | 60 |
| Seed | 42 |
| Epoch 上限 | 20 |
| Batch size | 32 |
| Learning rate | 0.001 |
| 优化器与损失 | AdamW，MSE |

在固定 seed 下，Residual LCT-Riesz、No-LCT 和 Temporal 的 main branch 参数名称、形状和初始数值逐元素一致。Temporal 与 Residual LCT-Riesz 的 auxiliary projection、delta head 和 residual scale 初始值也逐元素一致；两者总参数量均为 59,067，唯一区别是 56 参数 LCT-Riesz block 与 56 参数共享因果 Conv1d。No-LCT 参数量为 59,011，比 Residual LCT-Riesz 少 56 个参数。No-LCT 和 Temporal 的 `use_lct_riesz` 均为 `false`，不包含任何可学习的 LCT/Riesz 参数。

三种 residual 模型使用相同 BTC 数据、9 维输入、signal indices、时间划分、sequence length、seed 42、20 epoch 上限、batch size 32、learning rate 0.001、AdamW、weight decay 0.0001、MSE loss 和最低 validation loss checkpoint 规则。

公平性配置和对应测试见：

- `experiments/lstm/configs/lstm_volatility_5_residual_lct_signal_features.yaml`
- `experiments/lstm/configs/lstm_volatility_5_residual_no_lct_signal_features.yaml`
- `experiments/lstm/configs/lstm_volatility_5_residual_temporal_signal_features.yaml`
- `tests/test_residual_control_models.py`
- `tests/test_temporal_residual_control.py`

## 3. 正式结果

以下指标重新读取自四组正式实验各自的 `metrics.json`。

| 模型 | Auxiliary block | 参数量 | 最佳 epoch | MAE | MSE | RMSE | R² |
| -- | -- | --: | --: | --: | --: | --: | --: |
| Plain LSTM + signals | 无 | 58,753 | 6 | 0.007876970765 | 0.000110207683 | 0.010497984719 | 0.051217592807 |
| Residual LCT-Riesz | LCT-Riesz | 59,067 | 5 | 0.008156298901 | 0.000113881330 | 0.010671519574 | 0.019591018311 |
| No-LCT residual | Identity | 59,011 | 20 | 0.008399916758 | 0.000117047422 | 0.010818845697 | -0.007665997668 |
| Temporal residual | Shared causal Conv56 | 59,067 | 4 | 0.010494205201 | 0.000160190021 | 0.012656619645 | -0.379082376388 |

原始结果位置：

- Plain LSTM：`experiments/lstm/outputs/volatility_5/baseline_signal_features/20260622_155816/metrics.json`
- Residual LCT-Riesz：`experiments/lstm/outputs/volatility_5/residual_lct_signal_features/20260625_163554/metrics.json`
- No-LCT residual：`experiments/lstm/outputs/volatility_5/residual_no_lct_signal_features/20260823_191733/metrics.json`
- Temporal residual：`experiments/lstm/outputs/volatility_5/residual_temporal_signal_features/20260823_200340/metrics.json`

No-LCT 的 RMSE 比 Residual LCT-Riesz 高 0.000147326123，即恶化 1.380554%。Residual LCT-Riesz 的预设 ±0.5% RMSE 筛查区间为 `[0.010618161976, 0.010724877172]`，No-LCT 未进入该范围。No-LCT 最佳 validation loss 出现在第 20 轮，可能尚未完全收敛；这只是收敛性提示，不能用于修改本次结果、追加训练或推断延长训练必然改善测试表现。

Temporal 的 RMSE 比 Residual LCT-Riesz 高 0.001985100072，即恶化 18.601850%，也未进入预设 ±0.5% 范围。Temporal 最佳 validation loss 出现在第 4 轮；本次只报告按预注册训练协议选择的 checkpoint，不据此声称 Temporal 已充分收敛或追加训练。

## 4. No-LCT 贡献诊断

诊断直接加载正式 `best_model.pth`，没有重新训练或更新权重。这里的 Main-only 是 residual 模型联合训练后的内部 main branch，不是独立训练的 Plain LSTM。

| 设置 | MAE | MSE | RMSE | R² |
| -- | --: | --: | --: | --: |
| Full | 0.008399916758 | 0.000117047422 | 0.010818845697 | -0.007665997668 |
| Main-only | 0.012856406406 | 0.000216090508 | 0.014700017294 | -0.860331937822 |
| Test correction mean | 0.008304827385 | 0.000113326931 | 0.010645512240 | 0.024363863037 |
| Validation correction mean | 0.008530689723 | 0.000117210834 | 0.010826395264 | -0.009072819974 |
| Validation MSE bias | 0.008458204507 | 0.000115955864 | 0.010768280470 | 0.001731268841 |
| Validation MAE bias | 0.007830591702 | 0.000105906091 | 0.010291068514 | 0.088250172451 |

Full 明显优于联合训练模型内部的 Main-only，但 Test correction mean 和 Validation MAE bias 均优于 Full。因而 Full 相对 Main-only 的改善不能证明辅助分支具有有效的样本级变化。

### 4.1 置换结果

| 置换 | 指标 | 置换均值 | 2.5% 分位数 | 97.5% 分位数 | Full 优于比例 | 经验 p 值 |
| -- | -- | --: | --: | --: | --: | --: |
| Random（1,000 次） | MAE | 0.008380240623 | 0.008289735167 | 0.008473105017 | 32.30% | 0.677322677323 |
| Random（1,000 次） | MSE | 0.000115646619 | 0.000113511719 | 0.000117821472 | 10.40% | 0.896103896104 |
| Random（1,000 次） | RMSE | 0.010753784555 | 0.010654187861 | 0.010854559951 | 10.40% | 0.896103896104 |
| Random（1,000 次） | R² | 0.004393577709 | -0.014329821245 | 0.022773014270 | 10.40% | 0.896103896104 |
| Circular shift（726 次） | MAE | 0.008383033022 | 0.008193189776 | 0.008603808085 | 43.66% | 0.563961485557 |
| Circular shift（726 次） | MSE | 0.000115674426 | 0.000111485256 | 0.000120548839 | 31.82% | 0.682255845942 |
| Circular shift（726 次） | RMSE | 0.010754554130 | 0.010558657860 | 0.010979473532 | 31.82% | 0.682255845942 |
| Circular shift（726 次） | R² | 0.004154182038 | -0.037809837627 | 0.040218918222 | 31.82% | 0.682255845942 |

Full 没有处于 random permutation 或 circular shift 分布的优异尾部。由于 `volatility_5` 相邻目标窗口重叠，经验 p 值只能作为描述性稳健性参考，不能解释为严格统计显著性。

### 4.2 Correction 性质与波动分区

- 727 个 correction 全部为负；
- correction 均值：-0.007141030784；
- correction 标准差：0.001532769960；
- 去均值 correction 与 `target - main_pred` 的相关系数：-0.044684483669；
- correction 改善 593/727 个样本，占 81.5681%；
- High volatility 区间仅改善 50/182 个样本，占 27.4725%。

| 波动区间 | 样本数 | Main MAE | Full MAE | Main RMSE | Full RMSE | 改善比例 |
| -- | --: | --: | --: | --: | --: | --: |
| Low | 182 | 0.019812706144 | 0.012692318853 | 0.020051848130 | 0.013078362257 | 100.00% |
| Normal | 363 | 0.012457638268 | 0.005713814205 | 0.013194681495 | 0.006958738107 | 99.45% |
| High | 182 | 0.006695451911 | 0.009464960965 | 0.010670307479 | 0.014139342483 | 27.47% |

诊断来源：

- `experiments/lstm/diagnostics/residual_contribution/no_lct_20260823_191733/residual_ablation_metrics.json`
- `experiments/lstm/diagnostics/residual_contribution/no_lct_20260823_191733/permutation_summary.json`
- `experiments/lstm/diagnostics/residual_contribution/no_lct_20260823_191733/residual_contribution_summary.json`
- `experiments/lstm/diagnostics/residual_contribution/no_lct_20260823_191733/residual_regime_metrics.csv`

## 5. Temporal 贡献诊断

Temporal 诊断直接加载正式第 4 轮 `best_model.pth`，没有重新训练或更新权重。诊断 Full 与正式 `metrics.json` 的四项指标差均为 0，逐样本 target 和 final prediction 的最大绝对差均小于 `1e-15`。

| 设置 | MAE | MSE | RMSE | R² |
| -- | --: | --: | --: | --: |
| Full | 0.010494205201 | 0.000160190021 | 0.012656619645 | -0.379082376388 |
| Main-only | 0.011762485643 | 0.000189163225 | 0.013753662231 | -0.628513862065 |
| Test correction mean | 0.010442654219 | 0.000158472229 | 0.012588575328 | -0.364293835000 |
| Validation correction mean | 0.010605099664 | 0.000162105902 | 0.012732081590 | -0.395576271475 |
| Validation MSE bias | 0.010506424159 | 0.000159887393 | 0.012644658667 | -0.376477039309 |
| Validation MAE bias | 0.009382975588 | 0.000136146108 | 0.011668166452 | -0.172087359991 |

Full 优于联合训练模型内部的 Main-only，但 Test correction mean 和 Validation MAE bias 均优于 Full，未显示 Temporal correction 的样本级变化优于简单偏置校准。

### 5.1 置换结果

| 置换 | 指标 | 置换均值 | 2.5% 分位数 | 97.5% 分位数 | Full 优于比例 | 经验 p 值 |
| -- | -- | --: | --: | --: | --: | --: |
| Random（1,000 次） | MAE | 0.010459569863 | 0.010417079983 | 0.010499111489 | 4.80% | 0.952047952048 |
| Random（1,000 次） | RMSE | 0.012614636654 | 0.012564767926 | 0.012660238865 | 4.00% | 0.960039960040 |
| Circular shift（726 次） | MAE | 0.010460574725 | 0.010377156266 | 0.010552667976 | 36.23% | 0.638239339752 |
| Circular shift（726 次） | RMSE | 0.012614941014 | 0.012498518218 | 0.012744000179 | 33.47% | 0.665749656121 |

Full 没有处于 random permutation 或 circular shift 分布的优异尾部。`volatility_5` 相邻目标窗口重叠，因此这些经验 p 值仍只是描述性稳健性参考，不构成严格统计显著性结论。

### 5.2 Correction 与预测分布

- correction 均值：-0.001884470864；
- correction 标准差：0.000819105760；
- correction 范围：-0.003386634169 至 0.000724396145，并非全部同号；
- 去均值 correction 与 `target - main_pred` 的相关系数：-0.061887077997；
- correction 改善 592/727 个样本，占 81.4305%；
- High volatility 区间仅改善 69/182 个样本，占 37.9121%，且 Full MAE/RMSE 均比 Main-only 更差；
- prediction 标准差为 0.002090597710，target 标准差为 0.010777613985，标准差比仅为 0.193976，存在明显预测幅度收缩；
- prediction 与 target 相关系数为 0.270519，无负预测或 NaN/Inf。

诊断来源：

- `experiments/lstm/diagnostics/residual_contribution/temporal_20260823_200340/residual_ablation_metrics.json`
- `experiments/lstm/diagnostics/residual_contribution/temporal_20260823_200340/permutation_summary.json`
- `experiments/lstm/diagnostics/residual_contribution/temporal_20260823_200340/residual_contribution_summary.json`
- `experiments/lstm/diagnostics/residual_contribution/temporal_20260823_200340/residual_regime_metrics.csv`
- `experiments/lstm/outputs/volatility_5/residual_temporal_signal_features/20260823_200340/prediction_results.csv`

## 6. 三种 Residual 机制比较

表中“优于 Test correction mean”要求 Full 的 MAE 和 RMSE 同时更低。

| 诊断项 | LCT residual | No-LCT residual | Temporal residual |
| -- | --: | --: | --: |
| residual scale | 0.011100948788 | -0.016056546941 | 0.008938671090 |
| correction 均值 | -0.004071498680 | -0.007141030784 | -0.001884470864 |
| correction 标准差 | 0.000684579525 | 0.001532769960 | 0.000819105760 |
| correction 符号 | 全负 | 全负 | 正负均有 |
| correction 与 `target-main_pred` 相关性 | -0.024922039415 | -0.044684483669 | -0.061887077997 |
| 改善样本比例 | 76.89% | 81.57% | 81.43% |
| High volatility 改善比例 | 12.09% | 27.47% | 37.91% |
| Full 优于 Test correction mean | 否 | 否 | 否 |
| Random：Full 优于比例（MAE/RMSE） | 90.40% / 24.10% | 32.30% / 10.40% | 4.80% / 4.00% |
| Circular：Full 优于比例（MAE/RMSE） | 64.19% / 45.59% | 43.66% / 31.82% | 36.23% / 33.47% |

三种 residual correction 与真实所需残差的相关性都接近零，Full 均未在 MAE 和 RMSE 上同时超过 Test correction mean，并且三者都恶化 High volatility 区间的整体 MAE/RMSE。该对照是固定 seed 42 的机制筛查，不能解释为统计显著性、统计等效性或稳定的跨 seed 排名。

Residual LCT-Riesz 的历史诊断来源为 `experiments/lstm/diagnostics/residual_contribution/20260625_163554/`；No-LCT 的诊断来源保留在第 4 节。

## 7. 最终结果解释

- Residual LCT-Riesz 优于 No-LCT：RMSE 低 1.380554%；
- Residual LCT-Riesz 优于参数完全匹配的 Temporal：RMSE 低 18.601850%；
- 因此 Identity 辅助分支和本次共享因果 Conv56 不能完全解释 LCT 与控制模型的性能差异；
- 但 Residual LCT-Riesz 的 RMSE 仍比独立 Plain LSTM 高约 1.65%；
- LCT Full 没有明确优于 Test correction mean：其 MAE 略好，但 RMSE 和 R²更差；
- LCT correction 与实际所需残差的相关系数仅为 -0.024922，接近零；
- LCT Full 在 random permutation 中只优于 24.10% 的 RMSE，在 circular shift 中只优于 45.59% 的 RMSE，均不处于优异尾部；
- LCT correction 在 High volatility 区间仅改善 12.09% 的样本，并明显恶化整体 MAE/RMSE。

因此，“LCT 优于两个辅助控制”不等于“LCT 提高了预测性能”，也不能证明频域分支学到了稳定、有效的样本级 correction。

## 8. 最终路线决策

> 当前证据支持 LCT-Riesz block 相对 Identity 和本次参数匹配时域卷积具有一定结构差异，但不支持其相对 Plain LSTM 产生有效预测增益，也不支持其学到稳定的样本级频域修正。项目在此停止扩展当前 residual auxiliary 结构，不开发 gated LCT，不对当前结构开展多 seed，不继续增加辅助控制模型。

停止的是当前 residual 融合结构，不等于证明所有 LCT-Riesz 金融应用无效。下一步应重新评估任务定义、LCT 特征使用位置和论文研究问题；当前结果应作为严格消融和失败机制分析保留。
