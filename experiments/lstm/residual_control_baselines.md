# Residual auxiliary 控制实验记录

## 1. 控制目的

No-LCT signal auxiliary residual 是 existing residual LCT-Riesz 的机制控制。它保留相同的 main branch、4 个 signal features 的辅助输入、时间均值池化、residual 联合训练和可学习 residual scale，仅将 LCT-Riesz 模块替换为 Identity。该控制用于检验以下因素是否足以解释原 residual LCT-Riesz 的结果：

- 增加辅助分支本身；
- signal features 在辅助分支中的重复输入；
- 时间均值池化；
- main branch 与 auxiliary branch 的联合训练；
- correction 带来的整体偏置校准。

本次是固定 seed 42 的单次机制筛查，不是调参实验，也不构成统计等效性或显著性检验。

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

在固定 seed 下，Residual LCT-Riesz 与 No-LCT 的 main branch 参数名称、形状和初始数值逐元素一致。No-LCT 参数量为 59,011，Residual LCT-Riesz 参数量为 59,067，仅相差 LCT-Riesz block 的 56 个参数。No-LCT 的 `use_lct_riesz` 为 `false`，模型中不包含任何可学习的 LCT/Riesz 参数。

公平性配置和对应测试见：

- `experiments/lstm/configs/lstm_volatility_5_residual_lct_signal_features.yaml`
- `experiments/lstm/configs/lstm_volatility_5_residual_no_lct_signal_features.yaml`
- `tests/test_residual_control_models.py`

## 3. 正式结果

以下指标重新读取自三组正式实验各自的 `metrics.json`。

| 模型 | LCT-Riesz | 参数量 | 最佳 epoch | MAE | MSE | RMSE | R² |
| -- | -- | --: | --: | --: | --: | --: | --: |
| Plain LSTM + signals | 否 | 58,753 | 6 | 0.007876970765 | 0.000110207683 | 0.010497984719 | 0.051217592807 |
| Residual LCT-Riesz | 是 | 59,067 | 5 | 0.008156298901 | 0.000113881330 | 0.010671519574 | 0.019591018311 |
| No-LCT residual | 否 | 59,011 | 20 | 0.008399916758 | 0.000117047422 | 0.010818845697 | -0.007665997668 |

原始结果位置：

- Plain LSTM：`experiments/lstm/outputs/volatility_5/baseline_signal_features/20260622_155816/metrics.json`
- Residual LCT-Riesz：`experiments/lstm/outputs/volatility_5/residual_lct_signal_features/20260625_163554/metrics.json`
- No-LCT residual：`experiments/lstm/outputs/volatility_5/residual_no_lct_signal_features/20260823_191733/metrics.json`

No-LCT 的 RMSE 比 Residual LCT-Riesz 高 0.000147326123，即恶化 1.380554%。Residual LCT-Riesz 的预设 ±0.5% RMSE 筛查区间为 `[0.010618161976, 0.010724877172]`，No-LCT 未进入该范围。No-LCT 最佳 validation loss 出现在第 20 轮，可能尚未完全收敛；这只是收敛性提示，不能用于修改本次结果、追加训练或推断延长训练必然改善测试表现。

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

## 5. LCT 与 No-LCT 机制比较

| 诊断项 | Residual LCT-Riesz | No-LCT residual |
| -- | --: | --: |
| residual scale | 0.011100948788 | -0.016056546941 |
| correction 均值 | -0.004071498680 | -0.007141030784 |
| correction 标准差 | 0.000684579525 | 0.001532769960 |
| correction 与 `target-main_pred` 相关性 | -0.024922039415 | -0.044684483669 |
| 改善样本比例 | 76.89% | 81.57% |
| High volatility 改善比例 | 12.09% | 27.47% |
| Full 明确优于常数修正 | 否 | 否 |
| Full 处于置换分布优异尾部 | 否 | 否 |

两种模型的 correction 均全部为负，与所需样本残差的相关性均接近零；两者都没有明确优于常数修正或置换分布，并且都在 High volatility 区间恶化整体误差。Residual LCT-Riesz 的正式 RMSE 比 No-LCT 低 1.380554%，而 No-LCT 未进入预设的 ±0.5% 筛查范围。该结果只是单 seed 机制筛查，不能解释为统计等效性、统计显著性或稳定的跨 seed 优势。

Residual LCT-Riesz 的历史诊断来源：`experiments/lstm/diagnostics/residual_contribution/20260625_163554/`。

## 6. 当前结论

> 在相同 seed 和训练协议下，Residual LCT-Riesz 的整体测试指标优于 No-LCT 辅助分支，说明 Identity/No-LCT 控制尚不能完全解释两者的性能差异。但两种模型的 correction 均主要表现为全局负向校准，均缺少样本级误差对齐证据；且 Residual LCT-Riesz 仍弱于独立 Plain LSTM。因此当前结果不足以证明 LCT-Riesz 具有独特且有效的频域预测贡献。

## 7. 下一步

- 按预注册规则，下一项允许实施 parameter-matched temporal auxiliary control；
- 该控制用于排除普通时域建模能力及相近参数量带来的解释；
- 暂不实现 improved gated LCT；
- 暂不进行多 seed；
- temporal control 完成后必须停止，并重新判断当前 residual LCT-Riesz 路线是否值得继续。
