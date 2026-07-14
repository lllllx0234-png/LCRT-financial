# TCN 实验总结

## 1. 实验目的

本文档总结 Plain TCN 主干在 BTC `volatility_5` 预测任务上的阶段性实验结果。本阶段的目标是评估因果 Temporal Convolutional Network 在与当前 Plain LSTM 信号特征基线相同的数据集、预测目标、输入特征和训练流程下，是否能够作为具有竞争力的非循环时序主干模型。

本阶段实验范围严格限定为 Plain TCN。不引入 LCT-Riesz 模块、不实现 LCRT-TCN 变体、不实现 Transformer、不进行多随机种子实验，也不加入额外的输出约束。

TCN 实验目录均根据 `experiments/tcn/experiment_index.csv` 中记录的 run_dir 精确确定，而不是根据文件系统中的最新目录推断：

- `experiments/tcn/outputs/volatility_5/baseline_signal_features/20260713_192701`
- `experiments/tcn/outputs/volatility_5/baseline_signal_features/20260713_200655`
- `experiments/tcn/outputs/volatility_5/baseline_signal_features/20260714_192542`

Plain LSTM 对照行使用 `experiments/lstm/experiment_index.csv` 中正式记录的 `lstm_volatility_5_baseline_signal_features` 实验结果及其对应 run 目录下的指标文件。

## 2. 实验设置

TCN 与 LSTM 的比较使用相同的受控预测设置：

- 数据集：`data/processed/BTC_daily_train.csv`
- 划分方式：chronological train / validation / test split
- 预测目标：`target_type = volatility_5`
- 输入窗口长度：`60`
- 随机种子：`42`
- 批大小：`32`
- 权重衰减：`0.0001`
- 输出维度：`1`
- LCT-Riesz 使用情况：Plain TCN 和 Plain LSTM 基线均未启用

输入特征完全一致，共包含 9 个特征：

| 序号 | 特征 |
|---:|---|
| 0 | Open |
| 1 | High |
| 2 | Low |
| 3 | Close |
| 4 | Volume |
| 5 | log_return |
| 6 | abs_log_return |
| 7 | high_low_range |
| 8 | close_open_return |

派生信号特征为 `log_return`、`abs_log_return`、`high_low_range` 和 `close_open_return`。数据划分比例为 `train_ratio = 0.7`、`val_ratio = 0.15` 和 `test_ratio = 0.15`。

## 3. TCN 网络结构

正式 Plain TCN 配置如下：

- `model_type`：`plain_tcn`
- `input_dim`：`9`
- `channels`：`[52, 52, 52, 52]`
- `kernel_size`：`3`
- `dropout`：`0.1`
- `dilation`：`[1, 2, 4, 8]`
- `output_dim`：`1`
- `trainable_parameters`：`59177`
- `use_lct_riesz`：`False`

TCN 使用带显式左侧 padding 的因果一维卷积，因此时间位置 `t` 的时序特征只依赖于 `t` 及其之前的输入位置。理论感受野为：

```text
1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 61
```

因此，正式 TCN 可以覆盖完整的 `sequence_length = 60` 输入窗口。这使其与 Plain LSTM 信号特征基线在可用历史信息范围上具有结构层面的公平性。

## 4. 已完成实验

本阶段完成了三组受控 Plain TCN 实验：

| 实验 | 学习率 | 训练轮数 | 实验目的 |
|---|---:|---:|---|
| `tcn_volatility_5_baseline_signal_features` | 0.001 | 20 | 初始 Plain TCN 基线 |
| `tcn_volatility_5_baseline_signal_features_lr3e4` | 0.0003 | 20 | 学习率降低诊断 |
| `tcn_volatility_5_baseline_signal_features_lr3e4_40epoch` | 0.0003 | 40 | 收敛训练轮数诊断 |

第二组实验相对于初始 TCN 基线仅修改了学习率和实验名称。第三组实验相对于 20 轮低学习率 TCN 配置仅修改了训练轮数和实验名称。

## 5. 实验结果

下表汇总了每组实验 `metrics.json` 和对应 experiment index 记录中的正式指标。表中数值为便于展示进行了截断或四舍五入，后续分析使用原始完整精度数值。

| 模型 | 学习率 | 训练轮数 | 最佳轮次 | MAE | RMSE | MSE | R² | 最佳验证损失 | 测试损失 | 可训练参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain LSTM + 信号特征 | 0.001 | 20 | 6 | 0.00787697 | 0.0104980 | 0.000110208 | 0.0512176 | 0.000149639 | 0.000110208 | 58753 |
| Plain TCN，lr=0.001，20轮 | 0.001 | 20 | 14 | 0.0340233 | 0.0379713 | 0.00144182 | -11.4127 | 0.000372528 | 0.00144182 | 59177 |
| Plain TCN，lr=0.0003，20轮 | 0.0003 | 20 | 20 | 0.0204080 | 0.0252602 | 0.000638077 | -4.49323 | 0.000468466 | 0.000638077 | 59177 |
| Plain TCN，lr=0.0003，40轮 | 0.0003 | 40 | 33 | 0.0109133 | 0.0147282 | 0.000216919 | -0.867461 | 0.000284969 | 0.000216919 | 59177 |

最佳 Plain TCN 实验为 `tcn_volatility_5_baseline_signal_features_lr3e4_40epoch`。该结果相较初始 TCN 基线有明显改善，但在所有主要测试误差指标上仍弱于 Plain LSTM 信号特征基线。

## 6. 学习率分析

初始 TCN 基线使用 `learning_rate = 0.001`，训练 20 轮。该实验的最佳验证轮次为第 14 轮，但训练损失和验证损失存在较明显波动。其测试结果较弱，`R2 = -11.412654404092867`，说明在当前指标定义下，该配置未能有效解释测试集 `volatility_5` 目标方差。

在保持 20 轮训练预算不变的情况下，将学习率从 `0.001` 降低到 `0.0003` 后，TCN 测试指标得到改善：

- MAE 下降 `40.0174%`。
- RMSE 下降 `33.4755%`。
- MSE 下降 `55.7449%`。
- R2 提升 `6.91942`，从 `-11.412654404092867` 提升至 `-4.4932311775769405`。

这表明在当前设置下，原始 `0.001` 学习率对 Plain TCN 可能偏大。不过，20 轮低学习率实验的最佳验证 checkpoint 位于第 20 轮，即最后一轮。这提示模型在 20 轮训练预算内尚未明确完成收敛。

## 7. 训练轮数分析

40 轮诊断实验保持 `learning_rate = 0.0003`，并保持其他数据、模型和训练参数不变。将训练预算从 20 轮延长到 40 轮后，Plain TCN 进一步改善：

- MAE 下降 `46.5247%`。
- RMSE 下降 `41.6942%`。
- MSE 下降 `66.0043%`。
- R2 提升 `3.62577`，从 `-4.4932311775769405` 提升至 `-0.8674606088738896`。

最佳验证 checkpoint 出现在第 33 轮。第 33 轮之后，验证损失稳定性有所下降：例如，验证损失从第 33 轮的 `0.0002849693770088081` 变化到第 34 轮的 `0.00029596844614296125`、第 35 轮的 `0.0003638688599005773`，以及第 36 轮的 `0.0006390209628856605`。这说明 40 轮训练为 TCN 提供了进一步改善的空间，但从已观察到的验证损失行为来看，继续无限增加训练轮数缺乏充分依据。

在当前阶段，Plain TCN 相比 20 轮版本已经达到更合理的收敛状态；进一步进行无边界调参将使研究偏离受控主干比较的目标。

## 8. 与 Plain LSTM 的比较

最佳 TCN 为 40 轮低学习率实验。与 Plain LSTM + 信号特征基线相比：

- TCN MAE 比 LSTM MAE 高 `38.5464%`。
- TCN RMSE 比 LSTM RMSE 高 `40.2951%`。
- TCN MSE 为 LSTM MSE 的 `1.96827` 倍。
- TCN R2 低 `0.918678`，其中 TCN 为 `-0.8674606088738896`，LSTM 为 `0.051217592807093815`。

在相同数据集、chronological split、预测目标、输入特征、随机种子、批大小、权重衰减以及近似参数规模条件下，最佳 Plain TCN 在当前 BTC `volatility_5` 实验中仍弱于 Plain LSTM 信号特征基线。

该结论仅适用于当前实验条件，不应被解释为 TCN 不适合所有金融预测任务，也不应被解释为 LSTM 在所有数据集和预测目标上必然优于 TCN。

## 9. 预测行为分析

低学习率 TCN 的预测曲线和 `prediction_results.csv` 显示出若干有参考价值的诊断现象：

- 20 轮低学习率实验存在整体高估倾向。其测试集预测值均值为 `0.035933423208321304`，而真实值均值为 `0.019215400665716628`。
- 40 轮实验降低了这种整体高估程度，但局部预测振幅在部分位置仍可能大于附近真实波动率。
- Plain Linear 输出层可能产生负波动率预测。20 轮低学习率实验包含 1 个负预测值，40 轮低学习率实验包含 24 个负预测值。
- 尽管 40 轮实验相较 20 轮 TCN 实验在测试指标上明显改善，其预测曲线仍存在局部尖峰和负值波动。

这些现象可能有助于解释为什么最佳 TCN 在经过学习率和训练轮数诊断后测试 R2 仍为负值。但这些内容应被视为诊断观察，而不是已经证明的因果解释。

本阶段实验没有加入 Softplus、ReLU、截断或其他非负输出约束。这是有意控制变量的结果：三组 TCN 实验旨在隔离学习率和训练轮数的影响，同时保持模型结构和输出层不变。

## 10. 当前结论

当前 Plain TCN 阶段已完成以下工作与判断：

1. Plain TCN 的实现、正式训练和受控收敛诊断已经完成。
2. 将学习率从 `0.001` 降低到 `0.0003` 明显改善了 TCN 的测试性能和训练行为。
3. 将低学习率实验从 20 轮延长到 40 轮进一步改善了 TCN 性能。
4. 最佳 Plain TCN 为 `tcn_volatility_5_baseline_signal_features_lr3e4_40epoch`。
5. 在当前 BTC `volatility_5` 设置下，最佳 Plain TCN 仍弱于 Plain LSTM + 信号特征。
6. 当前阶段不建议继续进行开放式、无边界的 TCN 超参数搜索。
7. Plain TCN 应作为论文实验中的独立时序主干对比模型保留。
8. 当前结果支持继续将 LSTM 作为后续 LCT-Riesz 增强实验的主要主干。
9. 除非后续论文结构明确需要，否则当前暂不实现 LCRT-TCN。

## 11. 下一步工作

下一组受控 baseline 应进入 Plain Transformer。这符合从 LSTM 到 TCN 再到 Transformer 的主干比较路线，也有助于在决定是否将频域增强扩展到 LSTM 以外主干之前，建立更完整的架构对照。

Transformer 阶段仍应尽量遵守与当前实验相同的约束：保持相同的数据、预测目标、输入特征列、chronological split、随机种子、批大小和评估输出。只有在 Plain Transformer baseline 建立之后，才适合进一步判断论文论证是否需要非 LSTM 的 LCT-Riesz 变体。
