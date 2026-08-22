# Plain Transformer 实验总结

## 1. 实验目的

本实验用于补齐 Plain LSTM、Plain TCN 与 Plain Transformer 三种时序骨干在相同 BTC `volatility_5` 预测任务、相同时间划分和相同输入特征下的受控基线比较。实验仅评估 Plain Transformer，不使用 LCT-Riesz，也不进行超参数搜索。

## 2. 实验配置

- 数据文件：`data/processed/BTC_daily_train.csv`
- 时间划分：按时间顺序划分，train / validation / test = `0.70 / 0.15 / 0.15`
- 预测任务：`volatility_5`
- Sequence length：`60`
- 输入特征：`Open`、`High`、`Low`、`Close`、`Volume`、`log_return`、`abs_log_return`、`high_low_range`、`close_open_return`
- Input dimension：`9`（5 个 OHLCV + 4 个 signal features）
- Seed：`42`
- Epoch：`20`
- Batch size：`32`
- Learning rate：`0.001`
- Weight decay：`0.0001`
- Transformer 层数：`2`
- Attention heads：`4`
- `d_model`：`56`
- FFN dimension：`112`
- Dropout：`0.1`
- Causal attention：启用
- 可训练参数量：`55,497`
- 优化器：AdamW
- Loss：MSE
- 运行设备：NVIDIA GeForce GTX 1650（CUDA）

本次运行共得到 3,624 个训练样本、726 个验证样本和 727 个测试样本。最佳 validation loss 为 `0.0005524914407887439`，出现在第 20 轮。

## 3. 正式实验结果

下表指标均重新读取自各正式实验目录中的 `metrics.json`。参数量来自对应实验的 `experiment_summary.txt`；naive baseline 不含可训练模型，因此最佳 epoch 和参数量不适用。

| 模型 | MAE | MSE | RMSE | R² | 最佳 epoch | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| Plain LSTM + signal features | 0.007876970764874714 | 0.00011020768316091732 | 0.010497984719026663 | 0.051217592807093815 | 6 | 58,753 |
| Plain Transformer + signal features | 0.008635788236807473 | 0.00013165292309795775 | 0.011474010767728858 | -0.1334053462359639 | 20 | 55,497 |
| Historical volatility naive baseline | 0.00966588673854477 | 0.00018621895153580795 | 0.013646206488830805 | -0.603166497324142 | — | — |
| 当前最佳 Plain TCN + signal features | 0.010913256223363471 | 0.00021691855322990034 | 0.014728155119698473 | -0.8674606088738896 | 33 | 59,177 |

## 4. 预测分布检查

以下统计量由本次 Transformer 实验的 `prediction_results.csv` 重新计算；标准差使用总体标准差定义。

| 检查项 | 数值 |
|---|---:|
| Prediction 均值 | 0.0195948493281351 |
| Prediction 标准差 | 0.00820342329336754 |
| Target 均值 | 0.0192154006657166 |
| Target 标准差 | 0.0107776139847375 |
| Prediction / target 标准差比 | 0.761153934904764 |
| Prediction 与 target 相关系数 | 0.293757600540316 |
| 负预测数量 | 0 / 727 |
| Prediction 中 NaN / Inf 数量 | 0 |
| Target 中 NaN / Inf 数量 | 0 |

Prediction 均值与 target 均值接近，但 prediction 标准差仅为 target 标准差的约 76.12%，说明预测振幅存在可见的收缩。该现象是当前单次实验的诊断观察，不构成其成因已经确定的结论。

## 5. 当前结论

1. 在本次相同数据、任务和输入特征的单 seed 比较中，Plain Transformer 优于当前最佳 Plain TCN 和 historical-volatility naive baseline。
2. Plain Transformer 没有超过 Plain LSTM；当前三种时序骨干中，Plain LSTM 仍然最好。
3. Transformer 的最佳 validation loss 出现在第 20 轮，提示它可能尚未完全收敛，但不能据此认定延长训练一定会改善测试结果。
4. 当前结果仅来自 `seed = 42`，不能作为统计显著性结论，也不能推广为 Transformer 在其他资产或任务上的一般表现。
5. 当前不建议进行大规模超参数搜索；如需继续核查，后续最多进行一次预先限定训练预算和判断标准的收敛性诊断。
6. 后续 LCT-Riesz 研究仍以 LSTM 作为首选主干；本次结果不支持优先转向 Transformer 主干。

## 6. 实验文件位置

- 配置：`experiments/transformer/configs/transformer_volatility_5_baseline_signal_features.yaml`
- 输出目录：`experiments/transformer/outputs/volatility_5/baseline_signal_features/20260821_215702`
- Checkpoint 目录：`experiments/transformer/checkpoints/volatility_5/baseline_signal_features/20260821_215702`
- 指标：`experiments/transformer/outputs/volatility_5/baseline_signal_features/20260821_215702/metrics.json`
- 预测结果：`experiments/transformer/outputs/volatility_5/baseline_signal_features/20260821_215702/prediction_results.csv`
- 训练日志：`experiments/transformer/outputs/volatility_5/baseline_signal_features/20260821_215702/training_log.csv`
- 实验摘要：`experiments/transformer/outputs/volatility_5/baseline_signal_features/20260821_215702/experiment_summary.txt`

本次正式实验登记在 `experiments/transformer/experiment_index.csv` 中。该索引不包含 smoke test、调试运行或不存在的实验。
