# Experiments Summary

## 1. Project Goal

本项目的目标是将一维 Learnable LCT-Riesz 频域增强模块接入 LSTM，用于 BTC 日线金融时间序列预测，并与普通 LSTM 以及简单 naive baseline 进行对比。当前阶段重点关注两个预测任务：

- 下一日收益率预测（return prediction）；
- 下一日收盘价预测（close price prediction）。

实验设计的核心问题是：LCT-Riesz 频域增强模块是否能在相同 LSTM 主干结构下，相比普通 LSTM baseline 带来稳定增益；同时，该增益是否能够超过金融时间序列中常用的简单 persistence / zero-return 基准。

## 2. Dataset

- 数据文件：`data/processed/BTC_daily_train.csv`
- 特征列：`Open`, `High`, `Low`, `Close`, `Volume`
- 数据划分：按时间顺序划分为 train / validation / test，不进行随机划分。
- 输入窗口长度：`sequence_length = 60`
- 数据集任务：
  - `return`：预测下一日收益率；
  - `close`：预测下一日收盘价。

对于输入特征，所有神经网络实验均使用训练集统计量进行 `StandardScaler` 标准化，并将同一 scaler 应用于 validation / test。对于 `close` 任务，后续已修复 target scaling：训练时 close target 使用训练集 close 拟合的 target scaler 进行标准化，测试输出和保存结果时再反标准化回原始价格尺度。

## 3. Models Compared

本阶段比较以下模型：

- **LCT-Riesz + LSTM**：输入投影后接入一维 Learnable LCT-Riesz 频域增强模块，再输入 LSTM。
- **Plain LSTM baseline**：使用相同输入投影、LSTM 主干和预测头，但关闭 LCT-Riesz 模块。
- **Naive zero-return baseline**：用于收益率任务，预测下一日收益率恒为 0。
- **Naive last-close baseline**：用于收盘价任务，预测下一日 Close 等于输入窗口最后一天 Close。

其中，LCT-Riesz + LSTM 与 Plain LSTM 的差异仅在于是否启用 LCT-Riesz 频域增强模块。二者使用相同输入维度、隐藏维度、LSTM 层数、预测头和训练配置。Naive baseline 不使用神经网络，也不进行梯度训练。

## 4. Return Prediction Results

有效 return 实验包括：

- `outputs/experiment_20260616_151529_lct`
- `outputs/experiment_20260616_153120`
- `outputs/experiment_20260616_161430`
- `outputs/experiment_20260616_161511`
- `outputs/naive_20260617_114424`

| Run | Model | Epochs / Rule | RMSE | MAE | MSE | R2 | Directional Accuracy | Best Val Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `experiment_20260616_151529_lct` | LCT-Riesz + LSTM | 5 epoch | 0.024379 | 0.017374 | 0.000594348 | -0.001564 | 0.493844 | 0.000874162 |
| `experiment_20260616_153120` | Plain LSTM baseline | 5 epoch | 0.024416 | 0.017418 | 0.000596122 | -0.004553 | 0.496580 | 0.000863460 |
| `experiment_20260616_161430` | LCT-Riesz + LSTM | 20 epoch | 0.024611 | 0.017703 | 0.000605717 | -0.020721 | 0.504788 | 0.000839883 |
| `experiment_20260616_161511` | Plain LSTM baseline | 20 epoch | 0.024732 | 0.017845 | 0.000611674 | -0.030760 | 0.504788 | 0.000844206 |
| `naive_20260617_114424` | Naive zero-return baseline | `y_pred = 0` | 0.024361 | 0.017236 | 0.000593451 | -0.000051 | 0.000000 | N/A |

从误差指标看，LCT-Riesz + LSTM 相比 Plain LSTM 在 5 epoch 和 20 epoch 设置下均有轻微优势。例如，5 epoch 下 LCT-Riesz + LSTM 的 RMSE 为 0.024379，Plain LSTM 为 0.024416；20 epoch 下 LCT-Riesz + LSTM 的 RMSE 为 0.024611，Plain LSTM 为 0.024732。该差异说明频域增强模块对普通 LSTM 有一定增益，但幅度较小。

不过，Naive zero-return baseline 在 RMSE、MAE 和 MSE 上仍然非常有竞争力，甚至略优于两个深度模型。这说明 BTC 日线收益率预测中，收益率序列接近零均值、噪声较强，模型容易学习到接近 0 的平滑预测。深度模型的 direction accuracy 接近 50%，约等同于随机方向判断，说明当前模型尚未形成稳定的方向预测能力。

需要注意，Naive zero-return 的 directional accuracy 为 0.0，这是因为其预测符号恒为 0，而真实收益率通常非 0；该指标对 zero-return 基准的解释需要谨慎。更重要的比较应放在 RMSE、MAE 和 MSE 等误差指标上。

## 5. Close Prediction Results

有效 close 实验包括：

- `outputs/experiment_20260617_104838`
- `outputs/experiment_20260617_105015`
- `outputs/naive_20260617_113209`

| Run | Model | Rule | RMSE | MAE | MSE | MAPE | R2 | Best Val Loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `experiment_20260617_104838` | LCT-Riesz + LSTM | target scaling fixed | 30374.8 | 24962.0 | 922629046 | 25.4326 | -1.62736 | 0.0186519 |
| `experiment_20260617_105015` | Plain LSTM baseline | target scaling fixed | 30396.6 | 25085.5 | 923952850 | 25.6134 | -1.63113 | 0.0196311 |
| `naive_20260617_113209` | Naive last-close baseline | `y_pred[t] = last close` | 2031.34 | 1459.96 | 4126331 | 1.72367 | 0.988249 | N/A |

修复 target scaling 后，close 任务的 `prediction_results.csv`、`metrics.json` 和预测曲线均回到原始价格尺度。此时神经网络模型的训练损失在标准化 target 空间中计算，而最终评估指标在原始 Close 价格空间中计算。

LCT-Riesz + LSTM 相比 Plain LSTM 仍然有轻微误差优势：RMSE 从 30396.6 降至 30374.8，MAE 从 25085.5 降至 24962.0，MAPE 从 25.6134 降至 25.4326。但该优势非常有限。

相比之下，Naive last-close baseline 明显优于两个深度模型。其 RMSE 为 2031.34，MAE 为 1459.96，R2 为 0.988249。该结果表明，在直接预测价格水平时，当前 LSTM 类模型未能有效学习价格序列的强连续性和局部 persistence 特征。对于 close 任务，`next close ≈ last close` 是一个非常强的金融基准，必须在论文实验比较中保留。

## 6. Invalid Experiments

以下两组实验明确标记为无效/调试实验，不纳入正式比较：

- `outputs/experiment_20260616_181022`
- `outputs/experiment_20260616_181405`

原因：这两组 close 实验发生在 target scaling 修复前。当时 `target_type="close"` 时，输入特征已经经过 `StandardScaler` 标准化，但 target y 仍为原始 Close 价格，导致训练和验证损失达到 `1e9` 级别，预测曲线中模型输出接近 0，而真实 Close 位于约 60000 到 120000 的价格区间。因此这两组实验仅作为调试记录，不能作为有效实验结果。

无效实验的主要指标如下：

| Run | Model | Target | RMSE | MAE | MSE | Best Val Loss | Status |
|---|---|---|---:|---:|---:|---:|---|
| `experiment_20260616_181022` | LCT-Riesz + LSTM | close | 89373.5 | 87386.9 | 7987629771 | 1048381844 | invalid |
| `experiment_20260616_181405` | Plain LSTM baseline | close | 89374.2 | 87387.5 | 7987745280 | 1048421436 | invalid |

## 7. Learned LCT Parameters

有效 LCT-Riesz 实验中，`learned_lct_parameters.txt` 记录了可学习 LCT 参数与对应矩阵。摘录如下：

| Run | Task | alpha | m | q | gamma | A | B | C | D | determinant |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `experiment_20260616_151529_lct` | return, 5 epoch | 1.00965 | 1.01298 | -3.93239e-06 | 1 | -0.0153485 | 1.01287 | -0.987069 | -0.0149536 | 0.9999999445 |
| `experiment_20260616_161430` | return, 20 epoch | 1.00298 | 0.991464 | -7.08560e-07 | 1 | -0.00463527 | 0.991453 | -1.00860 | -0.00471473 | 1.000000094 |
| `experiment_20260617_104838` | close, 20 epoch | 0.949332 | 0.977415 | -0.0100036 | 1 | 0.0777088 | 0.974321 | -1.01909 | 0.0910882 | 0.9999999513 |

三组有效 LCT-Riesz 实验的矩阵行列式均满足 `AD - BC ≈ 1`。这说明当前 Learnable LCT 参数化机制运行正常，在训练过程中保持了合法 LCT 矩阵约束。参数发生了小幅变化，但未出现明显数值失稳。

## 8. Current Conclusion

当前实验结果支持以下谨慎结论：

1. LCT-Riesz 频域增强模块相对普通 LSTM 在若干误差指标上有轻微改善。
2. 该改善在 return 和 close 任务中均存在，但幅度有限。
3. 当前模型没有超过 naive baseline。
4. return 预测存在明显接近零均值的平滑预测现象，方向预测能力有限。
5. close 预测中 naive last-close baseline 显著更强，说明直接预测价格水平时，当前 LSTM 类模型未能充分利用价格连续性。
6. 因此，目前结果可以说明“LCT-Riesz 模块相对 Plain LSTM 有一定增益”，但尚不足以证明其具备强金融预测能力。

## 9. Next Steps

后续建议：

1. 新增 `log_return` 预测任务，使目标更平稳，也更符合金融收益率建模习惯。
2. 预测 return 或 log_return 后再还原 next close，与 naive last-close baseline 进行更公平的价格尺度比较。
3. 进行多随机种子实验，评估 LCT-Riesz 增益是否稳定，而不是依赖单次随机初始化。
4. 尝试更稳健的损失函数，例如 `HuberLoss`，降低极端波动日对训练的影响。
5. 加入滞后收益率、移动均线、滚动波动率、成交量变化率等金融特征。
6. 后续再考虑 TCN / Transformer，不宜立即堆叠复杂模型；应先确认目标定义、baseline 和特征工程是否合理。

当前实验框架已经闭环，但模型预测能力仍需通过更合理目标和更强 baseline 进一步验证。
