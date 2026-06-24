# Experiments Summary

## 1. Project Goal

本项目研究一维 Learnable LCT-Riesz / LCRT 频域增强模块在 BTC 日线金融时间序列预测中的作用。早期实验主要围绕下一日收益率预测和下一日收盘价预测展开，并将 LCT-Riesz + LSTM 与 Plain LSTM、naive baseline 进行比较。随着实验推进，当前论文实验主线建议从 `return` / `close` 点预测，转向更适合频域结构建模的 `volatility_5` 任务，即未来 5 日实现波动率预测。

这一转向来自已有实验结果：`close` 价格预测中 naive last-close baseline 极强，当前神经网络模型难以超过；`return` 与 `log_return` 预测接近零均值噪声，zero-return baseline 在误差指标上非常有竞争力，方向预测接近随机；而 `volatility_5` 表示未来局部波动强度，更接近“局部高频波动结构建模”，与 LCT-Riesz / LCRT 的频域结构特征提取动机更一致。

因此，当前有效实验可以分为两类：`return`、`log_return`、`close` 作为预实验和对照任务保留，用于说明金融预测目标选择的难度；`volatility_5` 作为后续论文主实验任务，重点检验 LCT-Riesz 作为频域辅助结构是否能够改善 LSTM 对局部波动模式的刻画。

## 2. Dataset and Targets

实验数据文件为 `data/processed/BTC_daily_train.csv`，基础输入特征为 `Open`、`High`、`Low`、`Close`、`Volume`。所有神经网络实验均按照时间顺序划分 train / validation / test，不进行随机划分，避免时间序列信息泄漏。默认输入窗口长度为 `sequence_length = 60`。

当前数据模块已支持以下 `target_type`：

- `close`：预测下一日 `Close`。训练时对 close target 使用训练集拟合的 target scaler 标准化，评估和保存时反标准化回原始价格尺度。
- `return`：预测下一日普通收益率。
- `log_return`：预测下一日对数收益率，公式为 `log_return_t = log(Close_t / Close_{t-1})`。
- `volatility_5`：给定输入窗口截至时刻 `t`，预测未来 5 日对数收益率标准差，公式为 `volatility_5 = std(log_return_{t+1}, ..., log_return_{t+5})`。

对于 `volatility_5`，未来 5 日数据只用于构造监督目标 `y`，不进入输入 `X`，因此不引入未来信息泄漏。由于 `volatility_5` 是非负波动率水平，不是方向预测任务，评估时不计算 `directional_accuracy`。

## 3. Feature Settings

除 5 个基础 OHLCV 特征外，项目已支持以下 derived features：

- `log_return`
- `abs_log_return`
- `high_low_range`
- `close_open_return`
- `rolling_vol_5`
- `rolling_vol_10`
- `rolling_vol_20`
- `volume_change`

其中 all derived features 实验使用 5 个 OHLCV 特征加 8 个派生特征，`input_dim = 13`。signal features 实验只使用局部变化型信号特征：

- `log_return`
- `abs_log_return`
- `high_low_range`
- `close_open_return`

因此 signal features 输入为 5 个 OHLCV 特征加 4 个派生特征，`input_dim = 9`。对应特征索引为：

| Index | Feature |
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

`rolling_vol_5`、`rolling_vol_10`、`rolling_vol_20` 与 `volume_change` 更接近显式统计特征；signal features 更偏局部价格变化信号。后续双分支模型中，LCT-Riesz 只作用于 `signal_feature_indices = [5, 6, 7, 8]`，而不是对全部输入通道做频域变换。

## 4. Models Compared

当前实验比较以下模型与基准：

- **Single-branch LCT-Riesz + LSTM**：输入投影后，将序列特征送入一维 Learnable LCT-Riesz 频域增强模块，再进入 LSTM。
- **Plain LSTM baseline**：使用相同输入投影、LSTM 主干和预测头，但关闭 LCT-Riesz 模块。
- **Dual-branch LCT-Riesz LSTM**：主分支保留完整输入的 LSTM 时序建模，频域辅助分支只对 signal features 做 LCT-Riesz 变换，再与主分支表示融合。
- **Naive zero-return / zero-log-return baseline**：对 `return` 或 `log_return` 任务预测 0。
- **Naive last-close baseline**：对 `close` 任务预测输入窗口最后一天 Close。
- **Naive historical volatility_5 baseline**：对 `volatility_5` 任务使用过去 5 日 log_return 标准差预测未来 5 日波动率。

Single-branch LCT-Riesz + LSTM 与 Plain LSTM 的对比用于观察频域增强模块是否相对相同 LSTM 主干带来增益；dual-branch 模型则用于检验“频域辅助分支”是否比“对全部输入直接频域增强”的结构更合理。

## 5. Return and Log-Return Prediction Results

### Return Prediction

早期 return 实验为有效预实验，当前已归档在 `outputs/archive/20260616_old_runs/` 与 `outputs/naive/legacy/` 下。

| Run | Model | Epochs / Rule | RMSE | MAE | MSE | R2 | Directional Accuracy | Best Val Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs/archive/20260616_old_runs/151529` | LCT-Riesz + LSTM | 5 epoch | 0.024379 | 0.017374 | 0.000594 | -0.001564 | 0.493844 | 0.000874 |
| `outputs/archive/20260616_old_runs/153120` | Plain LSTM baseline | 5 epoch | 0.024416 | 0.017418 | 0.000596 | -0.004553 | 0.496580 | 0.000863 |
| `outputs/archive/20260616_old_runs/161430` | LCT-Riesz + LSTM | 20 epoch | 0.024611 | 0.017703 | 0.000606 | -0.020721 | 0.504788 | 0.000840 |
| `outputs/archive/20260616_old_runs/161511` | Plain LSTM baseline | 20 epoch | 0.024732 | 0.017845 | 0.000612 | -0.030760 | 0.504788 | 0.000844 |
| `outputs/naive/legacy/20260617_114424` | Naive zero-return | `y_pred = 0` | 0.024361 | 0.017236 | 0.000593 | -0.000051 | 0.000000 | N/A |

Return 任务中，LCT-Riesz + LSTM 相比 Plain LSTM 在 5 epoch 与 20 epoch 设置下都有轻微误差优势，但优势非常有限。Naive zero-return 在 RMSE、MAE、MSE 上仍然非常有竞争力，说明 BTC 日线收益率序列噪声较强，模型容易学习到接近 0 的平滑预测。深度模型的 directional accuracy 接近 50%，尚未形成稳定方向预测能力。

### Log-Return Prediction

后续新增的 `log_return` 任务进一步验证了收益率点预测的困难。

| Run | Model | Rule | RMSE | MAE | MSE | MAPE | R2 | Directional Accuracy | Best Epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs/log_return/lct_riesz/20260617_161624` | LCT-Riesz + LSTM | 20 epoch | 0.024589 | 0.017678 | 0.000605 | 191.102 | -0.019215 | 0.508892 | 18 |
| `outputs/log_return/baseline/20260617_161739` | Plain LSTM baseline | 20 epoch | 0.024466 | 0.017451 | 0.000599 | 160.079 | -0.009023 | 0.502052 | 16 |
| `outputs/naive/zero_log_return/20260617_161818` | Naive zero-log-return | `y_pred = 0` | 0.024357 | 0.017231 | 0.000593 | 100.000 | -0.000025 | 0.000000 | N/A |

在 log_return 任务中，Plain LSTM 在 RMSE、MAE、MSE 与 R2 上略优于 LCT-Riesz + LSTM，但 naive zero-log-return 仍然最强。LCT-Riesz + LSTM 的 directional accuracy 略高于 Plain LSTM，但两者都接近 50%，不足以构成稳定方向预测优势。因此，log_return 不适合作为当前论文主打任务，但可作为预实验说明日线收益率预测的噪声特征和 baseline 强度。

## 6. Close Prediction Results

Close 预测实验在 target scaling 修复后重新评估，最终指标均保存为原始价格尺度。

| Run | Model | Rule | RMSE | MAE | MSE | MAPE | R2 | Best Val Loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `outputs/archive/20260617_old_runs/104838` | LCT-Riesz + LSTM | target scaling fixed | 30374.8 | 24962.0 | 922629046 | 25.4326 | -1.62736 | 0.018652 |
| `outputs/archive/20260617_old_runs/105015` | Plain LSTM baseline | target scaling fixed | 30396.6 | 25085.5 | 923952850 | 25.6134 | -1.63113 | 0.019631 |
| `outputs/naive/legacy/20260617_113209` | Naive last-close | `y_pred[t] = last close` | 2031.34 | 1459.96 | 4126331 | 1.72367 | 0.988249 | N/A |

修复 target scaling 后，close 任务的 `prediction_results.csv`、`metrics.json` 和预测曲线均回到原始价格尺度。LCT-Riesz + LSTM 相比 Plain LSTM 有轻微误差优势，但幅度很小。相比之下，naive last-close baseline 显著优于两个神经网络模型，说明直接预测价格水平时，当前 LSTM 类模型没有充分利用价格序列的强连续性和局部 persistence。Close 任务应作为重要对照和失败分析保留，但不适合作为当前方法有效性的主实验任务。

## 7. Volatility_5 Prediction Results

`volatility_5` 是当前建议的论文主线任务。下表包含基础 OHLCV、all derived features、signal features 以及 dual-branch 结果。对于 signal features，优先采用新分层目录作为正式记录。

| Feature Setting | Run | Model | RMSE | MAE | MSE | MAPE | R2 | Best Epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| OHLCV | `outputs/volatility_5/lct_riesz/20260617_165504` | Single-branch LCT-Riesz + LSTM | 0.011797 | 0.009462 | 0.000139 | 75.1309 | -0.198046 | 4 |
| OHLCV | `outputs/volatility_5/baseline/20260617_165615` | Plain LSTM baseline | 0.011440 | 0.007576 | 0.000131 | 39.1961 | -0.126717 | 8 |
| OHLCV | `outputs/naive/historical_volatility_5/20260617_165649` | Naive historical volatility_5 | 0.013646 | 0.009666 | 0.000186 | 60.1919 | -0.603166 | N/A |
| All derived features | `outputs/volatility_5/lct_riesz_features/20260618_095354` | Single-branch LCT-Riesz + all features | 0.012692 | 0.009464 | 0.000161 | 58.8777 | -0.386748 | 16 |
| All derived features | `outputs/volatility_5/baseline_features/20260618_095605` | Plain LSTM + all features | 0.011012 | 0.007400 | 0.000121 | 40.7799 | -0.043915 | 13 |
| Signal features | `outputs/volatility_5/lct_signal_features/20260624_103450` | Single-branch LCT-Riesz + signal features | 0.014591 | 0.012617 | 0.000213 | 102.842 | -0.832935 | 15 |
| Signal features | `outputs/volatility_5/baseline_signal_features/20260624_104709` | Plain LSTM + signal features | 0.010498 | 0.007877 | 0.000110 | 57.5758 | 0.051218 | 6 |
| Signal features | `outputs/volatility_5/dual_branch_signal_features/20260624_141601` | Dual-branch LCT-Riesz + signal features | 0.011193 | 0.008752 | 0.000125 | 67.7719 | -0.078637 | 4 |

当前 `volatility_5` 任务中，Plain LSTM + signal features 是整体最优结果，RMSE = 0.0104979847，R2 = 0.0512176。Single-branch LCT-Riesz + signal features 表现最差，RMSE = 0.0145913717，R2 = -0.8329347，说明“将全部输入通道直接送入 LCT-Riesz 后再进入 LSTM”的结构可能破坏金融时序特征，不适合作为主模型结构。

Dual-branch LCT-Riesz + signal features 明显优于 single-branch LCT-Riesz：RMSE 从 0.0145913717 降至 0.0111933575，约下降 23.3%；R2 从 -0.8329347 改善至 -0.0786375。这说明将原始时序主分支与 LCT-Riesz 频域辅助分支分离后，能够缓解单分支直接频域增强带来的性能下降。

同时，dual-branch LCT-Riesz 也优于 naive historical volatility_5 baseline：RMSE 从 0.0136462065 降至 0.0111933575，约下降 18.0%。不过 dual-branch 仍未超过 Plain LSTM + signal features，因此不能写作“本文方法最优”。更稳妥的结论是：双分支结构显著改善了单分支 LCT-Riesz，并优于 naive volatility baseline，但仍弱于当前最强 Plain LSTM signal-feature baseline。

## 8. Single-Branch vs Dual-Branch LCT-Riesz Analysis

早期 single-branch LCT-Riesz 结构将输入投影后的全部通道直接进行频域增强，然后再送入 LSTM。这种设计在 MNIST 类图像任务中较自然，因为空间维度具有相对一致的几何含义；但金融特征中的 OHLCV、rolling statistics 与局部变化特征含义不同，直接对所有通道做同一种频域变换，可能会破坏原始金融统计结构。

Dual-branch LCT-Riesz LSTM 的动机是将“原始时序建模”和“频域辅助增强”解耦：

- 主分支：完整输入 `x` 直接进入 input projection + LSTM，保留原始金融时序信息；
- 频域辅助分支：只选取 `signal_feature_indices = [5, 6, 7, 8]`，即 `log_return`、`abs_log_return`、`high_low_range`、`close_open_return`；
- 选中的 signal features 经 LearnableLCTRiesz1D 和轻量时序建模得到 spectral representation；
- 主分支表示与频域分支表示融合后输出 `volatility_5` 预测。

Dual-branch 的实验意义不是证明 LCT-Riesz 已经超过 Plain LSTM，而是证明“分支化频域增强”比“单分支粗暴频域增强”更合理。该结果为下一阶段 residual auxiliary LCT-Riesz branch 提供了依据：LCT-Riesz 更适合作为辅助修正分支，而不是替代主时序建模分支。

## 9. Learned LCT Parameters

有效 LCT-Riesz 实验中的 `learned_lct_parameters.txt` 记录了可学习 LCT 参数与对应矩阵。当前关键实验参数如下：

| Run | Task | alpha | m | q | gamma | A | B | C | D | determinant |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/archive/20260616_old_runs/151529` | return, 5 epoch | 1.00965 | 1.01298 | -3.93239e-06 | 1 | -0.0153485 | 1.01287 | -0.987069 | -0.0149536 | 1.00000 |
| `outputs/archive/20260616_old_runs/161430` | return, 20 epoch | 1.00298 | 0.991464 | -7.08560e-07 | 1 | -0.00463527 | 0.991453 | -1.00860 | -0.00471473 | 1.00000 |
| `outputs/archive/20260617_old_runs/104838` | close, fixed scaling | 0.949332 | 0.977415 | -0.0100036 | 1 | 0.0777088 | 0.974321 | -1.01909 | 0.0910882 | 1.00000 |
| `outputs/log_return/lct_riesz/20260617_161624` | log_return | 1.04852 | 0.912192 | 6.49269e-06 | 1 | -0.0694563 | 0.909544 | -1.09308 | -0.0834777 | 1.00000 |
| `outputs/volatility_5/lct_riesz/20260617_165504` | volatility_5, OHLCV | 0.973821 | 1.02958 | -6.42102e-06 | 1 | 0.0423262 | 1.02871 | -0.970451 | 0.0399359 | 1.00000 |
| `outputs/volatility_5/lct_riesz_features/20260618_095354` | volatility_5, all features | 1.11495 | 0.933607 | 0.00503878 | 1 | -0.167662 | 0.918428 | -1.05286 | -0.196984 | 1.00000 |
| `outputs/volatility_5/lct_signal_features/20260624_103450` | volatility_5, signal features | 0.924780 | 1.06997 | -0.000122924 | 1 | 0.126127 | 1.06251 | -0.928078 | 0.110302 | 1.00000 |
| `outputs/volatility_5/dual_branch_signal_features/20260624_141601` | volatility_5, dual-branch | 0.985621 | 1.00989 | -6.05299e-05 | 1 | 0.0228085 | 1.00963 | -0.989957 | 0.0224252 | 1.00000 |

Dual-branch 实验中，LCT matrix determinant = 0.999999997，仍接近 1，说明可学习 LCT 参数化约束运行正常，没有发生数值失稳。整体来看，LCT 参数在训练后发生小幅偏移，但矩阵合法性保持良好，这对论文中的可解释性分析是有价值的。

## 10. Invalid / Debug Experiments

以下 close 实验明确标记为无效 / 调试实验，不纳入正式比较：

- `outputs/archive/20260616_old_runs/181022`
- `outputs/archive/20260616_old_runs/181405`

这两组 close 实验发生在 target scaling 修复前。当时 `target_type="close"` 时，输入特征已经经过 `StandardScaler` 标准化，但 target `y` 仍为原始 Close 价格，导致训练和验证损失达到 `1e9` 级别，预测曲线中模型输出接近 0，而真实 Close 位于约 60000 到 120000 的价格区间。因此这两组实验只作为调试记录，不作为有效结果。

| Run | Model | Target | RMSE | MAE | MSE | Best Val Loss | Status |
|---|---|---|---:|---:|---:|---:|---|
| `outputs/archive/20260616_old_runs/181022` | LCT-Riesz + LSTM | close | 89373.5 | 87386.9 | 7987629771 | 1048381844 | invalid |
| `outputs/archive/20260616_old_runs/181405` | Plain LSTM baseline | close | 89374.2 | 87387.5 | 7987745280 | 1048421436 | invalid |

此外，`experiment_index.csv` 中同时存在旧平铺目录和新分层目录记录时，本文档优先采用新分层目录作为正式记录。例如 signal features 与 dual-branch 的正式记录使用 `outputs/volatility_5/lct_signal_features/20260624_103450`、`outputs/volatility_5/baseline_signal_features/20260624_104709` 和 `outputs/volatility_5/dual_branch_signal_features/20260624_141601`。

## 11. Current Conclusion

当前实验支持以下谨慎结论：

1. 在 `return` / `log_return` 预测中，zero-return naive baseline 仍然很强，深度模型未体现稳定预测优势；directional accuracy 接近 50%，方向预测能力有限。
2. 在 `close` 预测中，last-close naive baseline 明显强于神经网络模型，因此 close 不适合作为当前方法有效性的主实验任务。
3. `volatility_5` 更贴近局部波动结构建模，是当前更适合 LCT-Riesz / LCRT 频域增强的主任务。
4. Single-branch LCT-Riesz 在 `volatility_5` 上表现不稳定，特别是在 signal features 下明显弱于 Plain LSTM，说明直接对全部输入通道进行频域增强不可取。
5. Dual-branch LCT-Riesz 明显优于 single-branch LCT-Riesz 和 naive historical volatility baseline，说明 LCT-Riesz 作为辅助频域分支具有一定价值。
6. 当前最好结果仍来自 Plain LSTM + signal features，因此现阶段不能宣称 LCT-Riesz 方法整体最优。
7. 后续研究应将 LCT-Riesz 定位为辅助频域修正分支，而非替代主时序建模分支。

总体而言，当前结果不支持夸大 LCT-Riesz 的预测能力，但支持一个更明确的研究方向：频域结构不应粗暴作用于全部金融输入，而应作为受控、轻量、可解释的辅助分支服务于局部波动建模。

## 12. Next Steps

下一阶段建议围绕 `volatility_5` 主任务继续推进，而不是同时展开 close、return、log_return 多条主线。

1. 将模型改为 residual auxiliary LCT-Riesz branch：

   `main_pred = Plain LSTM prediction`

   `spectral_delta = LCT-Riesz branch correction`

   `final_pred = main_pred + lambda * spectral_delta`

   其中 `lambda` 初始较小，使模型初始接近 Plain LSTM，避免频域分支破坏主分支。

2. 对 `volatility_5` 主任务进行多随机种子实验，验证 Plain LSTM、single-branch LCT、dual-branch LCT 的稳定性。

3. 优先优化 `volatility_5`，将 close、return、log_return 放在预实验或失败分析中，用于说明金融预测任务选择的重要性。

4. 尝试 HuberLoss 或对波动率峰值加权，以改善模型对极端波动区间的响应不足。

5. 在方法章节中强调 LCT-Riesz / LCRT 的作用是辅助提取局部波动结构，而不是直接替代全部原始金融特征。

6. 在进一步引入 TCN / Transformer 前，先确认 residual auxiliary LCT 分支是否能稳定超过 Plain LSTM + signal features。否则堆叠更复杂主干可能掩盖核心问题。

当前实验框架已经闭环，且实验主线已从收益率和价格点预测转向更合理的未来局部波动率预测；但模型预测能力仍需通过更合理的辅助分支设计、多随机种子验证和更强 baseline 进一步检验。
