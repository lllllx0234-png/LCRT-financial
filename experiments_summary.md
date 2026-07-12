# Experiments Summary

## 1. Project Goal

本项目研究一维 Learnable LCT-Riesz / LCRT 频域增强模块在 BTC 日线金融时间序列预测中的作用。早期实验围绕下一日收益率、下一日对数收益率和下一日收盘价预测展开；这些任务保留下来作为预实验和对照任务。随着实验推进，当前论文主线建议转向 `volatility_5`，即未来 5 日实现波动率预测。

这一调整的原因是：`close` 价格预测中 naive last-close baseline 极强；`return` 和 `log_return` 点预测接近零均值噪声，zero-return baseline 在误差指标上很有竞争力；而 `volatility_5` 更接近局部波动结构建模，与 LCT-Riesz / LCRT 的频域结构特征提取动机更一致。

## 2. Dataset and Targets

实验数据文件为 `data/processed/BTC_daily_train.csv`。基础输入特征为 `Open`、`High`、`Low`、`Close`、`Volume`。所有实验按时间顺序划分 train / validation / test，不进行随机划分，默认输入窗口长度为 `sequence_length = 60`。

当前支持的预测目标包括：

- `close`：下一日收盘价预测，训练时对 target 标准化，保存和评估时反标准化回原始价格尺度。
- `return`：下一日普通收益率预测。
- `log_return`：下一日对数收益率预测，`log_return_t = log(Close_t / Close_{t-1})`。
- `volatility_5`：未来 5 日对数收益率标准差，`volatility_5 = std(log_return_{t+1}, ..., log_return_{t+5})`。未来 5 日只用于构造 `y`，不进入输入 `X`。

`volatility_5` 是非负回归目标，不计算 `directional_accuracy`。

## 3. Feature Settings

基础特征为 5 个 OHLCV 特征。当前 derived features 包括：

- `log_return`
- `abs_log_return`
- `high_low_range`
- `close_open_return`
- `rolling_vol_5`
- `rolling_vol_10`
- `rolling_vol_20`
- `volume_change`

All derived features 实验使用 5 个 OHLCV + 8 个派生特征，`input_dim = 13`。Signal features 实验只使用 4 个局部变化型特征：`log_return`、`abs_log_return`、`high_low_range`、`close_open_return`，因此输入为 5 个 OHLCV + 4 个 signal features，`input_dim = 9`。

Signal features 的索引为：

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

在 dual-branch 与 residual auxiliary 模型中，LCT-Riesz 只作用于 `signal_feature_indices = [5, 6, 7, 8]`，而不是对全部输入通道做频域变换。

## 4. Models Compared

当前比较的模型包括：

- **Plain LSTM baseline**：完整输入进入 input projection + LSTM + prediction head，不启用 LCT-Riesz。
- **Single-branch LCT-Riesz + LSTM**：输入投影后先经过 Learnable LCT-Riesz，再进入 LSTM。
- **Dual-branch LCT-Riesz LSTM**：完整输入走 LSTM 主分支，signal features 走 LCT-Riesz 频域分支，二者融合后输出预测。
- **Residual auxiliary LCT-Riesz LSTM**：完整输入走 Plain LSTM 主分支，LCT-Riesz 分支只学习残差修正，形式为 `final_pred = main_pred + residual_scale * spectral_delta`。
- **Naive baselines**：包括 zero-return、zero-log-return、last-close 和 historical volatility_5。

当前重点不在证明 LCT-Riesz 已经整体最优，而在判断它更适合怎样的结构位置：直接替代主时序分支，还是作为辅助频域修正分支。

## 5. Return and Log-Return Prediction Results

### Return Prediction

| Run | Model | Epochs / Rule | RMSE | MAE | MSE | R2 | Directional Accuracy | Best Val Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs/archive/20260616_old_runs/151529` | LCT-Riesz + LSTM | 5 epoch | 0.024379 | 0.017374 | 0.000594 | -0.001564 | 0.493844 | 0.000874 |
| `outputs/archive/20260616_old_runs/153120` | Plain LSTM baseline | 5 epoch | 0.024416 | 0.017418 | 0.000596 | -0.004553 | 0.496580 | 0.000863 |
| `outputs/archive/20260616_old_runs/161430` | LCT-Riesz + LSTM | 20 epoch | 0.024611 | 0.017703 | 0.000606 | -0.020721 | 0.504788 | 0.000840 |
| `outputs/archive/20260616_old_runs/161511` | Plain LSTM baseline | 20 epoch | 0.024732 | 0.017845 | 0.000612 | -0.030760 | 0.504788 | 0.000844 |
| `outputs/naive/legacy/20260617_114424` | Naive zero-return | `y_pred = 0` | 0.024361 | 0.017236 | 0.000593 | -0.000051 | 0.000000 | N/A |

Return 任务中，LCT-Riesz 相比 Plain LSTM 只有轻微误差优势，而 naive zero-return 在 RMSE、MAE、MSE 上仍然非常强。深度模型 directional accuracy 接近 50%，说明当前收益率方向预测能力有限。

### Log-Return Prediction

| Run | Model | Rule | RMSE | MAE | MSE | MAPE | R2 | Directional Accuracy | Best Epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `outputs/log_return/lct_riesz/20260617_161624` | LCT-Riesz + LSTM | 20 epoch | 0.024589 | 0.017678 | 0.000605 | 191.102 | -0.019215 | 0.508892 | 18 |
| `outputs/log_return/baseline/20260617_161739` | Plain LSTM baseline | 20 epoch | 0.024466 | 0.017451 | 0.000599 | 160.079 | -0.009023 | 0.502052 | 16 |
| `outputs/naive/zero_log_return/20260617_161818` | Naive zero-log-return | `y_pred = 0` | 0.024357 | 0.017231 | 0.000593 | 100.000 | -0.000025 | 0.000000 | N/A |

Log-return 任务中，Plain LSTM 在误差指标上略优于 LCT-Riesz + LSTM，但 naive zero-log-return 仍然最强。因此 log_return 不适合作为当前论文主实验任务，只适合作为预实验说明收益率点预测的难度。

## 6. Close Prediction Results

| Run | Model | Rule | RMSE | MAE | MSE | MAPE | R2 | Best Val Loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `outputs/archive/20260617_old_runs/104838` | LCT-Riesz + LSTM | target scaling fixed | 30374.8 | 24962.0 | 922629046 | 25.4326 | -1.62736 | 0.018652 |
| `outputs/archive/20260617_old_runs/105015` | Plain LSTM baseline | target scaling fixed | 30396.6 | 25085.5 | 923952850 | 25.6134 | -1.63113 | 0.019631 |
| `outputs/naive/legacy/20260617_113209` | Naive last-close | `y_pred[t] = last close` | 2031.34 | 1459.96 | 4126331 | 1.72367 | 0.988249 | N/A |

Close 预测中，naive last-close baseline 显著优于两个神经网络模型。这说明直接预测价格水平时，当前 LSTM 类模型没有充分利用价格序列强连续性。因此 close 不适合作为当前方法有效性的主实验任务。

## 7. Volatility_5 Prediction Results

`volatility_5` 是当前建议的论文主线任务。下表明确纳入最新 residual auxiliary LCT-Riesz 实验结果。

| Feature Setting | Run | Model | RMSE | MAE | MSE | MAPE | R2 | Best Epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| OHLCV | `outputs/volatility_5/lct_riesz/20260617_165504` | Single-branch LCT-Riesz + LSTM | 0.011796667854544043 | 0.009461639953093114 | 0.00013916137247043278 | 75.13089729089293 | -0.19804589093828873 | 4 |
| OHLCV | `outputs/volatility_5/baseline/20260617_165615` | Plain LSTM baseline | 0.011440106519770546 | 0.007575706398803767 | 0.00013087603718369656 | 39.19606764002378 | -0.12671710394009117 | 8 |
| OHLCV | `outputs/naive/historical_volatility_5/20260617_165649` | Naive historical volatility_5 | 0.013646206488830805 | 0.00966588673854477 | 0.00018621895153580795 | 60.19194812937534 | -0.603166497324142 | N/A |
| All derived features | `outputs/volatility_5/lct_riesz_features/20260618_095354` | Single-branch LCT-Riesz + all features | 0.012691746404780548 | 0.009463998684441822 | 0.00016108042680325997 | 58.87774165582409 | -0.3867479173017905 | 16 |
| All derived features | `outputs/volatility_5/baseline_features/20260618_095605` | Plain LSTM + all features | 0.01101172102397565 | 0.007399955653602147 | 0.00012125799990986733 | 40.77986204846214 | -0.04391503156723742 | 13 |
| Signal features | `outputs/volatility_5/lct_signal_features/20260624_103450` | Single-branch LCT-Riesz + signal features | 0.014591371743682859 | 0.012617288515397042 | 0.00021290812936234656 | 102.84170007594959 | -0.8329347073959887 | 15 |
| Signal features | `outputs/volatility_5/baseline_signal_features/20260624_104709` | Plain LSTM + signal features | 0.010497984719026663 | 0.007876970764874714 | 0.00011020768316091732 | 57.57584047233924 | 0.051217592807093815 | 6 |
| Signal features | `outputs/volatility_5/dual_branch_signal_features/20260624_141601` | Dual-branch LCT-Riesz + signal features | 0.011193357521667537 | 0.008752387062488879 | 0.00012529125260787123 | 67.77192512540792 | -0.07863746737093713 | 4 |
| Signal features | `outputs/volatility_5/residual_lct_signal_features/20260625_163554` | Residual auxiliary LCT-Riesz + signal features | 0.010671519573653688 | 0.008156298901219602 | 0.00011388133001087377 | 62.123955075883345 | 0.019591018311474917 | 5 |

三组 signal features 结果的核心比较如下：

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Plain LSTM + signal features | 0.010497984719026663 | 0.007876970764874714 | 0.051217592807093815 |
| Dual-branch LCT-Riesz + signal features | 0.011193357521667537 | 0.008752387062488879 | -0.07863746737093713 |
| Residual auxiliary LCT-Riesz + signal features | 0.010671519573653688 | 0.008156298901219602 | 0.019591018311474917 |

Residual auxiliary LCT-Riesz 比 dual-branch LCT-Riesz 更好：RMSE 从 0.0111933575 降至 0.0106715196；MAE 从 0.0087523871 降至 0.0081562989；R2 从 -0.0786375 提升至 0.0195910。这说明 residual auxiliary 结构缓解了 dual-branch 直接融合带来的性能不足。

但 residual auxiliary LCT-Riesz 仍没有超过 Plain LSTM + signal features。当前不能写 LCT-Riesz 最优，只能写 residual auxiliary 结构提高了 LCT-Riesz 辅助分支的稳定性，并说明 LCT-Riesz 更适合作为辅助频域修正分支，而不是替代主时序建模分支。

## 8. Single-Branch / Dual-Branch / Residual LCT-Riesz Analysis

Single-branch LCT-Riesz 将输入特征整体送入频域增强模块，再进入 LSTM。该方式在 signal features 实验中表现较差，说明直接对全部输入通道进行频域增强可能破坏金融特征结构。

Dual-branch LCT-Riesz 将完整输入保留给 LSTM 主分支，同时只对 `signal_feature_indices = [5, 6, 7, 8]` 的局部信号特征引入 LCT-Riesz 频域分支。该结构明显优于 single-branch，但仍弱于 Plain LSTM + signal features。

Residual auxiliary LCT-Riesz 使用完整输入的 Plain LSTM 作为主分支，同时只对 `signal_feature_indices = [5, 6, 7, 8]` 的局部信号特征引入 LCT-Riesz 频域辅助分支。最终输出形式为：

```text
final_pred = main_pred + residual_scale * spectral_delta
```

本次实验中 `residual_scale = 0.01110094879`，说明 LCT-Riesz 分支确实参与了预测，但贡献幅度较小，主要作为轻量残差修正项，而不是替代主时序建模分支。这一结果支持当前论文主线表述：LCT-Riesz / LCRT 不适合直接替代主时序建模分支，更适合作为局部波动结构的辅助频域修正分支。

## 9. Learned LCT Parameters

有效 LCT-Riesz 实验中的 `learned_lct_parameters.txt` 记录了可学习 LCT 参数与对应矩阵。Residual auxiliary 行明确包含最新 learned LCT 参数。

| Run | Task | alpha | m | q | gamma | A | B | C | D | determinant | residual_scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/archive/20260616_old_runs/151529` | return, 5 epoch | 1.00965 | 1.01298 | -3.93239e-06 | 1 | -0.0153485 | 1.01287 | -0.987069 | -0.0149536 | 1.00000 | N/A |
| `outputs/archive/20260616_old_runs/161430` | return, 20 epoch | 1.00298 | 0.991464 | -7.08560e-07 | 1 | -0.00463527 | 0.991453 | -1.00860 | -0.00471473 | 1.00000 | N/A |
| `outputs/archive/20260617_old_runs/104838` | close, fixed scaling | 0.949332 | 0.977415 | -0.0100036 | 1 | 0.0777088 | 0.974321 | -1.01909 | 0.0910882 | 1.00000 | N/A |
| `outputs/log_return/lct_riesz/20260617_161624` | log_return | 1.04852 | 0.912192 | 6.49269e-06 | 1 | -0.0694563 | 0.909544 | -1.09308 | -0.0834777 | 1.00000 | N/A |
| `outputs/volatility_5/lct_riesz/20260617_165504` | volatility_5, OHLCV | 0.973821 | 1.02958 | -6.42102e-06 | 1 | 0.0423262 | 1.02871 | -0.970451 | 0.0399359 | 1.00000 | N/A |
| `outputs/volatility_5/lct_riesz_features/20260618_095354` | volatility_5, all features | 1.11495 | 0.933607 | 0.00503878 | 1 | -0.167662 | 0.918428 | -1.05286 | -0.196984 | 1.00000 | N/A |
| `outputs/volatility_5/lct_signal_features/20260624_103450` | volatility_5, signal features | 0.924780 | 1.06997 | -0.000122924 | 1 | 0.126127 | 1.06251 | -0.928078 | 0.110302 | 1.00000 | N/A |
| `outputs/volatility_5/dual_branch_signal_features/20260624_141601` | volatility_5, dual-branch | 0.985621 | 1.00989 | -6.05299e-05 | 1 | 0.0228085 | 1.00963 | -0.989957 | 0.0224252 | 1.00000 | N/A |
| `outputs/volatility_5/residual_lct_signal_features/20260625_163554` | volatility_5, residual auxiliary | 1.010118961 | 0.9873722196 | -9.806777962e-05 | 1 | -0.01569343731 | 0.9872475266 | -1.012662888 | -0.01600060239 | 1.000000036 | 0.01110094879 |

Residual auxiliary 实验中，LCT matrix determinant = 1.000000036，仍接近 1，说明可学习 LCT 参数化约束保持正常。`residual_scale = 0.01110094879` 表明频域分支被训练为小幅修正项，符合 residual auxiliary 设计目标。

## 10. Invalid / Debug Experiments

以下 close 实验明确标记为无效 / 调试实验，不纳入正式比较：

- `outputs/archive/20260616_old_runs/181022`
- `outputs/archive/20260616_old_runs/181405`

这两组 close 实验发生在 target scaling 修复前。当时 `target_type="close"` 时，输入特征已经经过 `StandardScaler` 标准化，但 target `y` 仍为原始 Close 价格，导致训练和验证损失达到 `1e9` 级别，预测曲线中模型输出接近 0，而真实 Close 位于约 60000 到 120000 的价格区间。因此这两组实验只作为调试记录，不作为有效结果。

| Run | Model | Target | RMSE | MAE | MSE | Best Val Loss | Status |
|---|---|---|---:|---:|---:|---:|---|
| `outputs/archive/20260616_old_runs/181022` | LCT-Riesz + LSTM | close | 89373.5 | 87386.9 | 7987629771 | 1048381844 | invalid |
| `outputs/archive/20260616_old_runs/181405` | Plain LSTM baseline | close | 89374.2 | 87387.5 | 7987745280 | 1048421436 | invalid |

## 11. Current Conclusion

当前实验支持以下谨慎结论：

1. `return` / `log_return` 点预测中，zero-return baseline 很强，深度模型没有稳定预测优势。
2. `close` 预测中，last-close naive baseline 明显强于神经网络模型，因此 close 不适合作为当前主实验任务。
3. `volatility_5` 更贴近局部波动结构建模，是当前更适合 LCT-Riesz / LCRT 频域增强的主任务。
4. Single-branch LCT-Riesz 在 signal features 下明显弱于 Plain LSTM，说明直接对全部输入通道进行频域增强不可取。
5. Dual-branch LCT-Riesz 明显优于 single-branch LCT-Riesz，但仍弱于 Plain LSTM + signal features。
6. Residual auxiliary LCT-Riesz 相比 dual-branch LCT-Riesz 进一步降低误差并将 R2 提升为正值，说明 residual auxiliary 结构比直接双分支融合更稳定。
7. Plain LSTM + signal features 仍是当前单次实验中的最优模型，因此不能宣称 LCT-Riesz 方法整体最优。
8. LCT-Riesz / LCRT 更适合作为局部波动结构的辅助频域修正分支，而不是替代主时序建模分支。

## 12. Next Steps

Residual auxiliary LCT-Riesz 已经完成单次实验，下一步不再把它作为“待实现模型”，而是进入稳定性验证阶段。

1. 进行多随机种子实验，建议 seeds 先使用 `42`、`2024`、`3407`。
2. 每个 seed 必须记录：`seed`、`run_dir`、`RMSE`、`MAE`、`MSE`、`MAPE`、`R2`、`best_epoch`。
3. 重点比较三类模型：
   - Plain LSTM + signal features
   - Dual-branch LCT-Riesz + signal features
   - Residual auxiliary LCT-Riesz + signal features
4. 多 seed 结果应单独汇总 `mean ± std`，不能只看单次实验结果。
5. 如果 residual auxiliary 在多个 seed 下稳定接近或超过 Plain LSTM + signal features，才适合进一步强化 LCT-Riesz 辅助频域分支有效性的论述。
