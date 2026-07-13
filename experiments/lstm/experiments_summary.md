# Experiments Summary

## 1. Project Goal

鏈」鐩爺绌朵竴缁?Learnable LCT-Riesz / LCRT 棰戝煙澧炲己妯″潡鍦?BTC 鏃ョ嚎閲戣瀺鏃堕棿搴忓垪棰勬祴涓殑浣滅敤銆傛棭鏈熷疄楠屽洿缁曚笅涓€鏃ユ敹鐩婄巼銆佷笅涓€鏃ュ鏁版敹鐩婄巼鍜屼笅涓€鏃ユ敹鐩樹环棰勬祴灞曞紑锛涜繖浜涗换鍔′繚鐣欎笅鏉ヤ綔涓洪瀹為獙鍜屽鐓т换鍔°€傞殢鐫€瀹為獙鎺ㄨ繘锛屽綋鍓嶈鏂囦富绾垮缓璁浆鍚?`volatility_5`锛屽嵆鏈潵 5 鏃ュ疄鐜版尝鍔ㄧ巼棰勬祴銆?
杩欎竴璋冩暣鐨勫師鍥犳槸锛歚close` 浠锋牸棰勬祴涓?naive last-close baseline 鏋佸己锛沗return` 鍜?`log_return` 鐐归娴嬫帴杩戦浂鍧囧€煎櫔澹帮紝zero-return baseline 鍦ㄨ宸寚鏍囦笂寰堟湁绔炰簤鍔涳紱鑰?`volatility_5` 鏇存帴杩戝眬閮ㄦ尝鍔ㄧ粨鏋勫缓妯★紝涓?LCT-Riesz / LCRT 鐨勯鍩熺粨鏋勭壒寰佹彁鍙栧姩鏈烘洿涓€鑷淬€?
## 2. Dataset and Targets

瀹為獙鏁版嵁鏂囦欢涓?`data/processed/BTC_daily_train.csv`銆傚熀纭€杈撳叆鐗瑰緛涓?`Open`銆乣High`銆乣Low`銆乣Close`銆乣Volume`銆傛墍鏈夊疄楠屾寜鏃堕棿椤哄簭鍒掑垎 train / validation / test锛屼笉杩涜闅忔満鍒掑垎锛岄粯璁よ緭鍏ョ獥鍙ｉ暱搴︿负 `sequence_length = 60`銆?
褰撳墠鏀寔鐨勯娴嬬洰鏍囧寘鎷細

- `close`锛氫笅涓€鏃ユ敹鐩樹环棰勬祴锛岃缁冩椂瀵?target 鏍囧噯鍖栵紝淇濆瓨鍜岃瘎浼版椂鍙嶆爣鍑嗗寲鍥炲師濮嬩环鏍煎昂搴︺€?- `return`锛氫笅涓€鏃ユ櫘閫氭敹鐩婄巼棰勬祴銆?- `log_return`锛氫笅涓€鏃ュ鏁版敹鐩婄巼棰勬祴锛宍log_return_t = log(Close_t / Close_{t-1})`銆?- `volatility_5`锛氭湭鏉?5 鏃ュ鏁版敹鐩婄巼鏍囧噯宸紝`volatility_5 = std(log_return_{t+1}, ..., log_return_{t+5})`銆傛湭鏉?5 鏃ュ彧鐢ㄤ簬鏋勯€?`y`锛屼笉杩涘叆杈撳叆 `X`銆?
`volatility_5` 鏄潪璐熷洖褰掔洰鏍囷紝涓嶈绠?`directional_accuracy`銆?
## 3. Feature Settings

鍩虹鐗瑰緛涓?5 涓?OHLCV 鐗瑰緛銆傚綋鍓?derived features 鍖呮嫭锛?
- `log_return`
- `abs_log_return`
- `high_low_range`
- `close_open_return`
- `rolling_vol_5`
- `rolling_vol_10`
- `rolling_vol_20`
- `volume_change`

All derived features 瀹為獙浣跨敤 5 涓?OHLCV + 8 涓淳鐢熺壒寰侊紝`input_dim = 13`銆係ignal features 瀹為獙鍙娇鐢?4 涓眬閮ㄥ彉鍖栧瀷鐗瑰緛锛歚log_return`銆乣abs_log_return`銆乣high_low_range`銆乣close_open_return`锛屽洜姝よ緭鍏ヤ负 5 涓?OHLCV + 4 涓?signal features锛宍input_dim = 9`銆?
Signal features 鐨勭储寮曚负锛?
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

鍦?dual-branch 涓?residual auxiliary 妯″瀷涓紝LCT-Riesz 鍙綔鐢ㄤ簬 `signal_feature_indices = [5, 6, 7, 8]`锛岃€屼笉鏄鍏ㄩ儴杈撳叆閫氶亾鍋氶鍩熷彉鎹€?
## 4. Models Compared

褰撳墠姣旇緝鐨勬ā鍨嬪寘鎷細

- **Plain LSTM baseline**锛氬畬鏁磋緭鍏ヨ繘鍏?input projection + LSTM + prediction head锛屼笉鍚敤 LCT-Riesz銆?- **Single-branch LCT-Riesz + LSTM**锛氳緭鍏ユ姇褰卞悗鍏堢粡杩?Learnable LCT-Riesz锛屽啀杩涘叆 LSTM銆?- **Dual-branch LCT-Riesz LSTM**锛氬畬鏁磋緭鍏ヨ蛋 LSTM 涓诲垎鏀紝signal features 璧?LCT-Riesz 棰戝煙鍒嗘敮锛屼簩鑰呰瀺鍚堝悗杈撳嚭棰勬祴銆?- **Residual auxiliary LCT-Riesz LSTM**锛氬畬鏁磋緭鍏ヨ蛋 Plain LSTM 涓诲垎鏀紝LCT-Riesz 鍒嗘敮鍙涔犳畫宸慨姝ｏ紝褰㈠紡涓?`final_pred = main_pred + residual_scale * spectral_delta`銆?- **Naive baselines**锛氬寘鎷?zero-return銆亃ero-log-return銆乴ast-close 鍜?historical volatility_5銆?
褰撳墠閲嶇偣涓嶅湪璇佹槑 LCT-Riesz 宸茬粡鏁翠綋鏈€浼橈紝鑰屽湪鍒ゆ柇瀹冩洿閫傚悎鎬庢牱鐨勭粨鏋勪綅缃細鐩存帴鏇夸唬涓绘椂搴忓垎鏀紝杩樻槸浣滀负杈呭姪棰戝煙淇鍒嗘敮銆?
## 5. Return and Log-Return Prediction Results

### Return Prediction

| Run | Model | Epochs / Rule | RMSE | MAE | MSE | R2 | Directional Accuracy | Best Val Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `experiments/lstm/outputs/archive/20260616_old_runs/151529` | LCT-Riesz + LSTM | 5 epoch | 0.024379 | 0.017374 | 0.000594 | -0.001564 | 0.493844 | 0.000874 |
| `experiments/lstm/outputs/archive/20260616_old_runs/153120` | Plain LSTM baseline | 5 epoch | 0.024416 | 0.017418 | 0.000596 | -0.004553 | 0.496580 | 0.000863 |
| `experiments/lstm/outputs/archive/20260616_old_runs/161430` | LCT-Riesz + LSTM | 20 epoch | 0.024611 | 0.017703 | 0.000606 | -0.020721 | 0.504788 | 0.000840 |
| `experiments/lstm/outputs/archive/20260616_old_runs/161511` | Plain LSTM baseline | 20 epoch | 0.024732 | 0.017845 | 0.000612 | -0.030760 | 0.504788 | 0.000844 |
| `experiments/lstm/outputs/naive/legacy/20260617_114424` | Naive zero-return | `y_pred = 0` | 0.024361 | 0.017236 | 0.000593 | -0.000051 | 0.000000 | N/A |

Return 浠诲姟涓紝LCT-Riesz 鐩告瘮 Plain LSTM 鍙湁杞诲井璇樊浼樺娍锛岃€?naive zero-return 鍦?RMSE銆丮AE銆丮SE 涓婁粛鐒堕潪甯稿己銆傛繁搴︽ā鍨?directional accuracy 鎺ヨ繎 50%锛岃鏄庡綋鍓嶆敹鐩婄巼鏂瑰悜棰勬祴鑳藉姏鏈夐檺銆?
### Log-Return Prediction

| Run | Model | Rule | RMSE | MAE | MSE | MAPE | R2 | Directional Accuracy | Best Epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `experiments/lstm/outputs/log_return/lct_riesz/20260617_161624` | LCT-Riesz + LSTM | 20 epoch | 0.024589 | 0.017678 | 0.000605 | 191.102 | -0.019215 | 0.508892 | 18 |
| `experiments/lstm/outputs/log_return/baseline/20260617_161739` | Plain LSTM baseline | 20 epoch | 0.024466 | 0.017451 | 0.000599 | 160.079 | -0.009023 | 0.502052 | 16 |
| `experiments/lstm/outputs/naive/zero_log_return/20260617_161818` | Naive zero-log-return | `y_pred = 0` | 0.024357 | 0.017231 | 0.000593 | 100.000 | -0.000025 | 0.000000 | N/A |

Log-return 浠诲姟涓紝Plain LSTM 鍦ㄨ宸寚鏍囦笂鐣ヤ紭浜?LCT-Riesz + LSTM锛屼絾 naive zero-log-return 浠嶇劧鏈€寮恒€傚洜姝?log_return 涓嶉€傚悎浣滀负褰撳墠璁烘枃涓诲疄楠屼换鍔★紝鍙€傚悎浣滀负棰勫疄楠岃鏄庢敹鐩婄巼鐐归娴嬬殑闅惧害銆?
## 6. Close Prediction Results

| Run | Model | Rule | RMSE | MAE | MSE | MAPE | R2 | Best Val Loss |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `experiments/lstm/outputs/archive/20260617_old_runs/104838` | LCT-Riesz + LSTM | target scaling fixed | 30374.8 | 24962.0 | 922629046 | 25.4326 | -1.62736 | 0.018652 |
| `experiments/lstm/outputs/archive/20260617_old_runs/105015` | Plain LSTM baseline | target scaling fixed | 30396.6 | 25085.5 | 923952850 | 25.6134 | -1.63113 | 0.019631 |
| `experiments/lstm/outputs/naive/legacy/20260617_113209` | Naive last-close | `y_pred[t] = last close` | 2031.34 | 1459.96 | 4126331 | 1.72367 | 0.988249 | N/A |

Close 棰勬祴涓紝naive last-close baseline 鏄捐憲浼樹簬涓や釜绁炵粡缃戠粶妯″瀷銆傝繖璇存槑鐩存帴棰勬祴浠锋牸姘村钩鏃讹紝褰撳墠 LSTM 绫绘ā鍨嬫病鏈夊厖鍒嗗埄鐢ㄤ环鏍煎簭鍒楀己杩炵画鎬с€傚洜姝?close 涓嶉€傚悎浣滀负褰撳墠鏂规硶鏈夋晥鎬х殑涓诲疄楠屼换鍔°€?
## 7. Volatility_5 Prediction Results

`volatility_5` 鏄綋鍓嶅缓璁殑璁烘枃涓荤嚎浠诲姟銆備笅琛ㄦ槑纭撼鍏ユ渶鏂?residual auxiliary LCT-Riesz 瀹為獙缁撴灉銆?
| Feature Setting | Run | Model | RMSE | MAE | MSE | MAPE | R2 | Best Epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| OHLCV | `experiments/lstm/outputs/volatility_5/lct_riesz/20260617_165504` | Single-branch LCT-Riesz + LSTM | 0.011796667854544043 | 0.009461639953093114 | 0.00013916137247043278 | 75.13089729089293 | -0.19804589093828873 | 4 |
| OHLCV | `experiments/lstm/outputs/volatility_5/baseline/20260617_165615` | Plain LSTM baseline | 0.011440106519770546 | 0.007575706398803767 | 0.00013087603718369656 | 39.19606764002378 | -0.12671710394009117 | 8 |
| OHLCV | `experiments/lstm/outputs/naive/historical_volatility_5/20260617_165649` | Naive historical volatility_5 | 0.013646206488830805 | 0.00966588673854477 | 0.00018621895153580795 | 60.19194812937534 | -0.603166497324142 | N/A |
| All derived features | `experiments/lstm/outputs/volatility_5/lct_riesz_features/20260618_095354` | Single-branch LCT-Riesz + all features | 0.012691746404780548 | 0.009463998684441822 | 0.00016108042680325997 | 58.87774165582409 | -0.3867479173017905 | 16 |
| All derived features | `experiments/lstm/outputs/volatility_5/baseline_features/20260618_095605` | Plain LSTM + all features | 0.01101172102397565 | 0.007399955653602147 | 0.00012125799990986733 | 40.77986204846214 | -0.04391503156723742 | 13 |
| Signal features | `experiments/lstm/outputs/volatility_5/lct_signal_features/20260624_103450` | Single-branch LCT-Riesz + signal features | 0.014591371743682859 | 0.012617288515397042 | 0.00021290812936234656 | 102.84170007594959 | -0.8329347073959887 | 15 |
| Signal features | `experiments/lstm/outputs/volatility_5/baseline_signal_features/20260624_104709` | Plain LSTM + signal features | 0.010497984719026663 | 0.007876970764874714 | 0.00011020768316091732 | 57.57584047233924 | 0.051217592807093815 | 6 |
| Signal features | `experiments/lstm/outputs/volatility_5/dual_branch_signal_features/20260624_141601` | Dual-branch LCT-Riesz + signal features | 0.011193357521667537 | 0.008752387062488879 | 0.00012529125260787123 | 67.77192512540792 | -0.07863746737093713 | 4 |
| Signal features | `experiments/lstm/outputs/volatility_5/residual_lct_signal_features/20260625_163554` | Residual auxiliary LCT-Riesz + signal features | 0.010671519573653688 | 0.008156298901219602 | 0.00011388133001087377 | 62.123955075883345 | 0.019591018311474917 | 5 |

涓夌粍 signal features 缁撴灉鐨勬牳蹇冩瘮杈冨涓嬶細

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Plain LSTM + signal features | 0.010497984719026663 | 0.007876970764874714 | 0.051217592807093815 |
| Dual-branch LCT-Riesz + signal features | 0.011193357521667537 | 0.008752387062488879 | -0.07863746737093713 |
| Residual auxiliary LCT-Riesz + signal features | 0.010671519573653688 | 0.008156298901219602 | 0.019591018311474917 |

Residual auxiliary LCT-Riesz 姣?dual-branch LCT-Riesz 鏇村ソ锛歊MSE 浠?0.0111933575 闄嶈嚦 0.0106715196锛汳AE 浠?0.0087523871 闄嶈嚦 0.0081562989锛汻2 浠?-0.0786375 鎻愬崌鑷?0.0195910銆傝繖璇存槑 residual auxiliary 缁撴瀯缂撹В浜?dual-branch 鐩存帴铻嶅悎甯︽潵鐨勬€ц兘涓嶈冻銆?
浣?residual auxiliary LCT-Riesz 浠嶆病鏈夎秴杩?Plain LSTM + signal features銆傚綋鍓嶄笉鑳藉啓 LCT-Riesz 鏈€浼橈紝鍙兘鍐?residual auxiliary 缁撴瀯鎻愰珮浜?LCT-Riesz 杈呭姪鍒嗘敮鐨勭ǔ瀹氭€э紝骞惰鏄?LCT-Riesz 鏇撮€傚悎浣滀负杈呭姪棰戝煙淇鍒嗘敮锛岃€屼笉鏄浛浠ｄ富鏃跺簭寤烘ā鍒嗘敮銆?
## 8. Single-Branch / Dual-Branch / Residual LCT-Riesz Analysis

Single-branch LCT-Riesz 灏嗚緭鍏ョ壒寰佹暣浣撻€佸叆棰戝煙澧炲己妯″潡锛屽啀杩涘叆 LSTM銆傝鏂瑰紡鍦?signal features 瀹為獙涓〃鐜拌緝宸紝璇存槑鐩存帴瀵瑰叏閮ㄨ緭鍏ラ€氶亾杩涜棰戝煙澧炲己鍙兘鐮村潖閲戣瀺鐗瑰緛缁撴瀯銆?
Dual-branch LCT-Riesz 灏嗗畬鏁磋緭鍏ヤ繚鐣欑粰 LSTM 涓诲垎鏀紝鍚屾椂鍙 `signal_feature_indices = [5, 6, 7, 8]` 鐨勫眬閮ㄤ俊鍙风壒寰佸紩鍏?LCT-Riesz 棰戝煙鍒嗘敮銆傝缁撴瀯鏄庢樉浼樹簬 single-branch锛屼絾浠嶅急浜?Plain LSTM + signal features銆?
Residual auxiliary LCT-Riesz 浣跨敤瀹屾暣杈撳叆鐨?Plain LSTM 浣滀负涓诲垎鏀紝鍚屾椂鍙 `signal_feature_indices = [5, 6, 7, 8]` 鐨勫眬閮ㄤ俊鍙风壒寰佸紩鍏?LCT-Riesz 棰戝煙杈呭姪鍒嗘敮銆傛渶缁堣緭鍑哄舰寮忎负锛?
```text
final_pred = main_pred + residual_scale * spectral_delta
```

鏈瀹為獙涓?`residual_scale = 0.01110094879`锛岃鏄?LCT-Riesz 鍒嗘敮纭疄鍙備笌浜嗛娴嬶紝浣嗚础鐚箙搴﹁緝灏忥紝涓昏浣滀负杞婚噺娈嬪樊淇椤癸紝鑰屼笉鏄浛浠ｄ富鏃跺簭寤烘ā鍒嗘敮銆傝繖涓€缁撴灉鏀寔褰撳墠璁烘枃涓荤嚎琛ㄨ堪锛歀CT-Riesz / LCRT 涓嶉€傚悎鐩存帴鏇夸唬涓绘椂搴忓缓妯″垎鏀紝鏇撮€傚悎浣滀负灞€閮ㄦ尝鍔ㄧ粨鏋勭殑杈呭姪棰戝煙淇鍒嗘敮銆?
## 9. Learned LCT Parameters

鏈夋晥 LCT-Riesz 瀹為獙涓殑 `learned_lct_parameters.txt` 璁板綍浜嗗彲瀛︿範 LCT 鍙傛暟涓庡搴旂煩闃点€俁esidual auxiliary 琛屾槑纭寘鍚渶鏂?learned LCT 鍙傛暟銆?
| Run | Task | alpha | m | q | gamma | A | B | C | D | determinant | residual_scale |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `experiments/lstm/outputs/archive/20260616_old_runs/151529` | return, 5 epoch | 1.00965 | 1.01298 | -3.93239e-06 | 1 | -0.0153485 | 1.01287 | -0.987069 | -0.0149536 | 1.00000 | N/A |
| `experiments/lstm/outputs/archive/20260616_old_runs/161430` | return, 20 epoch | 1.00298 | 0.991464 | -7.08560e-07 | 1 | -0.00463527 | 0.991453 | -1.00860 | -0.00471473 | 1.00000 | N/A |
| `experiments/lstm/outputs/archive/20260617_old_runs/104838` | close, fixed scaling | 0.949332 | 0.977415 | -0.0100036 | 1 | 0.0777088 | 0.974321 | -1.01909 | 0.0910882 | 1.00000 | N/A |
| `experiments/lstm/outputs/log_return/lct_riesz/20260617_161624` | log_return | 1.04852 | 0.912192 | 6.49269e-06 | 1 | -0.0694563 | 0.909544 | -1.09308 | -0.0834777 | 1.00000 | N/A |
| `experiments/lstm/outputs/volatility_5/lct_riesz/20260617_165504` | volatility_5, OHLCV | 0.973821 | 1.02958 | -6.42102e-06 | 1 | 0.0423262 | 1.02871 | -0.970451 | 0.0399359 | 1.00000 | N/A |
| `experiments/lstm/outputs/volatility_5/lct_riesz_features/20260618_095354` | volatility_5, all features | 1.11495 | 0.933607 | 0.00503878 | 1 | -0.167662 | 0.918428 | -1.05286 | -0.196984 | 1.00000 | N/A |
| `experiments/lstm/outputs/volatility_5/lct_signal_features/20260624_103450` | volatility_5, signal features | 0.924780 | 1.06997 | -0.000122924 | 1 | 0.126127 | 1.06251 | -0.928078 | 0.110302 | 1.00000 | N/A |
| `experiments/lstm/outputs/volatility_5/dual_branch_signal_features/20260624_141601` | volatility_5, dual-branch | 0.985621 | 1.00989 | -6.05299e-05 | 1 | 0.0228085 | 1.00963 | -0.989957 | 0.0224252 | 1.00000 | N/A |
| `experiments/lstm/outputs/volatility_5/residual_lct_signal_features/20260625_163554` | volatility_5, residual auxiliary | 1.010118961 | 0.9873722196 | -9.806777962e-05 | 1 | -0.01569343731 | 0.9872475266 | -1.012662888 | -0.01600060239 | 1.000000036 | 0.01110094879 |

Residual auxiliary 瀹為獙涓紝LCT matrix determinant = 1.000000036锛屼粛鎺ヨ繎 1锛岃鏄庡彲瀛︿範 LCT 鍙傛暟鍖栫害鏉熶繚鎸佹甯搞€俙residual_scale = 0.01110094879` 琛ㄦ槑棰戝煙鍒嗘敮琚缁冧负灏忓箙淇椤癸紝绗﹀悎 residual auxiliary 璁捐鐩爣銆?
## 10. Invalid / Debug Experiments

浠ヤ笅 close 瀹為獙鏄庣‘鏍囪涓烘棤鏁?/ 璋冭瘯瀹為獙锛屼笉绾冲叆姝ｅ紡姣旇緝锛?
- `experiments/lstm/outputs/archive/20260616_old_runs/181022`
- `experiments/lstm/outputs/archive/20260616_old_runs/181405`

杩欎袱缁?close 瀹為獙鍙戠敓鍦?target scaling 淇鍓嶃€傚綋鏃?`target_type="close"` 鏃讹紝杈撳叆鐗瑰緛宸茬粡缁忚繃 `StandardScaler` 鏍囧噯鍖栵紝浣?target `y` 浠嶄负鍘熷 Close 浠锋牸锛屽鑷磋缁冨拰楠岃瘉鎹熷け杈惧埌 `1e9` 绾у埆锛岄娴嬫洸绾夸腑妯″瀷杈撳嚭鎺ヨ繎 0锛岃€岀湡瀹?Close 浣嶄簬绾?60000 鍒?120000 鐨勪环鏍煎尯闂淬€傚洜姝よ繖涓ょ粍瀹為獙鍙綔涓鸿皟璇曡褰曪紝涓嶄綔涓烘湁鏁堢粨鏋溿€?
| Run | Model | Target | RMSE | MAE | MSE | Best Val Loss | Status |
|---|---|---|---:|---:|---:|---:|---|
| `experiments/lstm/outputs/archive/20260616_old_runs/181022` | LCT-Riesz + LSTM | close | 89373.5 | 87386.9 | 7987629771 | 1048381844 | invalid |
| `experiments/lstm/outputs/archive/20260616_old_runs/181405` | Plain LSTM baseline | close | 89374.2 | 87387.5 | 7987745280 | 1048421436 | invalid |

## 11. Current Conclusion

褰撳墠瀹為獙鏀寔浠ヤ笅璋ㄦ厧缁撹锛?
1. `return` / `log_return` 鐐归娴嬩腑锛寊ero-return baseline 寰堝己锛屾繁搴︽ā鍨嬫病鏈夌ǔ瀹氶娴嬩紭鍔裤€?2. `close` 棰勬祴涓紝last-close naive baseline 鏄庢樉寮轰簬绁炵粡缃戠粶妯″瀷锛屽洜姝?close 涓嶉€傚悎浣滀负褰撳墠涓诲疄楠屼换鍔°€?3. `volatility_5` 鏇磋创杩戝眬閮ㄦ尝鍔ㄧ粨鏋勫缓妯★紝鏄綋鍓嶆洿閫傚悎 LCT-Riesz / LCRT 棰戝煙澧炲己鐨勪富浠诲姟銆?4. Single-branch LCT-Riesz 鍦?signal features 涓嬫槑鏄惧急浜?Plain LSTM锛岃鏄庣洿鎺ュ鍏ㄩ儴杈撳叆閫氶亾杩涜棰戝煙澧炲己涓嶅彲鍙栥€?5. Dual-branch LCT-Riesz 鏄庢樉浼樹簬 single-branch LCT-Riesz锛屼絾浠嶅急浜?Plain LSTM + signal features銆?6. Residual auxiliary LCT-Riesz 鐩告瘮 dual-branch LCT-Riesz 杩涗竴姝ラ檷浣庤宸苟灏?R2 鎻愬崌涓烘鍊硷紝璇存槑 residual auxiliary 缁撴瀯姣旂洿鎺ュ弻鍒嗘敮铻嶅悎鏇寸ǔ瀹氥€?7. Plain LSTM + signal features 浠嶆槸褰撳墠鍗曟瀹為獙涓殑鏈€浼樻ā鍨嬶紝鍥犳涓嶈兘瀹ｇО LCT-Riesz 鏂规硶鏁翠綋鏈€浼樸€?8. LCT-Riesz / LCRT 鏇撮€傚悎浣滀负灞€閮ㄦ尝鍔ㄧ粨鏋勭殑杈呭姪棰戝煙淇鍒嗘敮锛岃€屼笉鏄浛浠ｄ富鏃跺簭寤烘ā鍒嗘敮銆?
## 12. Next Steps

Residual auxiliary LCT-Riesz 宸茬粡瀹屾垚鍗曟瀹為獙锛屼笅涓€姝ヤ笉鍐嶆妸瀹冧綔涓衡€滃緟瀹炵幇妯″瀷鈥濓紝鑰屾槸杩涘叆绋冲畾鎬ч獙璇侀樁娈点€?
1. 杩涜澶氶殢鏈虹瀛愬疄楠岋紝寤鸿 seeds 鍏堜娇鐢?`42`銆乣2024`銆乣3407`銆?2. 姣忎釜 seed 蹇呴』璁板綍锛歚seed`銆乣run_dir`銆乣RMSE`銆乣MAE`銆乣MSE`銆乣MAPE`銆乣R2`銆乣best_epoch`銆?3. 閲嶇偣姣旇緝涓夌被妯″瀷锛?   - Plain LSTM + signal features
   - Dual-branch LCT-Riesz + signal features
   - Residual auxiliary LCT-Riesz + signal features
4. 澶?seed 缁撴灉搴斿崟鐙眹鎬?`mean 卤 std`锛屼笉鑳藉彧鐪嬪崟娆″疄楠岀粨鏋溿€?5. 濡傛灉 residual auxiliary 鍦ㄥ涓?seed 涓嬬ǔ瀹氭帴杩戞垨瓒呰繃 Plain LSTM + signal features锛屾墠閫傚悎杩涗竴姝ュ己鍖?LCT-Riesz 杈呭姪棰戝煙鍒嗘敮鏈夋晥鎬х殑璁鸿堪銆?
