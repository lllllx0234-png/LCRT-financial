# 严格一维 LCRT 基线：理论定义与离散实现

## 范围与论文依据

本基线依据用户提供的正式论文 *Refining Image Edge Detection via Linear
Canonical Riesz Transforms*（SIAM manuscript `M175222`）中的 Definition 2.1、
Definition 2.4、Definition 2.5、Remark 2.6、Theorem 2.8、Equation (2.1) 和
Lemma 2.10 实现。

代码位于 `src/models/strict_lcrt_1d.py`，与历史
`src/models/lct_riesz_1d.py` 完全分离。本阶段不修改旧模块、不接入预测模型、
不加入项目 gamma 扩展。

## 原论文一维定义

设

\[
\mathbf A=\begin{pmatrix}A&B\\C&D\end{pmatrix},\qquad
AD-BC=1,\qquad B\ne0.
\]

Definition 2.5 在一维退化为 Definition 2.1 的 linear canonical Hilbert
transform：

\[
\mathcal R^{\mathbf A} f(x)
=\frac{1}{\pi}e^{-iDx^2/(2B)}
\operatorname{p.v.}\int_{\mathbb R}
\frac{f(y)e^{iAy^2/(2B)}}{x-y}\,dy.
\]

Remark 2.6 指出 Fourier 矩阵

\[
\begin{pmatrix}0&1\\-1&0\end{pmatrix}
\]

使 LCRT 退化为经典 Riesz 变换；在一维即经典 Hilbert 变换。

Theorem 2.8 进一步要求 `A=D`。在 `A=D` 且 `B!=0` 时，

\[
\mathcal L_{\mathbf A}
[\mathcal R^{\mathbf A}f](\omega)
=-i\,\operatorname{sign}(\omega/B)\,
\mathcal L_{\mathbf A}f(\omega).
\]

Lemma 2.10 给出 LCT 逆变换，Equation (2.1) 因而写成

\[
\mathcal R^{\mathbf A}f
=\mathcal L_{\mathbf A}^{-1}
\left[-i\operatorname{sign}(\omega/B)\mathcal L_{\mathbf A}f\right].
\]

## 为什么乘子定理要求 A=D

将空间域 LCRT 代入 LCT 核时会出现

\[
\exp\left(i\frac{A-D}{2B}x^2\right).
\]

只有 `A=D` 时，该依赖积分变量的二次相位才消失，剩余积分才能化为经典
Hilbert/Riesz Fourier multiplier。`A!=D` 时空间域 Definition 2.5 仍可定义，
但不能继续使用 Theorem 2.8 的固定逐频点乘子宣称严格等价。

## 严格矩阵参数化

新模块使用两参数矩阵：

\[
A=D=\cos\theta,\qquad
B=s\sin\theta,\qquad
C=-\frac{\sin\theta}{s},\qquad s>0.
\]

因此解析满足

\[
AD-BC=\cos^2\theta+\sin^2\theta=1.
\]

训练参数为：

- `theta_unconstrained`：经 sigmoid 映射到一条固定的合法 theta 分支；
- `log_scale`：通过 `s=exp(log_scale)` 保证 `s>0`。

默认初始化为 `theta=pi/2, s=1`，即 Fourier 矩阵。theta 的合法范围分为

- 正 B 分支：`theta_margin < theta < pi-theta_margin`；
- 负 B 分支：`pi+theta_margin < theta < 2*pi-theta_margin`。

初始化时由 theta 确定分支，训练过程中不跨越 `B=0` 奇点。若由于极小 scale
导致 `|B|<b_epsilon`，代码抛出清晰 `ValueError`，不钳制参数，也不悄悄替换
B。所有 theta、scale、A、B、C、D 计算均保留自动微分。

## 离散 LCT 网格与归一化

令中心化索引为

\[
\nu_n=n-\lfloor N/2\rfloor,
\]

采用 canonical 输入间隔

\[
\Delta x=\sqrt{2\pi/N},\qquad x_n=\Delta x\,\nu_n,
\]

以及 LCT 输出间隔

\[
\Delta u=|B|\Delta x,\qquad u_k=\Delta u\,\nu_k.
\]

对 quadrature-weighted 样本，离散 LCT 为

\[
L_{\mathbf A,N}f
=e^{-i\,\operatorname{sign}(B)\pi/4}
M_{DB}\,F_{\operatorname{sign}(B)}\,M_{A/B}f,
\]

其中 `M` 是论文核给出的输入/输出 chirp；`B>0` 时
`F_sign(B)` 为中心化正交 FFT，`B<0` 时为中心化正交 IFFT。该定义：

- 在 Fourier 参数下严格成为 `exp(-i*pi/4)` 乘中心化 unitary FFT；
- 具有显式、有限维精确逆变换；
- 正逆 roundtrip 不依赖拟合比例或相位；
- 保留 B 正负方向。

## multiplier、DC 与 Nyquist

严格模块固定使用论文 gamma=1 端点：

\[
m_{\mathbf A}[k]=-i\operatorname{sign}(\omega_k/B).
\]

没有 gamma 参数。离散约定为：

- DC bin：`0`；
- 偶数长度 Nyquist bin：`0`；
- 其余频点严格使用 `sign(omega/B)`，所以 B 变号时 multiplier 变号。

Nyquist 的 Hilbert 奇分量设零，是为了保持实 Fourier 输入的共轭对称性。
Fourier 矩阵下，实输入输出为实数加 float 舍入量级虚部。

一般 LCT 参数下，论文空间域公式包含复 chirp，因此即使输入为实数，LCRT
输出也通常是复数。新模块始终返回复张量，不使用 `.real` 静默丢弃合法虚部。

## 空间域 periodic principal-value 参考

小规模参考实现采用周期边界和谱配置 principal value。对周期网格，连续
`1/(pi(x-y))` 核的周期化是 cotangent PV 核。离散核写为

\[
K_N(r)=\frac{2}{N}\sum_{k=1}^{K_{\max}}
\sin\left(\frac{2\pi kr}{N}\right),
\]

其中：

- 奇数 N：`K_max=(N-1)/2`；
- 偶数 N：`K_max=N/2-1`，即显式排除 Nyquist；
- `r=0` 的对角项明确设为零，实现 principal value 的 `y=x` 排除。

随后直接计算

\[
e^{-iDx_n^2/(2B)}
\sum_{m\ne n}K_N(n-m)
f(x_m)e^{iAx_m^2/(2B)}.
\]

该实现是 O(N^2)，只用于小 N 验证。它与 multiplier 实现共享同一个明确的
周期边界、DC/Nyquist 和 canonical 网格，但不调用 FFT multiplier，也不拟合
逐点比例、全局尺度或相位。

## 与历史模块的区别

| 项目 | 严格基线 | 历史 `lct_riesz_1d.py` |
|---|---|---|
| 矩阵约束 | `A=D` 且 determinant 1 | 只保证 determinant 1 |
| 参数 | theta、s | alpha、m、q |
| multiplier | `-i sign(omega/B)` | 项目 gamma interpolation |
| gamma | 不存在，固定论文端点 | 固定或可学习 |
| q | 不存在 | 存在但在组合算子中冗余 |
| DC | 0 | 所有 gamma 无条件为 0 |
| Nyquist | 显式为 0 | 按负频率处理 |
| 输出 | 保留复数 | 上层模块取 `.real` 后融合 |
| gate/卷积/残差 | 不包含 | 包含 |

新代码没有修改历史模块、训练入口、模型工厂或 checkpoint key，因此不改变
历史 checkpoint 的加载行为。

## 最新矩阵数值验证

最新验证目录 `strict_lcrt/20260825_204341` 的最大矩阵误差为：

| 指标 | 最大误差 |
|---|---:|
| `|A-D|` | `0.0` |
| `|AD-BC-1|` | `0.0` |

## 当前不包含的内容

- 不包含项目 gamma fractional Hilbert interpolation 扩展；
- 不包含 q、gate、卷积、残差融合或预测头；
- 尚未接入 LSTM、TCN 或 Transformer；
- 未执行任何正式 BTC 训练。

## 已知离散限制

1. 空间域参考采用周期边界，不是无限实线上的非周期截断积分；边界含义必须
   在论文实验部分明确。
2. 快速离散 LCT 使用 canonical quadrature-weighted 网格；若未来输入带有真实
   时间间隔，需要显式映射采样单位。
3. `B=0` 的 identity/chirp 特殊分支不属于 Definition 2.1 和 Theorem 2.8
   的本基线范围，因此明确拒绝。
4. 一般参数下复输出是理论性质；接入实值预测网络前必须设计显式复值表示，
   不能直接取实部掩盖信息。
5. 当前数值通过只说明底层严格基线具备受控接入条件，不代表已经证明其对金融
   预测任务有效。
