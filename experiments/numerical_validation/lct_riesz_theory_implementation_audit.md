# 一维 LCT–Riesz 理论与实现对应审计

## 审计范围

本文档记录原论文 *Refining Image Edge Detection via Linear Canonical Riesz
Transforms*（DOI: `10.1137/25M1752225`）与项目当前一维实现之间的理论对应。
审计对象为：

- `src/models/lct_riesz_1d.py`
- `validate_lct_riesz_numerics.py`
- 已有正式实验的 `learned_lct_parameters.txt` 与对应 best checkpoint

本次工作仅修订验证、测试和结论表述，不修改正式模型数学逻辑，不修复
DC/Nyquist，不删除 `q`，不训练或重跑正式实验。

公开预印本和正式排版的定理编号可能有顺延。本文按 Definition 2.1、
Definition 2.4、Definition 2.5、Remark 2.6、空间域公式及 LCT 域乘子结论的
实际内容进行对应。

## 原论文的一维 LCRT

对

\[
\mathbf A=\begin{pmatrix}a&b\\c&d\end{pmatrix},\qquad ad-bc=1,
\qquad b\ne0,
\]

Definition 2.5 在一维退化为 linear canonical Hilbert transform：

\[
\mathcal R^{\mathbf A}f(x)
=\frac{1}{\pi}e^{-id x^2/(2b)}
\operatorname{p.v.}\int_{\mathbb R}
\frac{f(y)e^{ia y^2/(2b)}}{x-y}\,dy.
\]

其中：

- `a` 出现在输入 pre-chirp；
- `b` 决定 chirp 尺度和频率方向；
- `d` 出现在输出 de-chirp；
- `c` 不在该空间域公式中单独出现，但受 `ad-bc=1` 约束，并属于完整
  LCT 及其逆矩阵。

论文的简单 LCT 域乘子表示还要求 `a=d`。在这一条件及 `b!=0` 下，

\[
\mathcal L_{\mathbf A}(\mathcal R^{\mathbf A}f)(\omega)
=-i\,\operatorname{sign}(\omega/b)\,
\mathcal L_{\mathbf A}f(\omega),\qquad \omega\ne0.
\]

`a=d` 的作用是消去推导中的
`exp(i(a-d)x^2/(2b))` 残余二次相位。若 `a!=d`，Definition 2.5 仍可由
空间域公式定义，但不能继续用上述固定逐频点乘子声称二者严格等价。

## 当前代码实际算子

当前 learnable 矩阵参数化为：

\[
\begin{aligned}
A&=m\cos\theta, & B&=m\sin\theta,\\
C&=-qm\cos\theta-\sin\theta/m,
&D&=-qm\sin\theta+\cos\theta/m,
\end{aligned}
\qquad \theta=\pi\alpha/2.
\]

该参数化解析保证 `AD-BC=1`，但不保证 `A=D`。当前通用模块计算：

\[
x\longmapsto
\operatorname{Re}\left\{
\mathcal L_{\mathbf A}^{-1}
\left[h_\gamma\,\mathcal L_{\mathbf A}x\right]
\right\},
\]

并经 gate/residual 结构与原输入融合。因此：

- 离散 LCT 本身是同长度 chirp–FFT–IFFT–chirp 实现；
- Fourier 特例与 `exp(-i*pi/4) * FFT(x) / sqrt(N)` 一致；
- 正逆 roundtrip 接近 float64/complex128 机器精度；
- `A!=D` 时，该共轭滤波算子不能称为 Definition 2.5 的严格论文 LCRT；
- 更准确的通用名称是 **learnable LCT-domain fractional Hilbert
  interpolation/filtering**。

## gamma 是项目扩展

代码中的

\[
h_\gamma(\omega)=
\cos(\pi\gamma/2)-i\operatorname{sign}(\omega)\sin(\pi\gamma/2)
\]

不是原论文 Definition 2.4 中“fractional Riesz”的定义。原论文的
“fractional”来自 FrFT/LCT 旋转角；当前 `gamma` 是项目额外引入的 Identity
与 Hilbert 变换之间的 fractional Hilbert interpolation order。

后续论文表述应分别使用 `alpha_LCT` 和 `gamma_H`，避免将二者称为同一个
fractional order。

## DC 与 Nyquist 的问题边界

连续 Riesz/Hilbert multiplier 在零频单点的赋值不影响几乎处处意义下的
算子；离散实现则必须明确约定。

当前代码对所有 gamma 无条件令 DC 为零：

- `gamma=0` 时，非 DC 乘子为 1，但 DC 被删除，所以不是严格 Identity；
- `gamma=1` 时，DC 为零符合经典离散 Hilbert/Riesz 约定，不应标记为原论文
  LCRT 错误；
- 已有正式实验均使用 `gamma=1`，因此该 DC 约定不使历史实验失效；
- 偶数长度下，`torch.fft.fftfreq` 将 Nyquist bin 标记为负频率，当前
  `gamma=1` multiplier 在该 bin 为 `+i`。这是一项尚未显式论证的离散约定，
  尤其在模型最终取实部时需要在后续定义中明确。

本次验证只记录现状，不修改 DC 或 Nyquist 行为。

## q 的代数消去

当前矩阵可写为

\[
\mathbf A(q)=
\begin{pmatrix}1&0\\-q&1\end{pmatrix}\mathbf N.
\]

左侧 shear 对应 LCT 输出域的逐点 chirp。由于该 chirp 与当前逐点 multiplier
可交换，

\[
\mathcal L_{\mathbf A(q)}^{-1}h_\gamma
\mathcal L_{\mathbf A(q)}
=\mathcal L_{\mathbf N}^{-1}h_\gamma\mathcal L_{\mathbf N}.
\]

因此 `q` 在当前完整共轭滤波结构中代数消去。验证应预期：

- 改变 `q` 不改变完整模块输出（仅有浮点舍入量级差异）；
- `q_t.grad` 为有限值但接近数值零；
- 这是组合算子的参数不可辨识，不是 LCT 正逆实现错误；
- 历史 checkpoint 中 `q` 的微小变化不能解释为学到了有效 chirp 几何。

## 历史实验解释

仍然有效：

- 所有预测指标、checkpoint 和输出，作为“当前已实现模块”的经验结果；
- Fourier 特例、roundtrip、determinant 和梯度通路数值结果；
- `gamma=1` 实验作为 LCT-domain Hilbert filtering 的性能结果；
- gate、residual scale 以及 alpha/m 的数值行为。

需要撤回或降级：

- 将当前一般 learnable 模块表述为严格复现原论文 LCRT；
- 将项目 `gamma` 称为原论文 fractional Riesz 阶数；
- 在 `A!=D` 时引用论文乘子定理证明与 Definition 2.5 严格等价；
- 将历史 `q` 变化解释为有效学习贡献；
- 将 gamma=1 的 DC 零值描述为实现错误；
- 把性能变化唯一归因于原论文 LCRT 理论机制。

这些限制要求修订理论归因，但不要求删除已有经验结果，也不表示本次需要
重新训练。

## 分层结论

1. **离散 LCT 核心实现：A。** Fourier 特例、全局归一化/固定相位、
   roundtrip、determinant 以及 alpha/m 梯度均符合数值预期。
2. **原论文 LCRT 对应性：C。** 当前一般矩阵不保证 `A=D`，因此不能将
   通用 multiplier 共轭实现宣称为 Definition 2.5 的严格等价实现。
3. **项目 gamma 扩展：需要修订。** 需要重新定义 gamma 来源，并明确
   gamma=0 的 DC 端点以及偶数长度 Nyquist 约定。
4. **q 参数：结构冗余。** 它在当前组合算子中不可辨识，不属于 LCT 核心
   正逆错误。

## 推荐研究路线

推荐顺序为：

1. 先建立严格原论文 LCRT 基线：使用满足 `A=D, B!=0` 的矩阵与
   `-i*sign(omega/B)`，或直接实现 Definition 2.5 的空间域公式；
2. 再在严格基线上定义项目 gamma 扩展，单独说明 DC/Nyquist 约定；
3. 对严格基线、gamma 扩展、矩阵约束和 gate/residual 进行消融；
4. 历史当前实现结果保留，但统一重新命名并降低超出实现证据的理论归因。

当前阶段暂不修改正式模型数学逻辑，也不实现严格 LCRT。
