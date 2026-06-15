"""One-dimensional learnable LCT-Riesz enhancement for financial sequences."""

from __future__ import annotations

import math

import torch
from torch import nn


LCTMatrix = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def learnable_lct_matrix(
    alpha: torch.Tensor,
    log_m: torch.Tensor,
    q: torch.Tensor,
) -> LCTMatrix:
    """Construct a unit-determinant LCT matrix from trainable parameters."""
    m = torch.exp(log_m)
    angle = alpha * (torch.pi / 2.0)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # This parameterization analytically enforces A * D - B * C = 1.
    a = m * cos_angle
    b = m * sin_angle
    c = -q * m * cos_angle - sin_angle / m
    d = -q * m * sin_angle + cos_angle / m
    return a, b, c, d


def inverse_lct_matrix(matrix: LCTMatrix) -> LCTMatrix:
    """Return the analytic inverse of a unit-determinant LCT matrix."""
    a, b, c, d = matrix
    return d, -b, -c, a


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a complex dtype suitable for a real input dtype."""
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def chirp_1d(
    length: int,
    theta: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate the differentiable chirp exp(i*pi*theta*n^2/N)."""
    if length <= 0:
        raise ValueError("length must be positive.")

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    indices = torch.arange(length, device=device, dtype=real_dtype)
    phase = (torch.pi / length) * theta.to(real_dtype) * indices.square()
    return torch.polar(torch.ones_like(phase), phase).to(dtype)


def lct_1d(
    signal: torch.Tensor,
    matrix: LCTMatrix,
    *,
    singularity_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Apply a differentiable chirp-FFT-IFFT LCT along the last dimension."""
    if signal.ndim < 1:
        raise ValueError("signal must have at least one dimension.")
    if signal.shape[-1] == 0:
        raise ValueError("signal sequence length must be positive.")

    if torch.is_complex(signal):
        transformed = signal
    else:
        transformed = signal.to(_complex_dtype(signal.dtype))

    a, b, c, d = matrix
    length = transformed.shape[-1]
    epsilon = torch.as_tensor(
        singularity_epsilon,
        device=transformed.device,
        dtype=a.dtype,
    )

    # The regular branch covers the useful learnable family around the
    # Fourier initialization (alpha=1, m=1, q=0, hence B=1).
    if bool((b.detach().abs() >= singularity_epsilon).item()):
        chirp_pre = chirp_1d(
            length,
            (a - 1.0) / b,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        chirp_frequency = chirp_1d(
            length,
            -b,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        chirp_post = chirp_1d(
            length,
            (d - 1.0) / b,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        spectrum = torch.fft.fft(transformed * chirp_pre, dim=-1)
        return torch.fft.ifft(spectrum * chirp_frequency, dim=-1) * chirp_post

    # Degenerate B=0 cases follow the same decomposition as the reference
    # implementation. Branch selection is numerical; operations remain
    # differentiable with respect to the active matrix parameters.
    safe_a = a + torch.where(a >= 0, epsilon, -epsilon)
    safe_d = d + torch.where(d >= 0, epsilon, -epsilon)
    if bool((a.detach().abs() > d.detach().abs() + singularity_epsilon).item()):
        chirp_pre = chirp_1d(
            length,
            (c + 1.0) / safe_d,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        chirp_frequency = chirp_1d(
            length,
            d,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        chirp_post = chirp_1d(
            length,
            1.0 / safe_d,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        spectrum = torch.fft.fft(transformed * chirp_pre, dim=-1)
        result = torch.fft.ifft(spectrum * chirp_frequency, dim=-1)
        return (
            torch.fft.fft(result * chirp_post, dim=-1)
            * torch.sqrt(transformed.new_tensor(-1j))
        )

    if bool((a.detach().abs() + singularity_epsilon < d.detach().abs()).item()):
        chirp_pre = chirp_1d(
            length,
            -1.0 / safe_a,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        chirp_frequency = chirp_1d(
            length,
            -a,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        chirp_post = chirp_1d(
            length,
            (c - 1.0) / safe_a,
            device=transformed.device,
            dtype=transformed.dtype,
        )
        spectrum = torch.fft.fft(
            torch.fft.ifft(transformed, dim=-1) * chirp_pre,
            dim=-1,
        )
        result = torch.fft.ifft(spectrum * chirp_frequency, dim=-1)
        return result * chirp_post * torch.sqrt(transformed.new_tensor(1j))

    return transformed * chirp_1d(
        length,
        c,
        device=transformed.device,
        dtype=transformed.dtype,
    )


def fractional_riesz_multiplier(
    length: int,
    gamma: torch.Tensor | float = 1.0,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a 1D fractional Hilbert/Riesz multiplier for FFT ordering."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("dtype must be torch.complex64 or torch.complex128.")

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    frequencies = torch.fft.fftfreq(length, device=device, dtype=real_dtype)
    signs = torch.sign(frequencies)
    order = torch.as_tensor(gamma, device=device, dtype=real_dtype)
    phase = order * (torch.pi / 2.0)

    # H_gamma(w) = cos(pi*gamma/2) - i*sign(w)*sin(pi*gamma/2).
    # At gamma=1 this is the classical -i*sign(w) Riesz/Hilbert multiplier.
    multiplier = torch.cos(phase) - 1j * signs * torch.sin(phase)
    multiplier = multiplier.to(dtype)
    return multiplier * (frequencies != 0).to(dtype)


class LearnableLCT1D(nn.Module):
    """Learn a single valid LCT matrix for the temporal axis."""

    def __init__(
        self,
        alpha: float = 1.0,
        m: float = 1.0,
        q: float = 0.0,
        singularity_epsilon: float = 1e-8,
    ) -> None:
        """Initialize the temporal LCT near the standard Fourier transform."""
        super().__init__()
        if m <= 0:
            raise ValueError("m must be positive.")
        if singularity_epsilon <= 0:
            raise ValueError("singularity_epsilon must be positive.")

        self.alpha_t = nn.Parameter(torch.tensor(float(alpha)))
        self.log_m_t = nn.Parameter(torch.tensor(math.log(m)))
        self.q_t = nn.Parameter(torch.tensor(float(q)))
        self.singularity_epsilon = float(singularity_epsilon)

    def matrix(self) -> LCTMatrix:
        """Return the current temporal LCT matrix (A, B, C, D)."""
        return learnable_lct_matrix(self.alpha_t, self.log_m_t, self.q_t)

    def forward(self, signal: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply the forward or analytic inverse temporal LCT."""
        matrix = self.matrix()
        if inverse:
            matrix = inverse_lct_matrix(matrix)
        return lct_1d(
            signal,
            matrix,
            singularity_epsilon=self.singularity_epsilon,
        )

    def export_parameters(self) -> dict[str, float]:
        """Export scalar LCT parameters and matrix entries."""
        a, b, c, d = self.matrix()
        return {
            "alpha": float(self.alpha_t.detach().cpu()),
            "m": float(torch.exp(self.log_m_t.detach()).cpu()),
            "q": float(self.q_t.detach().cpu()),
            "A": float(a.detach().cpu()),
            "B": float(b.detach().cpu()),
            "C": float(c.detach().cpu()),
            "D": float(d.detach().cpu()),
        }


class LearnableLCTRiesz1D(nn.Module):
    """Enhance multichannel financial sequences with learnable LCT-Riesz features."""

    def __init__(
        self,
        channels: int,
        *,
        alpha: float = 1.0,
        m: float = 1.0,
        q: float = 0.0,
        gamma: float = 1.0,
        learnable_gamma: bool = False,
        gate_init: float = 0.0,
        magnitude_epsilon: float = 1e-6,
        singularity_epsilon: float = 1e-8,
    ) -> None:
        """Initialize the temporal transform, fractional order, and fusion gate."""
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if magnitude_epsilon <= 0:
            raise ValueError("magnitude_epsilon must be positive.")

        self.channels = int(channels)
        self.magnitude_epsilon = float(magnitude_epsilon)
        self.lct = LearnableLCT1D(
            alpha=alpha,
            m=m,
            q=q,
            singularity_epsilon=singularity_epsilon,
        )

        gamma_tensor = torch.tensor(float(gamma))
        if learnable_gamma:
            self.gamma = nn.Parameter(gamma_tensor)
        else:
            self.register_buffer("gamma", gamma_tensor)

        self.feature_fusion = nn.Conv1d(
            in_channels=self.channels * 3,
            out_channels=self.channels,
            kernel_size=1,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an enhanced tensor with shape (batch, channels, sequence)."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch, channels, sequence_length)."
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, received {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")

        lct_spectrum = self.lct(x)
        multiplier = fractional_riesz_multiplier(
            x.shape[-1],
            self.gamma,
            device=x.device,
            dtype=lct_spectrum.dtype,
        )
        riesz_complex = self.lct(lct_spectrum * multiplier, inverse=True)
        riesz_response = riesz_complex.real.to(x.dtype)

        # The analytic-signal envelope combines the original temporal value
        # and its quadrature Riesz/Hilbert response.
        magnitude = torch.sqrt(
            x.square()
            + riesz_response.square()
            + self.magnitude_epsilon
        )
        fused = self.feature_fusion(
            torch.cat((x, riesz_response, magnitude), dim=1)
        )
        return x + self.gate * fused

    def export_parameters(self) -> dict[str, float]:
        """Export learned LCT entries and the current fractional Riesz order."""
        parameters = self.lct.export_parameters()
        parameters["gamma"] = float(self.gamma.detach().cpu())
        return parameters


__all__ = [
    "LCTMatrix",
    "LearnableLCT1D",
    "LearnableLCTRiesz1D",
    "chirp_1d",
    "fractional_riesz_multiplier",
    "inverse_lct_matrix",
    "lct_1d",
    "learnable_lct_matrix",
]
