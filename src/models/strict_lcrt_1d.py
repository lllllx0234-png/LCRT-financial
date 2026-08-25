"""Paper-faithful one-dimensional linear canonical Riesz transform baseline."""

from __future__ import annotations

import math

import torch
from torch import nn


StrictLCTMatrix = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the complex counterpart of a supported real or complex dtype."""
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    if dtype in (torch.float32, torch.complex64):
        return torch.complex64
    raise TypeError("signal must use float32, float64, complex64, or complex128.")


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the real counterpart of a supported real or complex dtype."""
    return torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32


def _validate_nonzero_b(b: torch.Tensor, epsilon: float) -> None:
    """Raise a clear error when the paper's nonzero-B condition is violated."""
    if epsilon <= 0:
        raise ValueError("b_epsilon must be positive.")
    detached = b.detach()
    if not bool(torch.isfinite(detached).item()):
        raise ValueError("LCT matrix entry B must be finite.")
    if bool((detached.abs() < epsilon).item()):
        raise ValueError(
            f"Strict LCRT requires |B| >= {epsilon:g}; received "
            f"B={float(detached.cpu()):.6g}."
        )


def strict_lct_matrix(
    theta: torch.Tensor,
    log_scale: torch.Tensor,
    *,
    b_epsilon: float = 1e-8,
) -> StrictLCTMatrix:
    """Construct A=D=cos(theta), B=s*sin(theta), C=-sin(theta)/s."""
    scale = torch.exp(log_scale)
    sine = torch.sin(theta)
    cosine = torch.cos(theta)
    a = cosine
    b = scale * sine
    c = -sine / scale
    d = cosine
    _validate_nonzero_b(b, b_epsilon)
    return a, b, c, d


def centered_indices(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return centered integer sample indices in FFT-shifted ordering."""
    if length <= 0:
        raise ValueError("length must be positive.")
    return torch.arange(length, device=device, dtype=dtype) - length // 2


def _centered_fourier(signal: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    """Apply an orthonormal centered FFT or IFFT along the last dimension."""
    shifted = torch.fft.ifftshift(signal, dim=-1)
    if inverse:
        transformed = torch.fft.ifft(shifted, dim=-1, norm="ortho")
    else:
        transformed = torch.fft.fft(shifted, dim=-1, norm="ortho")
    return torch.fft.fftshift(transformed, dim=-1)


def _strict_lct_factors(
    length: int,
    matrix: StrictLCTMatrix,
    *,
    device: torch.device,
    dtype: torch.dtype,
    b_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Build the input chirp, output chirp, phase, and Fourier direction."""
    a, b, _, d = matrix
    _validate_nonzero_b(b, b_epsilon)
    real_dtype = _real_dtype(dtype)
    complex_dtype = _complex_dtype(dtype)
    indices = centered_indices(length, device=device, dtype=real_dtype)
    a_value = a.to(device=device, dtype=real_dtype)
    b_value = b.to(device=device, dtype=real_dtype)
    d_value = d.to(device=device, dtype=real_dtype)
    input_phase = (torch.pi / length) * (a_value / b_value) * indices.square()
    output_phase = (torch.pi / length) * (d_value * b_value) * indices.square()
    input_chirp = torch.polar(torch.ones_like(input_phase), input_phase).to(
        complex_dtype
    )
    output_chirp = torch.polar(torch.ones_like(output_phase), output_phase).to(
        complex_dtype
    )
    positive_b = bool((b.detach() > 0).item())
    b_sign = 1.0 if positive_b else -1.0
    phase_angle = torch.as_tensor(
        -b_sign * math.pi / 4.0,
        device=device,
        dtype=real_dtype,
    )
    canonical_phase = torch.polar(
        torch.ones((), device=device, dtype=real_dtype),
        phase_angle,
    ).to(complex_dtype)
    return input_chirp, output_chirp, canonical_phase, positive_b


def strict_lct_1d(
    signal: torch.Tensor,
    matrix: StrictLCTMatrix,
    *,
    b_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Apply the paper-kernel LCT on canonical quadrature-weighted sample grids."""
    if signal.ndim < 1 or signal.shape[-1] == 0:
        raise ValueError("signal must have a nonempty final dimension.")
    complex_dtype = _complex_dtype(signal.dtype)
    transformed = signal.to(complex_dtype)
    input_chirp, output_chirp, phase, positive_b = _strict_lct_factors(
        signal.shape[-1],
        matrix,
        device=signal.device,
        dtype=signal.dtype,
        b_epsilon=b_epsilon,
    )
    fourier = _centered_fourier(
        transformed * input_chirp,
        inverse=not positive_b,
    )
    return phase * output_chirp * fourier


def inverse_strict_lct_1d(
    spectrum: torch.Tensor,
    matrix: StrictLCTMatrix,
    *,
    b_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Apply the exact finite-dimensional inverse of :func:`strict_lct_1d`."""
    if spectrum.ndim < 1 or spectrum.shape[-1] == 0:
        raise ValueError("spectrum must have a nonempty final dimension.")
    complex_dtype = _complex_dtype(spectrum.dtype)
    transformed = spectrum.to(complex_dtype)
    input_chirp, output_chirp, phase, positive_b = _strict_lct_factors(
        spectrum.shape[-1],
        matrix,
        device=spectrum.device,
        dtype=spectrum.dtype,
        b_epsilon=b_epsilon,
    )
    dechirped = transformed * output_chirp.conj() / phase
    inverted = _centered_fourier(dechirped, inverse=positive_b)
    return input_chirp.conj() * inverted


def strict_lcrt_multiplier(
    length: int,
    b: torch.Tensor | float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    b_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return the paper multiplier -i*sign(omega/B) with discrete endpoint rules."""
    if dtype not in (torch.complex64, torch.complex128):
        raise TypeError("dtype must be torch.complex64 or torch.complex128.")
    real_dtype = _real_dtype(dtype)
    b_value = torch.as_tensor(b, device=device, dtype=real_dtype)
    _validate_nonzero_b(b_value, b_epsilon)
    frequencies = torch.fft.fftshift(
        torch.fft.fftfreq(length, device=device, dtype=real_dtype)
    )
    multiplier = (-1j * torch.sign(frequencies / b_value)).to(dtype)
    multiplier = torch.where(
        frequencies == 0,
        torch.zeros((), device=device, dtype=dtype),
        multiplier,
    )
    if length % 2 == 0:
        # The centered first bin is Nyquist.  Zeroing its Hilbert component
        # preserves conjugate symmetry and therefore real Fourier-case output.
        multiplier = multiplier.clone()
        multiplier[0] = 0
    return multiplier


def periodic_pv_hilbert_1d(signal: torch.Tensor) -> torch.Tensor:
    """Evaluate the O(N^2) periodic principal-value Hilbert reference."""
    if signal.ndim < 1 or signal.shape[-1] == 0:
        raise ValueError("signal must have a nonempty final dimension.")
    complex_dtype = _complex_dtype(signal.dtype)
    real_dtype = _real_dtype(signal.dtype)
    length = signal.shape[-1]
    indices = centered_indices(length, device=signal.device, dtype=real_dtype)
    offsets = indices[:, None] - indices[None, :]
    angle = 2.0 * torch.pi * offsets / length
    positive_mode_count = (length - 1) // 2 if length % 2 else length // 2 - 1
    half_angle = angle / 2.0
    denominator = torch.sin(half_angle)
    safe_denominator = torch.where(
        offsets == 0,
        torch.ones_like(denominator),
        denominator,
    )
    sine_sum = (
        torch.sin(positive_mode_count * half_angle)
        * torch.sin((positive_mode_count + 1) * half_angle)
        / safe_denominator
    )
    kernel = (2.0 / length) * sine_sum
    kernel = torch.where(offsets == 0, torch.zeros_like(kernel), kernel)
    return torch.einsum(
        "nm,...m->...n",
        kernel.to(complex_dtype),
        signal.to(complex_dtype),
    )


def strict_lcrt_spatial_reference(
    signal: torch.Tensor,
    matrix: StrictLCTMatrix,
    *,
    sample_spacing: float | None = None,
    b_epsilon: float = 1e-8,
    a_equals_d_tolerance: float = 1e-10,
) -> torch.Tensor:
    """Discretize Definition 2.1 using a periodic principal-value kernel."""
    if signal.ndim < 1 or signal.shape[-1] == 0:
        raise ValueError("signal must have a nonempty final dimension.")
    if sample_spacing is not None and sample_spacing <= 0:
        raise ValueError("sample_spacing must be positive.")
    a, b, _, d = matrix
    _validate_nonzero_b(b, b_epsilon)
    if bool(((a - d).detach().abs() > a_equals_d_tolerance).item()):
        raise ValueError("The multiplier comparison requires the paper condition A=D.")
    real_dtype = _real_dtype(signal.dtype)
    complex_dtype = _complex_dtype(signal.dtype)
    length = signal.shape[-1]
    spacing = (
        math.sqrt(2.0 * math.pi / length)
        if sample_spacing is None
        else float(sample_spacing)
    )
    locations = centered_indices(
        length,
        device=signal.device,
        dtype=real_dtype,
    ) * spacing
    a_value = a.to(device=signal.device, dtype=real_dtype)
    b_value = b.to(device=signal.device, dtype=real_dtype)
    d_value = d.to(device=signal.device, dtype=real_dtype)
    input_phase = a_value * locations.square() / (2.0 * b_value)
    output_phase = -d_value * locations.square() / (2.0 * b_value)
    input_chirp = torch.polar(torch.ones_like(input_phase), input_phase).to(
        complex_dtype
    )
    output_chirp = torch.polar(torch.ones_like(output_phase), output_phase).to(
        complex_dtype
    )
    return output_chirp * periodic_pv_hilbert_1d(
        signal.to(complex_dtype) * input_chirp
    )


class StrictLCT1D(nn.Module):
    """Learn a two-parameter LCT restricted to the paper's A=D class."""

    def __init__(
        self,
        *,
        theta: float = math.pi / 2.0,
        scale: float = 1.0,
        theta_margin: float = 1e-3,
        b_epsilon: float = 1e-8,
    ) -> None:
        """Initialize learnable theta and log-scale on one nonzero-B branch."""
        super().__init__()
        if not math.isfinite(theta) or not 0.0 < theta < 2.0 * math.pi:
            raise ValueError("theta must lie strictly between 0 and 2*pi.")
        if not 0.0 < theta_margin < math.pi / 2.0:
            raise ValueError("theta_margin must lie in (0, pi/2).")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive.")
        if b_epsilon <= 0:
            raise ValueError("b_epsilon must be positive.")

        if theta_margin < theta < math.pi - theta_margin:
            branch = 1
            lower, upper = theta_margin, math.pi - theta_margin
        elif math.pi + theta_margin < theta < 2.0 * math.pi - theta_margin:
            branch = -1
            lower, upper = math.pi + theta_margin, 2.0 * math.pi - theta_margin
        else:
            raise ValueError(
                "theta is too close to a B=0 singularity at 0, pi, or 2*pi."
            )
        initial_b = scale * math.sin(theta)
        if abs(initial_b) < b_epsilon:
            raise ValueError(
                f"Strict LCRT requires |B| >= {b_epsilon:g}; received "
                f"B={initial_b:.6g}."
            )

        fraction = (theta - lower) / (upper - lower)
        unconstrained_theta = math.log(fraction / (1.0 - fraction))
        self.theta_unconstrained = nn.Parameter(
            torch.tensor(float(unconstrained_theta))
        )
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))
        self.register_buffer(
            "theta_branch",
            torch.tensor(branch, dtype=torch.int8),
        )
        self.theta_margin = float(theta_margin)
        self.b_epsilon = float(b_epsilon)

    def theta(self) -> torch.Tensor:
        """Map the trainable internal value to one legal nonzero-B angle branch."""
        if int(self.theta_branch.item()) > 0:
            lower = self.theta_margin
            upper = math.pi - self.theta_margin
        else:
            lower = math.pi + self.theta_margin
            upper = 2.0 * math.pi - self.theta_margin
        return lower + (upper - lower) * torch.sigmoid(self.theta_unconstrained)

    def scale(self) -> torch.Tensor:
        """Return the strictly positive learned scale s=exp(log_scale)."""
        return torch.exp(self.log_scale)

    def matrix(self) -> StrictLCTMatrix:
        """Return the current strict matrix entries A, B, C, and D."""
        return strict_lct_matrix(
            self.theta(),
            self.log_scale,
            b_epsilon=self.b_epsilon,
        )

    def forward(self, signal: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Apply the strict LCT or its exact discrete inverse."""
        if inverse:
            return inverse_strict_lct_1d(
                signal,
                self.matrix(),
                b_epsilon=self.b_epsilon,
            )
        return strict_lct_1d(
            signal,
            self.matrix(),
            b_epsilon=self.b_epsilon,
        )

    def export_parameters(self) -> dict[str, float]:
        """Export physical parameters and strict matrix entries."""
        a, b, c, d = self.matrix()
        return {
            "theta": float(self.theta().detach().cpu()),
            "s": float(self.scale().detach().cpu()),
            "A": float(a.detach().cpu()),
            "B": float(b.detach().cpu()),
            "C": float(c.detach().cpu()),
            "D": float(d.detach().cpu()),
        }


class StrictLCRT1D(nn.Module):
    """Apply Equation (2.1) with gamma fixed to the paper's Riesz multiplier."""

    def __init__(
        self,
        *,
        theta: float = math.pi / 2.0,
        scale: float = 1.0,
        theta_margin: float = 1e-3,
        b_epsilon: float = 1e-8,
    ) -> None:
        """Initialize a strict learnable LCT without gamma, q, gates, or fusion."""
        super().__init__()
        self.lct = StrictLCT1D(
            theta=theta,
            scale=scale,
            theta_margin=theta_margin,
            b_epsilon=b_epsilon,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Return the complex LCRT response without silently discarding its imaginary part."""
        if signal.ndim < 1 or signal.shape[-1] == 0:
            raise ValueError("signal must have a nonempty final dimension.")
        matrix = self.lct.matrix()
        spectrum = strict_lct_1d(
            signal,
            matrix,
            b_epsilon=self.lct.b_epsilon,
        )
        multiplier = strict_lcrt_multiplier(
            signal.shape[-1],
            matrix[1],
            device=signal.device,
            dtype=spectrum.dtype,
            b_epsilon=self.lct.b_epsilon,
        )
        return inverse_strict_lct_1d(
            spectrum * multiplier,
            matrix,
            b_epsilon=self.lct.b_epsilon,
        )

    def spatial_reference(
        self,
        signal: torch.Tensor,
        *,
        sample_spacing: float | None = None,
    ) -> torch.Tensor:
        """Evaluate the independent periodic principal-value reference."""
        return strict_lcrt_spatial_reference(
            signal,
            self.lct.matrix(),
            sample_spacing=sample_spacing,
            b_epsilon=self.lct.b_epsilon,
        )

    def export_parameters(self) -> dict[str, float]:
        """Export the strict LCT parameters; gamma and q are intentionally absent."""
        return self.lct.export_parameters()


__all__ = [
    "StrictLCT1D",
    "StrictLCRT1D",
    "StrictLCTMatrix",
    "centered_indices",
    "inverse_strict_lct_1d",
    "periodic_pv_hilbert_1d",
    "strict_lct_1d",
    "strict_lct_matrix",
    "strict_lcrt_multiplier",
    "strict_lcrt_spatial_reference",
]
