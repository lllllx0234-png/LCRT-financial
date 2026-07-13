"""Plain Temporal Convolutional Network forecaster for financial sequences."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _validate_positive_int(name: str, value: int) -> None:
    """Validate a strictly positive integer constructor argument."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_dropout(dropout: float) -> None:
    """Validate dropout probability for temporal blocks."""
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be in the interval [0, 1).")


class CausalConv1d(nn.Module):
    """One-dimensional convolution with explicit left-only temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        """Initialize a strictly causal Conv1d layer."""
        super().__init__()
        _validate_positive_int("in_channels", in_channels)
        _validate_positive_int("out_channels", out_channels)
        _validate_positive_int("kernel_size", kernel_size)
        _validate_positive_int("dilation", dilation)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding_size = self.dilation * (self.kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            bias=bool(bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply left-padded causal convolution to ``(batch, channels, time)``."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, sequence_length).")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected in_channels={self.in_channels}, received {x.shape[1]}."
            )
        if x.shape[2] == 0:
            raise ValueError("sequence_length must be positive.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")

        padded = F.pad(x, (self.padding_size, 0))
        return self.conv(padded)


class TemporalBlock(nn.Module):
    """Residual TCN block with two causal convolutions at one dilation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize causal convolutions, dropout, and residual projection."""
        super().__init__()
        _validate_positive_int("in_channels", in_channels)
        _validate_positive_int("out_channels", out_channels)
        _validate_positive_int("kernel_size", kernel_size)
        _validate_positive_int("dilation", dilation)
        _validate_dropout(dropout)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv1 = CausalConv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))
        if self.in_channels == self.out_channels:
            self.downsample: nn.Module = nn.Identity()
        else:
            self.downsample = nn.Conv1d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a sequence while preserving its temporal length."""
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        if out.shape[-1] != x.shape[-1]:
            raise RuntimeError("TemporalBlock changed sequence_length unexpectedly.")
        return self.relu(out + residual)


class TCNForecaster(nn.Module):
    """Plain TCN regression forecaster for multivariate financial windows."""

    def __init__(
        self,
        input_dim: int,
        channels: Sequence[int],
        kernel_size: int,
        dropout: float,
        output_dim: int = 1,
    ) -> None:
        """Initialize stacked temporal blocks and the prediction head."""
        super().__init__()
        _validate_positive_int("input_dim", input_dim)
        _validate_positive_int("kernel_size", kernel_size)
        _validate_positive_int("output_dim", output_dim)
        _validate_dropout(dropout)
        if not channels:
            raise ValueError("channels cannot be empty.")
        for index, channel_count in enumerate(channels):
            _validate_positive_int(f"channels[{index}]", channel_count)

        self.input_dim = int(input_dim)
        self.channels = tuple(int(channel_count) for channel_count in channels)
        self.kernel_size = int(kernel_size)
        self.output_dim = int(output_dim)
        self.use_lct_riesz = False

        blocks = []
        current_channels = self.input_dim
        for index, output_channels in enumerate(self.channels):
            blocks.append(
                TemporalBlock(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    kernel_size=self.kernel_size,
                    dilation=2**index,
                    dropout=float(dropout),
                )
            )
            current_channels = output_channels
        self.temporal_network = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(self.channels[-1], self.output_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return full TCN features with shape ``(batch, channels, time)``."""
        self._validate_input(x)
        features = x.transpose(1, 2).contiguous()
        for block in self.temporal_network:
            features = block(features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict targets from ``(batch, sequence_length, input_dim)`` inputs."""
        temporal_features = self.forward_features(x)
        last_feature = temporal_features[:, :, -1]
        return self.output_layer(last_feature)

    def export_lct_parameters(self) -> Optional[dict[str, float]]:
        """Return None because this plain TCN has no LCT-Riesz parameters."""
        return None

    def count_parameters(self) -> int:
        """Return the number of trainable scalar parameters in the model."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate batch input before temporal convolution."""
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch, sequence_length, input_dim)."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {x.shape[-1]}."
            )
        if x.shape[1] == 0:
            raise ValueError("sequence_length must be positive.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")


__all__ = ["CausalConv1d", "TemporalBlock", "TCNForecaster"]
