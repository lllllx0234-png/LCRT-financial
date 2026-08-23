"""Parameter-matched causal temporal residual control for LSTM forecasting."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.residual_lct_lstm import (
    ResidualAuxiliaryLCTRieszLSTMForecaster,
)


class SharedCausalTemporalConv1D(nn.Module):
    """Enhance every signal channel with one shared causal convolution kernel."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 56,
        *,
        bias: bool = False,
        shared_across_channels: bool = True,
        causal_left_padding: int = 55,
        activation: str = "gelu",
        residual_connection: bool = True,
    ) -> None:
        """Initialize the strictly causal, parameter-matched temporal block."""
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if bias:
            raise ValueError("The parameter-matched temporal block requires bias=false.")
        if not shared_across_channels:
            raise ValueError(
                "The parameter-matched temporal block requires one shared kernel."
            )
        if causal_left_padding != kernel_size - 1:
            raise ValueError(
                "causal_left_padding must equal kernel_size - 1."
            )
        if str(activation).strip().lower() != "gelu":
            raise ValueError("The temporal block requires GELU activation.")
        if not residual_connection:
            raise ValueError("The temporal block requires a residual connection.")

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.causal_left_padding = int(causal_left_padding)
        self.shared_across_channels = True
        self.activation_name = "gelu"
        self.residual_connection = True
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            bias=False,
        )

    @property
    def receptive_field(self) -> int:
        """Return the number of current-and-past time steps used per output."""
        return self.kernel_size

    def causal_response(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shared convolution response with no right-side padding."""
        self._validate_input(x)
        batch_size, channels, sequence_length = x.shape
        flattened = x.reshape(batch_size * channels, 1, sequence_length)
        left_padded = F.pad(
            flattened,
            (self.causal_left_padding, 0),
        )
        response = self.conv(left_padded)
        return response.reshape(batch_size, channels, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the GELU-activated causal response to the original signals."""
        response = self.causal_response(x)
        return x + F.gelu(response)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate channel-first temporal input before convolution."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, channels, sequence_length).")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, received {x.shape[1]}."
            )
        if x.shape[-1] == 0:
            raise ValueError("sequence_length must be positive.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")


class ResidualAuxiliaryTemporalLSTMForecaster(
    ResidualAuxiliaryLCTRieszLSTMForecaster
):
    """Replace LCT-Riesz with a 56-parameter shared causal temporal block."""

    auxiliary_type = "causal_temporal_conv"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        lstm_hidden_dim: int,
        signal_feature_indices: Sequence[int],
        num_layers: int = 1,
        output_dim: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        use_lct_riesz: bool = False,
        *,
        spectral_hidden_dim: int | None = None,
        residual_scale_init: float = 0.0,
        temporal_kernel_size: int = 56,
        temporal_bias: bool = False,
        temporal_shared_across_channels: bool = True,
        temporal_causal_left_padding: int = 55,
        temporal_activation: str = "gelu",
        temporal_residual_connection: bool = True,
    ) -> None:
        """Initialize shared components identically to the LCT residual model."""
        if use_lct_riesz:
            raise ValueError(
                "residual_auxiliary_temporal_lstm requires use_lct_riesz=false."
            )

        # Construct the established residual layout first so every shared
        # parameter receives exactly the same seeded initialization as the LCT
        # comparator. The LCT module is then removed before this model is usable.
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            signal_feature_indices=signal_feature_indices,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            bidirectional=bidirectional,
            use_lct_riesz=True,
            spectral_hidden_dim=spectral_hidden_dim,
            residual_scale_init=residual_scale_init,
        )
        del self.lct_riesz
        self.use_lct_riesz = False
        self.temporal_block = SharedCausalTemporalConv1D(
            channels=len(self.signal_feature_indices),
            kernel_size=temporal_kernel_size,
            bias=temporal_bias,
            shared_across_channels=temporal_shared_across_channels,
            causal_left_padding=temporal_causal_left_padding,
            activation=temporal_activation,
            residual_connection=temporal_residual_connection,
        )

    @property
    def temporal_kernel_size(self) -> int:
        """Return the configured causal convolution kernel size."""
        return self.temporal_block.kernel_size

    def _enhance_auxiliary_input(
        self,
        auxiliary_input: torch.Tensor,
    ) -> torch.Tensor:
        """Apply time-domain enhancement without any LCT-Riesz operation."""
        return self.temporal_block(auxiliary_input)

    def forward_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return main prediction and a clearly named temporal auxiliary delta."""
        components = super().forward_components(x)
        temporal_delta = components.pop("spectral_delta")
        components["auxiliary_delta"] = temporal_delta
        return components


__all__ = [
    "ResidualAuxiliaryTemporalLSTMForecaster",
    "SharedCausalTemporalConv1D",
]
