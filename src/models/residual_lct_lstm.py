"""Residual auxiliary LCT-Riesz LSTM forecaster for financial sequences."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from src.models.lct_riesz_1d import LearnableLCTRiesz1D


class ResidualAuxiliaryLCTRieszLSTMForecaster(nn.Module):
    """Forecast with a plain LSTM main branch plus an LCT-Riesz correction.

    The main branch receives the full financial feature tensor and produces the
    primary prediction. The spectral branch receives only selected local signal
    features, applies Learnable LCT-Riesz along the temporal axis, and predicts
    a residual correction:

        final_pred = main_pred + residual_scale * spectral_delta

    ``residual_scale`` is learnable and initialized near zero so the model
    starts close to a plain LSTM baseline.

    Expected input shape:
        ``(batch, sequence_length, input_dim)``

    Output shape:
        ``(batch, output_dim)``
    """

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
        use_lct_riesz: bool = True,
        *,
        spectral_hidden_dim: Optional[int] = None,
        residual_scale_init: float = 0.0,
        lct_alpha: float = 1.0,
        lct_m: float = 1.0,
        lct_q: float = 0.0,
        riesz_gamma: float = 1.0,
        learnable_gamma: bool = False,
        lct_gate_init: float = 1.0,
    ) -> None:
        """Initialize main prediction and auxiliary residual branches."""
        super().__init__()
        self._validate_dimensions(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )
        indices = self._validate_signal_indices(
            input_dim=input_dim,
            signal_feature_indices=signal_feature_indices,
        )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.bidirectional = bool(bidirectional)
        self.use_lct_riesz = bool(use_lct_riesz)
        self.signal_feature_indices: Tuple[int, ...] = indices
        self.spectral_hidden_dim = int(
            spectral_hidden_dim if spectral_hidden_dim is not None else hidden_dim
        )
        if self.spectral_hidden_dim <= 0:
            raise ValueError("spectral_hidden_dim must be positive.")

        self.register_buffer(
            "_signal_feature_index_tensor",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.main_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=self.bidirectional,
        )
        direction_multiplier = 2 if self.bidirectional else 1
        main_repr_dim = self.lstm_hidden_dim * direction_multiplier
        self.main_head = nn.Linear(main_repr_dim, self.output_dim)

        spectral_channels = len(indices)
        if self.use_lct_riesz:
            self.lct_riesz: nn.Module = LearnableLCTRiesz1D(
                channels=spectral_channels,
                alpha=lct_alpha,
                m=lct_m,
                q=lct_q,
                gamma=riesz_gamma,
                learnable_gamma=learnable_gamma,
                gate_init=lct_gate_init,
            )
        else:
            self.lct_riesz = nn.Identity()

        self.spectral_projection = nn.Sequential(
            nn.Linear(spectral_channels, self.spectral_hidden_dim),
            nn.LayerNorm(self.spectral_hidden_dim),
            nn.Tanh(),
        )
        self.spectral_delta_head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(self.spectral_hidden_dim, self.output_dim),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    @staticmethod
    def _validate_dimensions(
        *,
        input_dim: int,
        hidden_dim: int,
        lstm_hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        """Validate model dimensions and dropout before creating layers."""
        dimensions = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "lstm_hidden_dim": lstm_hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

    @staticmethod
    def _validate_signal_indices(
        *,
        input_dim: int,
        signal_feature_indices: Sequence[int],
    ) -> Tuple[int, ...]:
        """Validate selected signal feature indices against input_dim."""
        if not signal_feature_indices:
            raise ValueError("signal_feature_indices cannot be empty.")
        indices = tuple(int(index) for index in signal_feature_indices)
        for index in indices:
            if index < 0 or index >= input_dim:
                raise ValueError(
                    "signal_feature_indices contains out-of-range index "
                    f"{index}; valid range is [0, {input_dim - 1}]."
                )
        if len(set(indices)) != len(indices):
            raise ValueError("signal_feature_indices cannot contain duplicates.")
        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict targets using the main branch plus spectral correction."""
        return self.forward_components(x)["final_pred"]

    def forward_components(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the genuine main and spectral prediction components."""
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

        projected = self.input_projection(x)
        recurrent_output, _ = self.main_lstm(projected)
        main_repr = recurrent_output[:, -1, :]
        main_pred = self.main_head(main_repr)

        signal_features = x.index_select(
            dim=-1,
            index=self._signal_feature_index_tensor,
        )
        spectral_input = signal_features.transpose(1, 2).contiguous()
        spectral_features = self.lct_riesz(spectral_input)
        spectral_sequence = spectral_features.transpose(1, 2).contiguous()
        spectral_sequence = self.spectral_projection(spectral_sequence)
        spectral_repr = spectral_sequence.mean(dim=1)
        spectral_delta = self.spectral_delta_head(spectral_repr)

        residual_correction = self.residual_scale * spectral_delta
        final_pred = main_pred + residual_correction
        return {
            "main_pred": main_pred,
            "spectral_delta": spectral_delta,
            "residual_scale": self.residual_scale,
            "residual_correction": residual_correction,
            "final_pred": final_pred,
        }

    def export_lct_parameters(self) -> Optional[Dict[str, float]]:
        """Export LCT-Riesz parameters plus the residual correction scale."""
        if not self.use_lct_riesz:
            return None
        if not isinstance(self.lct_riesz, LearnableLCTRiesz1D):
            raise RuntimeError("The configured LCT-Riesz module is unavailable.")
        parameters = self.lct_riesz.export_parameters()
        parameters["residual_scale"] = float(
            self.residual_scale.detach().cpu()
        )
        return parameters

    def count_parameters(self) -> int:
        """Return the number of trainable scalar parameters in the model."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


__all__ = ["ResidualAuxiliaryLCTRieszLSTMForecaster"]
