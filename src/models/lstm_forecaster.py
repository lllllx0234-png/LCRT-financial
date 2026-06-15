"""LCT-Riesz enhanced LSTM forecaster for financial time series."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from src.models.lct_riesz_1d import LearnableLCTRiesz1D


class LCTRieszLSTMForecaster(nn.Module):
    """Forecast a future financial target from a multivariate input window.

    The model projects each time step into a hidden feature space, optionally
    enhances those features along the temporal axis with Learnable LCT-Riesz,
    and passes the resulting sequence to an LSTM prediction backbone.

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
        num_layers: int = 1,
        output_dim: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        use_lct_riesz: bool = True,
        *,
        lct_alpha: float = 1.0,
        lct_m: float = 1.0,
        lct_q: float = 0.0,
        riesz_gamma: float = 1.0,
        learnable_gamma: bool = False,
        lct_gate_init: float = 0.0,
    ) -> None:
        """Initialize the projection, optional spectral block, and LSTM."""
        super().__init__()
        self._validate_dimensions(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.bidirectional = bool(bidirectional)
        self.use_lct_riesz = bool(use_lct_riesz)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        if self.use_lct_riesz:
            self.lct_riesz: nn.Module = LearnableLCTRiesz1D(
                channels=self.hidden_dim,
                alpha=lct_alpha,
                m=lct_m,
                q=lct_q,
                gamma=riesz_gamma,
                learnable_gamma=learnable_gamma,
                gate_init=lct_gate_init,
            )
        else:
            self.lct_riesz = nn.Identity()

        # PyTorch applies LSTM dropout only between stacked recurrent layers.
        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=self.bidirectional,
        )

        direction_multiplier = 2 if self.bidirectional else 1
        self.output_layer = nn.Linear(
            self.lstm_hidden_dim * direction_multiplier,
            self.output_dim,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict targets from a batch of financial sequences."""
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

        if self.use_lct_riesz:
            # LCT-Riesz operates on (batch, channels, temporal sequence).
            spectral_input = projected.transpose(1, 2).contiguous()
            enhanced = self.lct_riesz(spectral_input)
            projected = enhanced.transpose(1, 2).contiguous()

        recurrent_output, _ = self.lstm(projected)
        last_time_step = recurrent_output[:, -1, :]
        return self.output_layer(last_time_step)

    def export_lct_parameters(self) -> Optional[Dict[str, float]]:
        """Export LCT-Riesz parameters, or return None for the LSTM baseline."""
        if not self.use_lct_riesz:
            return None
        if not isinstance(self.lct_riesz, LearnableLCTRiesz1D):
            raise RuntimeError("The configured LCT-Riesz module is unavailable.")
        return self.lct_riesz.export_parameters()

    def count_parameters(self) -> int:
        """Return the number of trainable scalar parameters in the model."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


__all__ = ["LCTRieszLSTMForecaster"]
