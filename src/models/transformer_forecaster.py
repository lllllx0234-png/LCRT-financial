"""Plain Transformer Encoder forecaster for financial sequences."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _validate_positive_int(name: str, value: int) -> None:
    """Validate a strictly positive integer constructor argument."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_dropout(dropout: float) -> None:
    """Validate dropout probability."""
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be in the interval [0, 1).")


class LearnablePositionalEncoding(nn.Module):
    """Add learnable position embeddings to batch-first temporal features."""

    def __init__(
        self,
        d_model: int,
        max_sequence_length: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize learnable positional parameters."""
        super().__init__()
        _validate_positive_int("d_model", d_model)
        _validate_positive_int("max_sequence_length", max_sequence_length)
        _validate_dropout(dropout)

        self.d_model = int(d_model)
        self.max_sequence_length = int(max_sequence_length)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.max_sequence_length, self.d_model)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return position-enhanced features with the same shape as ``x``."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence_length, d_model).")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, received {x.shape[-1]}."
            )
        sequence_length = x.shape[1]
        if sequence_length == 0:
            raise ValueError("sequence_length must be positive.")
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "sequence_length cannot exceed "
                f"max_sequence_length={self.max_sequence_length}."
            )
        positions = self.position_embedding[:, :sequence_length, :]
        return self.dropout(x + positions)


class TransformerForecaster(nn.Module):
    """Plain Transformer Encoder regression forecaster for financial windows."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_sequence_length: int,
        output_dim: int = 1,
        causal_attention: bool = True,
    ) -> None:
        """Initialize projection, positional encoding, encoder, and head."""
        super().__init__()
        self._validate_config(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_sequence_length=max_sequence_length,
            output_dim=output_dim,
            causal_attention=causal_attention,
        )

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.max_sequence_length = int(max_sequence_length)
        self.output_dim = int(output_dim)
        self.causal_attention = bool(causal_attention)
        self.use_lct_riesz = False

        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = LearnablePositionalEncoding(
            d_model=self.d_model,
            max_sequence_length=self.max_sequence_length,
            dropout=float(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        self.output_layer = nn.Linear(self.d_model, self.output_dim)

    @staticmethod
    def _validate_config(
        *,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_sequence_length: int,
        output_dim: int,
        causal_attention: bool,
    ) -> None:
        """Validate constructor arguments before creating layers."""
        _validate_positive_int("input_dim", input_dim)
        _validate_positive_int("d_model", d_model)
        _validate_positive_int("nhead", nhead)
        _validate_positive_int("num_layers", num_layers)
        _validate_positive_int("dim_feedforward", dim_feedforward)
        _validate_positive_int("max_sequence_length", max_sequence_length)
        _validate_positive_int("output_dim", output_dim)
        _validate_dropout(dropout)
        if int(d_model) % int(nhead) != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if not isinstance(causal_attention, bool):
            raise TypeError("causal_attention must be a bool.")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return full sequence features with shape ``(batch, time, d_model)``."""
        self._validate_input(x)
        projected = self.input_projection(x)
        encoded = self.positional_encoding(projected)
        mask = self._build_causal_mask(
            sequence_length=encoded.shape[1],
            device=encoded.device,
        )
        return self.encoder(encoded, mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict targets from ``(batch, sequence_length, input_dim)`` inputs."""
        features = self.forward_features(x)
        last_feature = features[:, -1, :]
        return self.output_layer(last_feature)

    def export_lct_parameters(self) -> Optional[dict[str, float]]:
        """Return None because this plain Transformer has no LCT-Riesz parameters."""
        return None

    def count_parameters(self) -> int:
        """Return the number of trainable scalar parameters in the model."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _build_causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Create a bool mask whose True entries block future attention."""
        if not self.causal_attention:
            return None
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate batch input before Transformer encoding."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch, sequence_length, input_dim)."
            )
        if x.shape[0] == 0:
            raise ValueError("batch_size must be positive.")
        if x.shape[1] == 0:
            raise ValueError("sequence_length must be positive.")
        if x.shape[1] > self.max_sequence_length:
            raise ValueError(
                "sequence_length cannot exceed "
                f"max_sequence_length={self.max_sequence_length}."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, received {x.shape[-1]}."
            )
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")


__all__ = ["LearnablePositionalEncoding", "TransformerForecaster"]
