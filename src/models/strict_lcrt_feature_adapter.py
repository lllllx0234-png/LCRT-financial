"""Real-valued feature adapters for the complex strict one-dimensional LCRT."""

from __future__ import annotations

import math
from typing import Literal, Sequence, cast

import torch
from torch import nn

from src.models.strict_lcrt_1d import StrictLCRT1D


StrictLCRTRepresentation = Literal[
    "real_imag",
    "magnitude_phase",
    "input_real_imag",
]

_VALID_REPRESENTATIONS = {
    "real_imag",
    "magnitude_phase",
    "input_real_imag",
}


class StrictLCRTFeatureAdapter(nn.Module):
    """Convert selected temporal strict-LCRT responses into real features.

    Inputs use the project convention ``(batch, window, features)``. Selected
    financial features are transposed to ``(batch, selected_features, window)``
    so :class:`StrictLCRT1D` acts only along the final, temporal dimension.

    Output channel order is explicit:

    - ``real_imag``: ``[real(z), imag(z)]``;
    - ``magnitude_phase``: ``[abs(z), phase(z)]``;
    - ``input_real_imag``: ``[original_input, real(z), imag(z)]``.

    For the first two modes, ``preserve_unselected=True`` prepends unselected
    original features in their input order. For ``input_real_imag``, all
    original features are inherently retained, so preservation must be true.
    """

    def __init__(
        self,
        selected_feature_indices: Sequence[int],
        *,
        representation: StrictLCRTRepresentation = "input_real_imag",
        preserve_unselected: bool | None = None,
        theta: float = math.pi / 2.0,
        scale: float = 1.0,
        theta_margin: float = 1e-3,
        b_epsilon: float = 1e-8,
        phase_epsilon: float | None = None,
    ) -> None:
        """Initialize feature selection, representation, and strict LCRT."""
        super().__init__()
        normalized_representation = str(representation).strip().lower()
        if normalized_representation not in _VALID_REPRESENTATIONS:
            raise ValueError(
                "representation must be one of {}.".format(
                    sorted(_VALID_REPRESENTATIONS)
                )
            )

        indices = tuple(int(index) for index in selected_feature_indices)
        if not indices:
            raise ValueError("selected_feature_indices cannot be empty.")
        if any(index < 0 for index in indices):
            raise ValueError("selected_feature_indices cannot contain negatives.")
        if len(indices) != len(set(indices)):
            raise ValueError("selected_feature_indices cannot contain duplicates.")

        if preserve_unselected is None:
            resolved_preservation = (
                normalized_representation == "input_real_imag"
            )
        elif not isinstance(preserve_unselected, bool):
            raise TypeError("preserve_unselected must be a bool or None.")
        else:
            resolved_preservation = preserve_unselected
        if (
            normalized_representation == "input_real_imag"
            and not resolved_preservation
        ):
            raise ValueError(
                "input_real_imag inherently preserves every original feature; "
                "preserve_unselected cannot be false."
            )

        if phase_epsilon is not None and (
            not math.isfinite(phase_epsilon) or phase_epsilon < 0.0
        ):
            raise ValueError("phase_epsilon must be finite and non-negative.")

        self.selected_feature_indices = indices
        self.representation = cast(
            StrictLCRTRepresentation,
            normalized_representation,
        )
        self.preserve_unselected = resolved_preservation
        self.phase_epsilon = phase_epsilon
        self.register_buffer(
            "_selected_feature_index_tensor",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )
        self.strict_lcrt = StrictLCRT1D(
            theta=theta,
            scale=scale,
            theta_margin=theta_margin,
            b_epsilon=b_epsilon,
        )

    def adapted_feature_count(self, input_features: int) -> int:
        """Return the output feature count for a given input feature count."""
        self._validate_feature_count(input_features)
        selected_count = len(self.selected_feature_indices)
        if self.representation == "input_real_imag":
            return input_features + 2 * selected_count
        if self.preserve_unselected:
            return input_features + selected_count
        return 2 * selected_count

    def complex_response(self, x: torch.Tensor) -> torch.Tensor:
        """Return selected complex LCRT responses as ``(batch, window, selected)``."""
        self._validate_input(x)
        selected = x.index_select(
            dim=-1,
            index=self._selected_feature_index_tensor,
        )
        channel_first = selected.transpose(1, 2).contiguous()
        response = self.strict_lcrt(channel_first)
        return response.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a real-valued, time-major feature sequence for downstream models."""
        response = self.complex_response(x)
        if self.representation == "magnitude_phase":
            first = torch.abs(response)
            second = self._stable_phase(response, first)
        else:
            # view_as_real exposes both components together; neither component
            # is silently discarded from the resulting representation.
            cartesian = torch.view_as_real(response)
            first = cartesian[..., 0]
            second = cartesian[..., 1]

        if self.representation == "input_real_imag":
            return torch.cat((x, first, second), dim=-1)

        components = (first, second)
        if not self.preserve_unselected:
            return torch.cat(components, dim=-1)
        unselected = x.index_select(
            dim=-1,
            index=self._unselected_indices(x.shape[-1], x.device),
        )
        return torch.cat((unselected, *components), dim=-1)

    def _stable_phase(
        self,
        response: torch.Tensor,
        magnitude: torch.Tensor,
    ) -> torch.Tensor:
        """Return principal phase in [-pi, pi], defining near-zero phase as zero."""
        threshold = (
            torch.finfo(magnitude.dtype).eps
            if self.phase_epsilon is None
            else self.phase_epsilon
        )
        has_defined_phase = magnitude > threshold
        safe_response = torch.where(
            has_defined_phase,
            response,
            torch.ones((), device=response.device, dtype=response.dtype),
        )
        phase = torch.angle(safe_response)
        return torch.where(has_defined_phase, phase, torch.zeros_like(phase))

    def _unselected_indices(
        self,
        input_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return unselected feature indices in their original input order."""
        mask = torch.ones(input_features, device=device, dtype=torch.bool)
        mask[self._selected_feature_index_tensor] = False
        return torch.arange(input_features, device=device)[mask]

    def _validate_feature_count(self, input_features: int) -> None:
        """Validate selected indices against a concrete input feature count."""
        if input_features <= 0:
            raise ValueError("input feature count must be positive.")
        largest_index = max(self.selected_feature_indices)
        if largest_index >= input_features:
            raise ValueError(
                "selected_feature_indices contains out-of-range index "
                f"{largest_index}; valid range is [0, {input_features - 1}]."
            )

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate the project's batch-window-feature tensor convention."""
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, window, features).")
        if x.shape[0] == 0:
            raise ValueError("batch size must be positive.")
        if x.shape[1] == 0:
            raise ValueError("window length must be positive.")
        self._validate_feature_count(x.shape[2])
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("x must use torch.float32 or torch.float64.")
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError("x must contain only finite values.")


__all__ = ["StrictLCRTFeatureAdapter", "StrictLCRTRepresentation"]
