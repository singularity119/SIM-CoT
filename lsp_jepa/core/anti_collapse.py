"""Anti-collapse regularizers for LSP-JEPA latent states."""

from __future__ import annotations

from typing import Literal

import torch

AntiCollapseMethod = Literal["none", "variance", "vicreg", "sigreg"]


def variance_regularizer(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg-style variance hinge over valid latent entries.

    Shape contract:
        states: ``[B, T, D]``
        mask: optional ``[B, T]`` step mask or ``[B, T, D]`` element mask

    Per-dimension standard deviation is computed from valid entries only. When
    the mask is all invalid, the function returns a finite zero scalar.
    """

    _validate_states(states)
    _, variance, valid_dims = _masked_per_dim_stats(states, mask)
    std = torch.sqrt(variance + eps)
    penalties = torch.relu(torch.as_tensor(target_std, device=states.device, dtype=states.dtype) - std)
    return _masked_mean(penalties, valid_dims, states)


def covariance_regularizer(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize off-diagonal covariance between latent dimensions."""

    _validate_states(states)
    covariance, valid_pairs = _masked_covariance(states, mask)
    if covariance.shape[0] <= 1:
        return _zero_scalar(states)
    eye = torch.eye(covariance.shape[0], device=states.device, dtype=torch.bool)
    off_diagonal = ~eye
    return _masked_mean(covariance.square(), valid_pairs & off_diagonal, states)


def vicreg_variance_covariance_loss(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
    return_components: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Simple VICReg-style variance/covariance anti-collapse fallback."""

    variance_loss = variance_regularizer(
        states,
        mask,
        target_std=target_std,
        eps=eps,
    )
    covariance_loss = covariance_regularizer(states, mask)
    total = variance_loss * variance_weight + covariance_loss * covariance_weight
    if return_components:
        return {
            "loss": total,
            "variance": variance_loss,
            "covariance": covariance_loss,
        }
    return total


def sigreg_loss(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    **_: object,
) -> torch.Tensor:
    """Stub for SIGReg anti-collapse regularization.

    TODO: implement the selected SIGReg formulation once the exact paper-level
    objective is fixed for this project.

    Shape contract:
        states: ``torch.Tensor`` with shape ``[B, T, D]``.
        mask: optional truthy tensor with shape ``[B, T]`` or ``[B, T, D]``.
        returns: scalar tensor loss over valid latent entries.
    """

    _validate_states(states)
    _as_element_mask(mask, states)
    raise NotImplementedError(
        "SIGReg anti-collapse loss is a stub. TODO: implement the project "
        "SIGReg objective for states [B, T, D] and mask [B, T] or [B, T, D]."
    )


def compute_anti_collapse_loss(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    method: AntiCollapseMethod | str = "vicreg",
    weight: float | torch.Tensor = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 1.0,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Dispatch anti-collapse regularization for projected latent states."""

    if method == "none":
        return _zero_scalar(states)
    if method == "variance":
        loss = variance_regularizer(states, mask, target_std=target_std, eps=eps)
    elif method == "vicreg":
        loss = vicreg_variance_covariance_loss(
            states,
            mask,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
            target_std=target_std,
            eps=eps,
        )
    elif method == "sigreg":
        loss = sigreg_loss(states, mask)
    else:
        raise ValueError(f"Unsupported anti-collapse method: {method}")
    return loss * weight


def _validate_states(states: torch.Tensor) -> None:
    if not isinstance(states, torch.Tensor):
        raise TypeError("states must be a torch.Tensor")
    if states.ndim != 3:
        raise ValueError("states must have shape [B, T, D]")


def _as_element_mask(
    mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(reference.shape, device=reference.device, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor or None")

    mask = mask.to(device=reference.device, dtype=torch.bool)
    if mask.ndim == 2:
        if mask.shape != reference.shape[:2]:
            raise ValueError(
                "2D mask must have shape [B, T] matching states; got "
                f"{tuple(mask.shape)} for states {tuple(reference.shape)}"
            )
        return mask.unsqueeze(-1).expand_as(reference)
    if mask.ndim == 3:
        if mask.shape != reference.shape:
            raise ValueError(
                "3D mask must have shape [B, T, D] matching states; got "
                f"{tuple(mask.shape)} for states {tuple(reference.shape)}"
            )
        return mask
    raise ValueError("mask must have shape [B, T] or [B, T, D]")


def _masked_per_dim_stats(
    states: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    element_mask = _as_element_mask(mask, states)
    mask_f = element_mask.to(dtype=states.dtype)
    count = mask_f.sum(dim=(0, 1))
    safe_states = torch.where(element_mask, states, torch.zeros_like(states))
    mean = safe_states.sum(dim=(0, 1)) / count.clamp_min(1.0)
    centered = torch.where(element_mask, states - mean.view(1, 1, -1), torch.zeros_like(states))
    variance = centered.square().sum(dim=(0, 1)) / count.clamp_min(1.0)
    return mean, variance, count > 0


def _masked_covariance(
    states: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    element_mask = _as_element_mask(mask, states)
    _, _, dim_valid = _masked_per_dim_stats(states, mask)

    flat_states = states.reshape(-1, states.shape[-1])
    flat_mask = element_mask.reshape(-1, states.shape[-1])
    mask_f = flat_mask.to(dtype=states.dtype)
    counts = mask_f.sum(dim=0)
    safe_states = torch.where(flat_mask, flat_states, torch.zeros_like(flat_states))
    mean = safe_states.sum(dim=0) / counts.clamp_min(1.0)
    centered = torch.where(flat_mask, flat_states - mean.view(1, -1), torch.zeros_like(flat_states))

    pair_counts = mask_f.transpose(0, 1).matmul(mask_f)
    covariance = centered.transpose(0, 1).matmul(centered) / (pair_counts - 1.0).clamp_min(1.0)
    valid_pairs = (pair_counts > 1) & dim_valid.view(-1, 1) & dim_valid.view(1, -1)
    return covariance, valid_pairs


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.shape != values.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match values shape {tuple(values.shape)}"
        )
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    denom = mask.to(dtype=values.dtype).sum().clamp_min(1.0)
    result = safe_values.sum() / denom
    if bool(mask.any()):
        return result
    return _zero_scalar(reference)


def _zero_scalar(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


__all__ = [
    "AntiCollapseMethod",
    "variance_regularizer",
    "covariance_regularizer",
    "vicreg_variance_covariance_loss",
    "sigreg_loss",
    "compute_anti_collapse_loss",
]
