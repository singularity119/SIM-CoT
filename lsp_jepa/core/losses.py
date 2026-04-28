"""Masked alignment losses for LSP-JEPA latent state targets."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

AlignmentLoss = Literal["mse", "normalized_mse", "cosine", "smooth_l1"]


def masked_mse(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over valid latent entries.

    Shape contract:
        pred_states: ``[B, T, D]``
        target_states: ``[B, T, D]``
        mask: optional ``[B, T]`` step mask or ``[B, T, D]`` element mask

    Invalid entries contribute zero to both numerator and denominator. If the
    mask has no valid entries, this returns a finite zero scalar with a valid
    autograd path to ``pred_states``.
    """

    _validate_state_pair(pred_states, target_states)
    element_mask = _as_element_mask(mask, pred_states)
    loss = (pred_states - target_states).square()
    return _masked_mean(loss, element_mask)


def masked_normalized_mse(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE between L2-normalized latent states over valid entries.

    Zero vectors are handled by ``torch.nn.functional.normalize`` with ``eps``;
    no NaN is introduced for zero-valued student or target states.
    """

    _validate_state_pair(pred_states, target_states)
    element_mask = _as_element_mask(mask, pred_states)
    pred_safe = _zero_invalid(pred_states, element_mask)
    target_safe = _zero_invalid(target_states, element_mask)
    pred_norm = F.normalize(pred_safe, p=2, dim=-1, eps=eps)
    target_norm = F.normalize(target_safe, p=2, dim=-1, eps=eps)
    return _masked_mean((pred_norm - target_norm).square(), element_mask)


def masked_cosine_loss(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean ``1 - cosine_similarity`` over valid latent steps.

    A ``[B, T, D]`` mask zeros invalid dimensions before cosine similarity and
    counts a step as valid when at least one dimension is valid.
    """

    _validate_state_pair(pred_states, target_states)
    element_mask = _as_element_mask(mask, pred_states)
    pred_safe = _zero_invalid(pred_states, element_mask)
    target_safe = _zero_invalid(target_states, element_mask)
    cosine = F.cosine_similarity(pred_safe, target_safe, dim=-1, eps=eps)
    step_mask = element_mask.any(dim=-1)
    return _masked_mean(1.0 - cosine, step_mask)


def masked_smooth_l1(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 alignment loss over valid latent entries."""

    _validate_state_pair(pred_states, target_states)
    element_mask = _as_element_mask(mask, pred_states)
    loss = F.smooth_l1_loss(pred_states, target_states, reduction="none", beta=beta)
    return _masked_mean(loss, element_mask)


def compute_lsp_state_loss(
    pred_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    alignment: AlignmentLoss | str = "normalized_mse",
    align_weight: float | torch.Tensor = 1.0,
    smooth_l1_beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dispatch the configured LSP-State alignment loss.

    The caller owns teacher target detaching. This function intentionally does
    not detach ``pred_states`` or ``target_states``.
    """

    if alignment in {"mse", "masked_mse"}:
        loss = masked_mse(pred_states, target_states, mask)
    elif alignment in {"normalized_mse", "masked_normalized_mse"}:
        loss = masked_normalized_mse(pred_states, target_states, mask, eps=eps)
    elif alignment in {"cosine", "cosine_loss", "masked_cosine_loss"}:
        loss = masked_cosine_loss(pred_states, target_states, mask, eps=eps)
    elif alignment in {"smooth_l1", "masked_smooth_l1"}:
        loss = masked_smooth_l1(
            pred_states,
            target_states,
            mask,
            beta=smooth_l1_beta,
        )
    else:
        raise ValueError(f"Unsupported LSP alignment loss: {alignment}")

    return loss * align_weight


def _validate_state_pair(pred_states: torch.Tensor, target_states: torch.Tensor) -> None:
    if not isinstance(pred_states, torch.Tensor):
        raise TypeError("pred_states must be a torch.Tensor")
    if not isinstance(target_states, torch.Tensor):
        raise TypeError("target_states must be a torch.Tensor")
    if pred_states.ndim != 3:
        raise ValueError("pred_states must have shape [B, T, D]")
    if target_states.ndim != 3:
        raise ValueError("target_states must have shape [B, T, D]")
    if pred_states.shape != target_states.shape:
        raise ValueError(
            "pred_states and target_states must have identical shape; got "
            f"{tuple(pred_states.shape)} and {tuple(target_states.shape)}"
        )


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


def _zero_invalid(values: torch.Tensor, element_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(element_mask, values, torch.zeros_like(values))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    denom = mask.to(dtype=values.dtype).sum().clamp_min(1.0)
    return safe_values.sum() / denom


__all__ = [
    "AlignmentLoss",
    "masked_mse",
    "masked_normalized_mse",
    "masked_cosine_loss",
    "masked_smooth_l1",
    "compute_lsp_state_loss",
]
