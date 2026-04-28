"""Autograd-preserving diagnostics for LSP-JEPA latent trajectories."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

Reduction = Literal["mean", "none"]


def pairwise_cosine(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: Reduction = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pairwise cosine similarity for valid latent states.

    By default this returns the scalar mean of off-diagonal pairwise values.
    Set ``reduction="none"`` to return the full ``[N, N]`` matrix.
    """

    _validate_states(states)
    vectors = _valid_vectors(states, mask)
    if vectors.shape[0] < 2:
        if reduction == "none":
            return vectors.new_zeros((vectors.shape[0], vectors.shape[0]))
        return _zero_scalar(states)

    normalized = F.normalize(vectors, p=2, dim=-1, eps=eps)
    matrix = normalized.matmul(normalized.transpose(0, 1))
    if reduction == "none":
        return matrix
    if reduction == "mean":
        return _off_diagonal_mean(matrix, states)
    raise ValueError(f"Unsupported reduction: {reduction}")


def pairwise_l2(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Pairwise L2 distance for valid latent states."""

    _validate_states(states)
    vectors = _valid_vectors(states, mask)
    if vectors.shape[0] < 2:
        if reduction == "none":
            return vectors.new_zeros((vectors.shape[0], vectors.shape[0]))
        return _zero_scalar(states)

    matrix = torch.cdist(vectors, vectors, p=2)
    if reduction == "none":
        return matrix
    if reduction == "mean":
        return _off_diagonal_mean(matrix, states)
    raise ValueError(f"Unsupported reduction: {reduction}")


def per_dim_variance(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    include_values: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-dimension variance summary over valid latent entries."""

    _validate_states(states)
    variance, valid_dims = _per_dim_variance_values(states, mask)
    if bool(valid_dims.any()):
        valid_values = variance[valid_dims]
        result: dict[str, torch.Tensor] = {
            "mean": valid_values.mean(),
            "min": valid_values.min(),
            "max": valid_values.max(),
        }
    else:
        zero = _zero_scalar(states)
        result = {"mean": zero, "min": zero, "max": zero}

    if include_values:
        result["values"] = variance
    return result


def effective_rank(
    states: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Entropy-based effective rank of the masked latent covariance."""

    _validate_states(states)
    covariance = _masked_covariance(states, mask)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total.clamp_min(eps)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(eps))).sum()
    rank = torch.exp(entropy)
    return rank * (total > eps).to(dtype=rank.dtype, device=rank.device)


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


def _valid_vectors(
    states: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    element_mask = _as_element_mask(mask, states)
    safe_states = torch.where(element_mask, states, torch.zeros_like(states))
    row_mask = element_mask.any(dim=-1)
    return safe_states.reshape(-1, states.shape[-1])[row_mask.reshape(-1)]


def _off_diagonal_mean(matrix: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if matrix.shape[0] < 2:
        return _zero_scalar(reference)
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.bool)
    values = matrix[~eye]
    if values.numel() == 0:
        return _zero_scalar(reference)
    return values.mean()


def _per_dim_variance_values(
    states: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    element_mask = _as_element_mask(mask, states)
    mask_f = element_mask.to(dtype=states.dtype)
    count = mask_f.sum(dim=(0, 1))
    safe_states = torch.where(element_mask, states, torch.zeros_like(states))
    mean = safe_states.sum(dim=(0, 1)) / count.clamp_min(1.0)
    centered = torch.where(element_mask, states - mean.view(1, 1, -1), torch.zeros_like(states))
    variance = centered.square().sum(dim=(0, 1)) / count.clamp_min(1.0)
    return variance, count > 0


def _masked_covariance(
    states: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    element_mask = _as_element_mask(mask, states)
    flat_states = states.reshape(-1, states.shape[-1])
    flat_mask = element_mask.reshape(-1, states.shape[-1])
    mask_f = flat_mask.to(dtype=states.dtype)
    counts = mask_f.sum(dim=0)
    safe_states = torch.where(flat_mask, flat_states, torch.zeros_like(flat_states))
    mean = safe_states.sum(dim=0) / counts.clamp_min(1.0)
    centered = torch.where(flat_mask, flat_states - mean.view(1, -1), torch.zeros_like(flat_states))

    pair_counts = mask_f.transpose(0, 1).matmul(mask_f)
    covariance = centered.transpose(0, 1).matmul(centered) / (pair_counts - 1.0).clamp_min(1.0)
    valid_pairs = pair_counts > 1
    return torch.where(valid_pairs, covariance, torch.zeros_like(covariance))


def _zero_scalar(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


__all__ = [
    "pairwise_cosine",
    "pairwise_l2",
    "per_dim_variance",
    "effective_rank",
]
