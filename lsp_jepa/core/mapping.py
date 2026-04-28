"""Minimal trajectory mapping contracts for LSP-JEPA.

Mapping functions align student latent states with teacher target states and
return a joint validity mask. They do not compute any alignment loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import torch


MappingStrategy = Literal["sequence", "one_to_one"]
StateTensor = Annotated[torch.Tensor, "shape: [batch, T, dim]"]
MaskTensor = Annotated[torch.Tensor, "shape: [batch, T]"]
IndexTensor = Annotated[torch.Tensor, "shape: [batch, T_mapped]"]


@dataclass
class MappingResult:
    """Mapped student/teacher trajectory pairs.

    Shape contract:
        student_states: ``torch.Tensor`` with shape ``[batch, T_mapped, dim]``.
        target_states: ``torch.Tensor`` with shape ``[batch, T_mapped, dim]``.
        mask: ``torch.Tensor`` with shape ``[batch, T_mapped]`` and bool dtype.
            Only positions where both student and teacher sides are valid are
            ``True``. Padding, missing steps, invalid steps, and sampled-out
            steps are ``False`` and must not contribute to later losses.
        student_indices: ``torch.Tensor`` with shape ``[batch, T_mapped]``
            containing gathered student step indices. Entries with
            ``mask=False`` are placeholders and must be ignored.
        target_indices: ``torch.Tensor`` with shape ``[batch, T_mapped]``
            containing gathered teacher step indices. Entries with
            ``mask=False`` are placeholders and must be ignored.
    """

    student_states: StateTensor
    target_states: StateTensor
    mask: MaskTensor
    student_indices: IndexTensor
    target_indices: IndexTensor


def map_sequence(
    student_states: StateTensor,
    target_states: StateTensor,
    student_mask: MaskTensor,
    target_mask: MaskTensor,
) -> MappingResult:
    """Map final valid student latent to final valid teacher target.

    Args:
        student_states: tensor with shape ``[batch, T_student, dim]``.
        target_states: tensor with shape ``[batch, T_target, dim]``.
        student_mask: tensor with shape ``[batch, T_student]``. Valid student
            latent steps are truthy.
        target_mask: tensor with shape ``[batch, T_target]``. Valid teacher
            target steps are truthy.

    Returns:
        ``MappingResult`` with mapped state shape ``[batch, 1, dim]`` and mask
        shape ``[batch, 1]``. Samples without at least one valid student step
        and one valid target step receive ``mask=False``.
    """

    _validate_mapping_inputs(student_states, target_states, student_mask, target_mask)
    student_valid = student_mask.to(dtype=torch.bool)
    target_valid = target_mask.to(dtype=torch.bool)

    has_student = student_valid.any(dim=1)
    has_target = target_valid.any(dim=1)
    joint_mask = (has_student & has_target).unsqueeze(1)

    student_indices = _last_valid_indices(student_valid).unsqueeze(1)
    target_indices = _last_valid_indices(target_valid).unsqueeze(1)

    mapped_student = _gather_steps(student_states, student_indices)
    mapped_target = _gather_steps(target_states, target_indices)

    return MappingResult(
        student_states=mapped_student,
        target_states=mapped_target,
        mask=joint_mask,
        student_indices=student_indices,
        target_indices=target_indices,
    )


def map_one_to_one(
    student_states: StateTensor,
    target_states: StateTensor,
    student_mask: MaskTensor,
    target_mask: MaskTensor,
) -> MappingResult:
    """Map student step ``i`` to teacher target step ``i``.

    Args:
        student_states: tensor with shape ``[batch, T_student, dim]``.
        target_states: tensor with shape ``[batch, T_target, dim]``.
        student_mask: tensor with shape ``[batch, T_student]``. Valid student
            latent steps are truthy.
        target_mask: tensor with shape ``[batch, T_target]``. Valid teacher
            target steps are truthy.

    Returns:
        ``MappingResult`` with mapped state shape ``[batch, T_mapped, dim]`` and
        mask shape ``[batch, T_mapped]``, where
        ``T_mapped = min(T_student, T_target)``. A position is valid only when
        both masks are true.
    """

    _validate_mapping_inputs(student_states, target_states, student_mask, target_mask)

    mapped_steps = min(student_states.shape[1], target_states.shape[1])
    student_valid = student_mask[:, :mapped_steps].to(dtype=torch.bool)
    target_valid = target_mask[:, :mapped_steps].to(dtype=torch.bool)
    joint_mask = student_valid & target_valid

    batch = student_states.shape[0]
    indices = torch.arange(mapped_steps, device=student_states.device).expand(batch, -1)
    target_indices = indices.to(device=target_states.device)

    return MappingResult(
        student_states=student_states[:, :mapped_steps, :],
        target_states=target_states[:, :mapped_steps, :],
        mask=joint_mask,
        student_indices=indices,
        target_indices=target_indices,
    )


def map_trajectories(
    student_states: StateTensor,
    target_states: StateTensor,
    student_mask: MaskTensor,
    target_mask: MaskTensor,
    *,
    strategy: MappingStrategy,
) -> MappingResult:
    """Dispatch minimal PR1 mapping strategies without computing loss.

    Args:
        student_states: tensor with shape ``[batch, T_student, dim]``.
        target_states: tensor with shape ``[batch, T_target, dim]``.
        student_mask: tensor with shape ``[batch, T_student]``.
        target_mask: tensor with shape ``[batch, T_target]``.
        strategy: ``"sequence"`` or ``"one_to_one"``.
    """

    if strategy == "sequence":
        return map_sequence(student_states, target_states, student_mask, target_mask)
    if strategy == "one_to_one":
        return map_one_to_one(student_states, target_states, student_mask, target_mask)
    raise ValueError(f"Unsupported mapping strategy for PR1: {strategy}")


def _validate_mapping_inputs(
    student_states: torch.Tensor,
    target_states: torch.Tensor,
    student_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    _validate_states("student_states", student_states)
    _validate_states("target_states", target_states)
    _validate_mask("student_mask", student_mask, student_states)
    _validate_mask("target_mask", target_mask, target_states)

    if student_states.shape[0] != target_states.shape[0]:
        raise ValueError("student_states and target_states must share batch size")
    if student_states.shape[2] != target_states.shape[2]:
        raise ValueError("student_states and target_states must share hidden dim")


def _validate_states(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, T, dim]")


def _validate_mask(name: str, value: torch.Tensor, states: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, T]")
    expected = states.shape[:2]
    if value.shape != expected:
        raise ValueError(
            f"{name} shape {tuple(value.shape)} must match state prefix shape "
            f"{tuple(expected)}"
        )


def _last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return final valid index per row, using zero as an ignored placeholder."""

    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    masked_positions = torch.where(mask, positions, torch.zeros_like(positions))
    return masked_positions.max(dim=1).values


def _gather_steps(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.unsqueeze(-1).expand(-1, -1, states.shape[-1])
    return states.gather(dim=1, index=gather_index)
