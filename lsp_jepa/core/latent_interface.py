"""Typed data contracts for LSP-JEPA core code.

The core package is host-agnostic: host-specific code must convert its native
batch/model outputs into these dataclasses before LSP-JEPA logic consumes them.
All masks mark valid positions with ``True`` and invalid/padded/missing
positions with ``False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

import torch

StateTensor = Annotated[torch.Tensor, "shape: [batch, T, dim]"]
MaskTensor = Annotated[torch.Tensor, "shape: [batch, T]"]
HostLogitsTensor = Annotated[torch.Tensor, "host-defined shape, e.g. [batch, seq_len, vocab]"]
HostLabelsTensor = Annotated[torch.Tensor, "host-defined answer-label shape"]
StepIndexTensor = Annotated[torch.Tensor, "shape: [batch, T]"]


@dataclass
class HostModelOutput:
    """Output produced by a host adapter after a student forward pass.

    Shape contract:
        latent_states: ``torch.Tensor`` with shape ``[batch, T_student, dim]``.
        latent_mask: ``torch.Tensor`` with shape ``[batch, T_student]``. Valid
            student latent steps are ``True``; padding, missing, or sampled-out
            steps are ``False``.
        answer_logits: optional ``torch.Tensor`` with shape chosen by the host,
            typically ``[batch, seq_len, vocab]`` or ``[batch, vocab]``.
        answer_labels: optional ``torch.Tensor`` with shape matching the host
            answer loss contract.
        host_losses: mapping from host loss name to scalar ``torch.Tensor``.
        debug: host/debug metadata. LSP-JEPA core must not depend on required
            host-specific keys here.
    """

    latent_states: StateTensor
    latent_mask: MaskTensor
    answer_logits: HostLogitsTensor | None = None
    answer_labels: HostLabelsTensor | None = None
    host_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_states_and_mask(
            self.latent_states,
            self.latent_mask,
            states_name="latent_states",
            mask_name="latent_mask",
        )


@dataclass
class TeacherTargetBatch:
    """EMA teacher trajectory targets for LSP-State alignment.

    Shape contract:
        target_states: ``torch.Tensor`` with shape ``[batch, T_target, dim]``.
            These are contextual hidden states, gathered at reasoning step
            boundaries by the caller.
        target_mask: ``torch.Tensor`` with shape ``[batch, T_target]``. Valid
            teacher targets are ``True``; padding, missing boundaries,
            answer-only steps, and invalid targets are ``False``.
        step_indices: optional ``torch.Tensor`` with shape
            ``[batch, T_target]`` containing source token positions for valid
            targets. Invalid entries should be ignored according to
            ``target_mask``.
        debug: target-building metadata such as invalid counts or leakage
            checks. Core loss code must not require host-specific keys here.
    """

    target_states: StateTensor
    target_mask: MaskTensor
    step_indices: StepIndexTensor | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_states_and_mask(
            self.target_states,
            self.target_mask,
            states_name="target_states",
            mask_name="target_mask",
        )
        if self.step_indices is not None:
            _validate_2d_mask_like(
                self.step_indices,
                self.target_mask,
                tensor_name="step_indices",
                mask_name="target_mask",
            )


@dataclass
class LSPBatch:
    """Minimal prepared batch consumed by LSP-JEPA adapters.

    Shape contract:
        student_inputs: host-neutral or host-prepared model inputs for the
            student. In core mode these inputs must represent question-only
            context and must not contain ground-truth CoT tokens.
        teacher_inputs: optional host-neutral or host-prepared teacher inputs
            used to build ``TeacherTargetBatch``. These may include
            ``Question + CoT_<=i`` information, but default target construction
            must exclude answer tokens and answer prefixes.
        metadata: per-sample metadata. If tensorized per sample, tensors should
            use leading shape ``[batch, ...]``.
    """

    student_inputs: dict[str, Any]
    teacher_inputs: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_states_and_mask(
    states: torch.Tensor,
    mask: torch.Tensor,
    *,
    states_name: str,
    mask_name: str,
) -> None:
    if not isinstance(states, torch.Tensor):
        raise TypeError(f"{states_name} must be a torch.Tensor")
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{mask_name} must be a torch.Tensor")
    if states.ndim != 3:
        raise ValueError(f"{states_name} must have shape [batch, T, dim]")
    _validate_2d_mask_like(mask, states, tensor_name=mask_name, mask_name=states_name)


def _validate_2d_mask_like(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    *,
    tensor_name: str,
    mask_name: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{tensor_name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{tensor_name} must have shape [batch, T]")
    if tensor.shape[0] != reference.shape[0] or tensor.shape[1] != reference.shape[1]:
        raise ValueError(
            f"{tensor_name} shape {tuple(tensor.shape)} must match the first two "
            f"dimensions of {mask_name} shape {tuple(reference.shape)}"
        )
