"""Minimal Coconut host adapter for LSP-JEPA.

This adapter is intentionally conservative: it does not modify Coconut and it
only consumes latent states that the host output already exposes directly, or
that can be recovered from returned ``inputs_embeds`` at Coconut latent-token
positions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from lsp_jepa.core.latent_interface import HostModelOutput, LSPBatch


class CoconutLSPAdapter:
    """Expose Coconut student outputs through the host-agnostic LSP contract."""

    def __init__(self, latent_token_id: int | None = None) -> None:
        self.latent_token_id = latent_token_id

    def prepare_batch(self, raw_batch: Any) -> dict[str, Any]:
        if isinstance(raw_batch, LSPBatch):
            return {
                "student_inputs": raw_batch.student_inputs,
                "teacher_inputs": raw_batch.teacher_inputs,
                "metadata": raw_batch.metadata,
            }
        if isinstance(raw_batch, Mapping):
            return dict(raw_batch)
        raise TypeError("CoconutLSPAdapter.prepare_batch expects a mapping or LSPBatch")

    def forward_student(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
        *,
        output_latent_states: bool,
    ) -> HostModelOutput:
        del output_latent_states
        model_inputs = _student_model_inputs(batch)
        host_output = model(**model_inputs)
        latent_states, latent_mask, source = self._extract_latents(
            model,
            model_inputs,
            host_output,
        )
        host_losses = _host_losses(host_output)
        answer_labels = _output_value(host_output, "answer_labels")
        if answer_labels is None:
            answer_labels = model_inputs.get("labels")

        return HostModelOutput(
            latent_states=latent_states,
            latent_mask=latent_mask,
            answer_logits=_output_value(host_output, "logits"),
            answer_labels=answer_labels,
            host_losses=host_losses,
            debug={
                "adapter": "coconut",
                "latent_source": source,
            },
        )

    def build_teacher_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        teacher_inputs = batch.get("teacher_inputs")
        if teacher_inputs is None:
            raise NotImplementedError(
                "CoconutLSPAdapter requires caller-provided teacher_inputs for "
                "teacher target construction; PR6 only implements student "
                "host-output adaptation."
            )
        if not isinstance(teacher_inputs, Mapping):
            raise TypeError("teacher_inputs must be a mapping")
        return dict(teacher_inputs)

    def extract_answer_loss(
        self,
        host_output: HostModelOutput,
        batch: dict[str, Any],
    ) -> torch.Tensor | None:
        del batch
        return host_output.host_losses.get("host_answer_ce")

    def _extract_latents(
        self,
        model: torch.nn.Module,
        model_inputs: Mapping[str, Any],
        host_output: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        latent_states = _output_value(host_output, "latent_states")
        if latent_states is not None:
            latent_mask = _output_value(host_output, "latent_mask")
            if latent_mask is None:
                latent_mask = torch.ones(
                    latent_states.shape[:2],
                    dtype=torch.bool,
                    device=latent_states.device,
                )
            else:
                latent_mask = latent_mask.to(device=latent_states.device, dtype=torch.bool)
            return latent_states, latent_mask, "host_output.latent_states"

        inputs_embeds = _output_value(host_output, "inputs_embeds")
        if inputs_embeds is not None:
            latent_token_id = self._resolve_latent_token_id(model)
            input_ids = model_inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "input_ids is required to extract Coconut latent states "
                    "from inputs_embeds."
                )
            return (
                _extract_latents_from_inputs_embeds(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    latent_token_id=latent_token_id,
                )
                + ("host_output.inputs_embeds",)
            )

        raise NotImplementedError(
            "Coconut host output does not expose latent_states or inputs_embeds. "
            "A later integration must add a use_lsp_jepa-protected hook that "
            "explicitly exposes latent_states without changing baseline behavior."
        )

    def _resolve_latent_token_id(self, model: torch.nn.Module) -> int:
        if self.latent_token_id is not None:
            return int(self.latent_token_id)
        model_latent_token_id = getattr(model, "latent_token_id", None)
        if model_latent_token_id is None:
            raise ValueError(
                "latent_token_id is required when extracting Coconut latent "
                "states from inputs_embeds."
            )
        return int(model_latent_token_id)


def forward_coconut_with_optional_lsp(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    use_lsp_jepa: bool,
    adapter_factory: Callable[[], CoconutLSPAdapter] | None = None,
    output_latent_states: bool = True,
) -> Any:
    """Small guarded forward helper for tests and future hook wiring.

    The flag-off path calls the host model directly and never constructs an
    adapter.
    """

    model_inputs = _student_model_inputs(batch)
    if not use_lsp_jepa:
        return model(**model_inputs)

    adapter = adapter_factory() if adapter_factory is not None else CoconutLSPAdapter()
    return adapter.forward_student(
        model,
        batch,
        output_latent_states=output_latent_states,
    )


def _student_model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    student_inputs = batch.get("student_inputs")
    if student_inputs is not None:
        if not isinstance(student_inputs, Mapping):
            raise TypeError("student_inputs must be a mapping")
        return dict(student_inputs)
    return {
        key: value
        for key, value in batch.items()
        if key not in {"teacher_inputs", "metadata"}
    }


def _output_value(output: Any, name: str) -> Any:
    if isinstance(output, Mapping) and name in output:
        return output[name]
    return getattr(output, name, None)


def _host_losses(host_output: Any) -> dict[str, torch.Tensor]:
    host_losses = _output_value(host_output, "host_losses")
    losses = dict(host_losses) if isinstance(host_losses, Mapping) else {}
    loss = _output_value(host_output, "loss")
    if loss is not None:
        losses["host_answer_ce"] = loss
    return losses


def _extract_latents_from_inputs_embeds(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    latent_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if not isinstance(inputs_embeds, torch.Tensor):
        raise TypeError("inputs_embeds must be a torch.Tensor")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq_len]")
    if inputs_embeds.ndim != 3:
        raise ValueError("inputs_embeds must have shape [batch, seq_len, dim]")
    if input_ids.shape != inputs_embeds.shape[:2]:
        raise ValueError(
            "input_ids shape must match the first two dimensions of inputs_embeds"
        )

    latent_positions = input_ids.to(device=inputs_embeds.device).eq(latent_token_id)
    batch_size, _, hidden_dim = inputs_embeds.shape
    counts = latent_positions.sum(dim=1)
    max_count = int(counts.max().item()) if counts.numel() else 0
    latent_mask = torch.zeros(
        batch_size,
        max_count,
        dtype=torch.bool,
        device=inputs_embeds.device,
    )

    if max_count == 0:
        return inputs_embeds.new_empty(batch_size, 0, hidden_dim), latent_mask

    rows: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        selected = inputs_embeds[batch_idx, latent_positions[batch_idx], :]
        count = selected.shape[0]
        latent_mask[batch_idx, :count] = True
        if count < max_count:
            padding = selected.new_zeros(max_count - count, hidden_dim)
            selected = torch.cat([selected, padding], dim=0)
        rows.append(selected)

    return torch.stack(rows, dim=0), latent_mask
