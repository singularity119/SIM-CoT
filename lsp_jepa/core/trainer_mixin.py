"""Trainer-facing loss composition utilities for LSP-JEPA core.

This module is intentionally host-agnostic. It consumes the public
``HostModelOutput`` and ``TeacherTargetBatch`` contracts, maps student latent
states to teacher targets, computes one scalar LSP batch loss, and combines it
with optional answer/readout and adapter losses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from lsp_jepa.core.anti_collapse import compute_anti_collapse_loss
from lsp_jepa.core.latent_interface import HostModelOutput, TeacherTargetBatch
from lsp_jepa.core.losses import compute_lsp_state_loss
from lsp_jepa.core.mapping import map_trajectories
from lsp_jepa.core.metrics import effective_rank, pairwise_cosine, pairwise_l2, per_dim_variance


@dataclass(frozen=True)
class LSPLossOutput:
    """Scalar LSP losses plus the mask-aware mapped trajectory used to build them."""

    lsp_loss: torch.Tensor
    anti_collapse_loss: torch.Tensor
    per_step_lsp_loss: torch.Tensor
    mapped_student_states: torch.Tensor
    mapped_target_states: torch.Tensor
    mapped_mask: torch.Tensor
    student_indices: torch.Tensor
    target_indices: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class CombinedLoss:
    """One scalar total loss and auditable weighted terms."""

    total_loss: torch.Tensor
    loss_terms: dict[str, torch.Tensor]
    raw_losses: dict[str, torch.Tensor]
    weights: dict[str, float]


def compute_lsp_losses(
    host_output: HostModelOutput,
    teacher_targets: TeacherTargetBatch,
    config: Any,
) -> LSPLossOutput:
    """Compute mask-aware LSP-State and anti-collapse losses for one batch.

    The teacher target states are detached here to enforce the EMA stop-gradient
    contract. Student latent states are used directly so gradients flow through
    the full latent rollout.
    """

    _validate_trainer_config(config)
    objective_type = _config_get(config, ("lsp_objective", "type"), "state")
    if objective_type != "state":
        raise NotImplementedError("trainer_mixin currently implements LSP-State only")

    mapping_strategy = _config_get(config, ("mapping", "strategy"), "sequence")
    mapped = map_trajectories(
        host_output.latent_states,
        teacher_targets.target_states,
        host_output.latent_mask,
        teacher_targets.target_mask,
        strategy=mapping_strategy,
    )
    teacher_target_states = mapped.target_states.detach()
    alignment = _config_get(config, ("loss", "alignment"), "normalized_mse")
    smooth_l1_beta = _as_float(
        _config_get(config, ("loss", "smooth_l1_beta"), 1.0),
        "loss.smooth_l1_beta",
    )
    eps = _as_float(_config_get(config, ("loss", "eps"), 1e-8), "loss.eps")

    lsp_loss = compute_lsp_state_loss(
        mapped.student_states,
        teacher_target_states,
        mapped.mask,
        alignment=alignment,
        align_weight=1.0,
        smooth_l1_beta=smooth_l1_beta,
        eps=eps,
    )
    per_step_lsp_loss = _per_step_alignment_losses(
        mapped.student_states,
        teacher_target_states,
        mapped.mask,
        alignment=alignment,
        smooth_l1_beta=smooth_l1_beta,
        eps=eps,
    )

    anti_method = _config_get(config, ("loss", "anti_collapse"), "none")
    anti_weight = _as_float(
        _config_get(config, ("loss", "anti_collapse_weight"), 0.0),
        "loss.anti_collapse_weight",
    )
    if anti_method == "none" or anti_weight == 0.0:
        anti_collapse_loss = _zero_like(host_output.latent_states)
    else:
        anti_collapse_loss = compute_anti_collapse_loss(
            host_output.latent_states,
            host_output.latent_mask,
            method=anti_method,
            weight=1.0,
            variance_weight=_as_float(
                _config_get(config, ("loss", "variance_weight"), 1.0),
                "loss.variance_weight",
            ),
            covariance_weight=_as_float(
                _config_get(config, ("loss", "covariance_weight"), 1.0),
                "loss.covariance_weight",
            ),
            target_std=_as_float(
                _config_get(config, ("loss", "target_std"), 1.0),
                "loss.target_std",
            ),
            eps=_as_float(_config_get(config, ("loss", "variance_eps"), 1e-4), "loss.variance_eps"),
        )

    diagnostics = {
        "valid_alignment_count": mapped.mask.to(dtype=host_output.latent_states.dtype).sum(),
        "valid_alignment_fraction": _mask_fraction(mapped.mask, host_output.latent_states),
        "valid_student_fraction": _mask_fraction(host_output.latent_mask, host_output.latent_states),
        "valid_target_fraction": _mask_fraction(teacher_targets.target_mask, host_output.latent_states),
    }
    return LSPLossOutput(
        lsp_loss=lsp_loss,
        anti_collapse_loss=anti_collapse_loss,
        per_step_lsp_loss=per_step_lsp_loss,
        mapped_student_states=mapped.student_states,
        mapped_target_states=teacher_target_states,
        mapped_mask=mapped.mask,
        student_indices=mapped.student_indices,
        target_indices=mapped.target_indices,
        diagnostics=diagnostics,
    )


def combine_losses(
    answer_loss: torch.Tensor | None,
    lsp_loss: torch.Tensor,
    anti_collapse_loss: torch.Tensor,
    host_losses: Mapping[str, torch.Tensor] | None,
    config: Any,
) -> CombinedLoss:
    """Combine all configured losses into exactly one scalar batch loss."""

    _validate_trainer_config(config)
    host_losses = {} if host_losses is None else dict(host_losses)
    mode = _config_get(config, ("experiment", "mode"), "core")
    if mode != "adapter" and _has_nonzero_host_loss_weight(config):
        raise ValueError("host_losses weights may be nonzero only in adapter mode")

    total = _zero_from_available(answer_loss, lsp_loss, anti_collapse_loss, *host_losses.values())
    loss_terms: dict[str, torch.Tensor] = {}
    raw_losses: dict[str, torch.Tensor] = {}
    weights: dict[str, float] = {}

    def add_term(name: str, raw_loss: torch.Tensor | None, weight: float) -> None:
        nonlocal total
        if raw_loss is None:
            if weight != 0.0:
                raise ValueError(f"{name} weight is nonzero but no loss tensor was provided")
            return
        _ensure_scalar_tensor(raw_loss, name)
        raw_losses[name] = raw_loss
        weights[name] = weight
        if weight == 0.0:
            return
        weighted = raw_loss * weight
        loss_terms[name] = weighted
        total = total + weighted

    add_term(
        "lsp",
        lsp_loss,
        _as_float(_config_get(config, ("loss", "align_weight"), 1.0), "loss.align_weight"),
    )
    add_term(
        "anti_collapse",
        anti_collapse_loss,
        _as_float(
            _config_get(config, ("loss", "anti_collapse_weight"), 0.0),
            "loss.anti_collapse_weight",
        ),
    )
    add_term(
        "answer_readout",
        answer_loss,
        _as_float(
            _config_get(config, ("loss", "answer_readout_weight"), 0.0),
            "loss.answer_readout_weight",
        ),
    )

    host_terms = _resolve_host_loss_terms(host_losses, config)
    if host_terms and mode != "adapter":
        raise ValueError("host_losses can only be added to total loss in adapter mode")
    for name, raw_loss, weight in host_terms:
        add_term(f"host/{name}", raw_loss, weight)

    return CombinedLoss(
        total_loss=total,
        loss_terms=loss_terms,
        raw_losses=raw_losses,
        weights=weights,
    )


def log_lsp_metrics(
    host_output: HostModelOutput,
    teacher_targets: TeacherTargetBatch,
    lsp_losses: LSPLossOutput | None = None,
    *,
    config: Any | None = None,
    logger: Any | None = None,
    prefix: str = "lsp",
    step: int | None = None,
) -> dict[str, float]:
    """Build and optionally emit detached LSP diagnostics."""

    if lsp_losses is None:
        if config is None:
            raise ValueError("config is required when lsp_losses is not provided")
        lsp_losses = compute_lsp_losses(host_output, teacher_targets, config)

    variance = per_dim_variance(host_output.latent_states, host_output.latent_mask)
    metrics = {
        f"{prefix}/loss": lsp_losses.lsp_loss,
        f"{prefix}/anti_collapse_loss": lsp_losses.anti_collapse_loss,
        f"{prefix}/valid_alignment_fraction": lsp_losses.diagnostics[
            "valid_alignment_fraction"
        ],
        f"{prefix}/valid_student_fraction": lsp_losses.diagnostics["valid_student_fraction"],
        f"{prefix}/valid_target_fraction": lsp_losses.diagnostics["valid_target_fraction"],
        f"{prefix}/valid_alignment_count": lsp_losses.diagnostics["valid_alignment_count"],
        f"{prefix}/student_pairwise_cosine": pairwise_cosine(
            host_output.latent_states,
            host_output.latent_mask,
        ),
        f"{prefix}/student_pairwise_l2": pairwise_l2(
            host_output.latent_states,
            host_output.latent_mask,
        ),
        f"{prefix}/student_effective_rank": effective_rank(
            host_output.latent_states,
            host_output.latent_mask,
        ),
        f"{prefix}/student_variance_mean": variance["mean"],
        f"{prefix}/student_variance_min": variance["min"],
        f"{prefix}/student_teacher_cosine": _student_teacher_cosine(
            lsp_losses.mapped_student_states,
            lsp_losses.mapped_target_states,
            lsp_losses.mapped_mask,
        ),
    }
    loggable = {name: _to_float(value) for name, value in metrics.items()}
    _emit_metrics(logger, loggable, step=step)
    return loggable


def _per_step_alignment_losses(
    student_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor,
    *,
    alignment: str,
    smooth_l1_beta: float,
    eps: float,
) -> torch.Tensor:
    mask = mask.to(device=student_states.device, dtype=torch.bool)
    if alignment in {"mse", "masked_mse"}:
        values = (student_states - target_states).square()
        return _masked_step_mean(values, mask.unsqueeze(-1).expand_as(values))
    if alignment in {"normalized_mse", "masked_normalized_mse"}:
        element_mask = mask.unsqueeze(-1).expand_as(student_states)
        safe_student = torch.where(element_mask, student_states, torch.zeros_like(student_states))
        safe_target = torch.where(element_mask, target_states, torch.zeros_like(target_states))
        values = (
            F.normalize(safe_student, p=2, dim=-1, eps=eps)
            - F.normalize(safe_target, p=2, dim=-1, eps=eps)
        ).square()
        return _masked_step_mean(values, element_mask)
    if alignment in {"cosine", "cosine_loss", "masked_cosine_loss"}:
        element_mask = mask.unsqueeze(-1).expand_as(student_states)
        safe_student = torch.where(element_mask, student_states, torch.zeros_like(student_states))
        safe_target = torch.where(element_mask, target_states, torch.zeros_like(target_states))
        values = 1.0 - F.cosine_similarity(safe_student, safe_target, dim=-1, eps=eps)
        return _masked_step_mean(values, mask)
    if alignment in {"smooth_l1", "masked_smooth_l1"}:
        values = F.smooth_l1_loss(
            student_states,
            target_states,
            reduction="none",
            beta=smooth_l1_beta,
        )
        return _masked_step_mean(values, mask.unsqueeze(-1).expand_as(values))
    raise ValueError(f"Unsupported LSP alignment loss: {alignment}")


def _masked_step_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    if values.ndim == 2:
        denom = mask.to(dtype=values.dtype).sum(dim=0).clamp_min(1.0)
        return safe_values.sum(dim=0) / denom
    if values.ndim == 3:
        denom = mask.to(dtype=values.dtype).sum(dim=(0, 2)).clamp_min(1.0)
        return safe_values.sum(dim=(0, 2)) / denom
    raise ValueError("per-step loss values must have shape [B, T] or [B, T, D]")


def _student_teacher_cosine(
    student_states: torch.Tensor,
    target_states: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    cosine = F.cosine_similarity(student_states, target_states, dim=-1, eps=eps)
    mask = mask.to(device=cosine.device, dtype=torch.bool)
    safe_values = torch.where(mask, cosine, torch.zeros_like(cosine))
    denom = mask.to(dtype=cosine.dtype).sum().clamp_min(1.0)
    return safe_values.sum() / denom


def _validate_trainer_config(config: Any) -> None:
    mode = _config_get(config, ("experiment", "mode"), "core")
    if mode == "core":
        for key in (
            "codi_distill_weight",
            "simcot_decoder_weight",
            "intermediate_cot_ce_weight",
        ):
            value = _config_get(config, ("host_losses", key), 0.0)
            if _is_nonzero(value):
                raise ValueError(f"host_losses.{key} must be zero in core mode")

    if bool(_config_get(config, ("student", "detach_between_steps"), False)):
        raise ValueError("student.detach_between_steps must be false for LSP trainer mixin")

    state_offset = _config_get(config, ("lsp_objective", "state_offset"), 0)
    if int(state_offset) != 0:
        raise NotImplementedError(
            "Non-zero lsp_objective.state_offset is reserved for explicit ablations"
        )


_HOST_WEIGHT_ALIASES = {
    "host_answer_ce_weight": {
        "answer_ce",
        "answer_loss",
        "host_answer_ce",
        "host_answer_loss",
    },
    "codi_distill_weight": {
        "codi_distill",
        "codi_distill_loss",
        "distill",
        "distill_loss",
    },
    "simcot_decoder_weight": {
        "simcot_decoder",
        "simcot_decoder_ce",
        "simcot_decoder_loss",
        "decoder_ce",
        "decoder_loss",
    },
    "intermediate_cot_ce_weight": {
        "intermediate_cot_ce",
        "intermediate_cot_ce_loss",
        "cot_ce",
        "cot_ce_loss",
        "intermediate_ce",
    },
}


def _resolve_host_loss_terms(
    host_losses: Mapping[str, torch.Tensor],
    config: Any,
) -> list[tuple[str, torch.Tensor, float]]:
    host_config = _config_get(config, ("host_losses",), {})
    terms: list[tuple[str, torch.Tensor, float]] = []
    matched_known_weights: set[str] = set()
    for name, loss in host_losses.items():
        weight_key = _host_weight_key_for_loss(name, host_config)
        if weight_key is None:
            continue
        weight = _as_float(_config_get(host_config, (weight_key,), 0.0), f"host_losses.{weight_key}")
        if weight == 0.0:
            continue
        matched_known_weights.add(weight_key)
        terms.append((name, loss, weight))

    for weight_key in _HOST_WEIGHT_ALIASES:
        configured = _as_float(_config_get(host_config, (weight_key,), 0.0), f"host_losses.{weight_key}")
        if configured != 0.0 and weight_key not in matched_known_weights:
            known_aliases = ", ".join(sorted(_HOST_WEIGHT_ALIASES[weight_key]))
            raise ValueError(
                f"host_losses.{weight_key} is nonzero but no matching host loss "
                f"was provided. Expected one of: {known_aliases}"
            )
    return terms


def _host_weight_key_for_loss(name: str, host_config: Any) -> str | None:
    normalized_name = _normalize_key(name)
    explicit_keys = (
        f"{normalized_name}_weight",
        normalized_name,
        f"{name}_weight",
        name,
    )
    for key in explicit_keys:
        if _config_get(host_config, (key,), None) is not None:
            return key

    for weight_key, aliases in _HOST_WEIGHT_ALIASES.items():
        if normalized_name in aliases and _config_get(host_config, (weight_key,), None) is not None:
            return weight_key
    return None


def _has_nonzero_host_loss_weight(config: Any) -> bool:
    host_config = _config_get(config, ("host_losses",), {})
    if not isinstance(host_config, Mapping):
        try:
            keys = list(host_config.keys())
        except AttributeError:
            keys = []
    else:
        keys = list(host_config.keys())
    return any(str(key).endswith("_weight") and _is_nonzero(_config_get(host_config, (key,), 0.0)) for key in keys)


def _config_get(config: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = config
    for key in path:
        if current is None:
            return default
        if isinstance(current, Mapping):
            if key not in current:
                return default
            current = current[key]
            continue
        try:
            current = current[key]
            continue
        except (KeyError, TypeError, AttributeError):
            pass
        if hasattr(current, key):
            current = getattr(current, key)
            continue
        return default
    return current


def _as_float(value: Any, name: str) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc


def _is_nonzero(value: Any) -> bool:
    return _as_float(value, "loss weight") != 0.0


def _normalize_key(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def _ensure_scalar_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _zero_from_available(*values: torch.Tensor | None) -> torch.Tensor:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.sum() * 0.0
    return torch.tensor(0.0)


def _mask_fraction(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return _zero_like(reference)
    return mask.to(device=reference.device, dtype=reference.dtype).mean()


def _to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            value = value.mean()
        return float(value.detach().cpu().item())
    return float(value)


def _emit_metrics(
    logger: Any | None,
    metrics: Mapping[str, float],
    *,
    step: int | None,
) -> None:
    if logger is None:
        return
    if isinstance(logger, dict):
        logger.update(metrics)
        return
    if hasattr(logger, "add_scalar"):
        for name, value in metrics.items():
            logger.add_scalar(name, value, step)
        return
    if hasattr(logger, "log"):
        try:
            logger.log(dict(metrics), step=step)
        except TypeError:
            logger.log(dict(metrics))
        return
    if isinstance(logger, Callable):
        logger(dict(metrics))
        return
    raise TypeError("logger must be None, dict, callable, or expose log/add_scalar")


__all__ = [
    "CombinedLoss",
    "LSPLossOutput",
    "compute_lsp_losses",
    "combine_losses",
    "log_lsp_metrics",
]
