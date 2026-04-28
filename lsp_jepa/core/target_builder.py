"""Teacher target construction utilities for LSP-JEPA core.

The functions in this module are host-agnostic. They build teacher inputs from
question plus ground-truth reasoning steps, keep answer tokens out by default,
and gather contextual hidden states at reasoning step boundaries. Default
boundary gathering assumes a decoder-only causal teacher: causal masking makes
the hidden state at step ``i`` equivalent to encoding ``Question + CoT_<=i``.
Non-causal or bidirectional teachers must use explicit per-step prefix inputs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from lsp_jepa.core.latent_interface import TeacherTargetBatch

TargetPooling = Literal["step_last_token", "prefix_last_token", "step_mean"]


@dataclass(frozen=True)
class StepFilterResult:
    """Answer-leakage filtering result for one sample."""

    steps: list[str]
    keep_mask: list[bool]
    kept_indices: list[int]
    dropped_indices: list[int]
    dropped_reasons: list[str | None]


@dataclass(frozen=True)
class StepBoundaryExtraction:
    """Tokenized teacher sequence and step-boundary metadata for one sample."""

    text: str
    input_ids: list[int]
    attention_mask: list[int]
    step_boundaries: list[int]
    step_starts: list[int]
    step_mask: list[bool]
    filtered_steps: list[str] = field(default_factory=list)
    kept_step_indices: list[int] = field(default_factory=list)


def build_teacher_inputs(
    tokenizer: Any,
    questions: str | Sequence[str],
    cot_steps: Sequence[str] | Sequence[Sequence[str]],
    *,
    answers: str | Sequence[str | None] | None = None,
    max_length: int | None = None,
    padding: bool | Literal["longest", "max_length"] = True,
    truncation: bool = False,
    return_tensors: Literal["pt"] | None = "pt",
    add_special_tokens: bool = True,
    question_step_separator: str = "\n",
    step_separator: str = "\n",
    answer_prefix: str = "\n#### ",
    exclude_answer_tokens: bool = True,
    exclude_answer_prefix: bool = True,
    filter_answer_steps: bool = True,
    target_step_offset: int = 0,
) -> dict[str, Any]:
    """Build padded teacher model inputs and step-boundary tensors.

    Defaults follow the LSP-JEPA-Core target semantics for decoder-only causal
    teachers: teacher inputs contain ``Question + CoT`` prefixes, answer tokens
    are excluded, answer-formatting steps are filtered, and target step ``i`` is
    aligned to latent state ``h_i`` through ``target_step_offset=0``. Non-zero
    offsets are reserved for explicit transition ablations and are not applied
    here.
    """

    if target_step_offset != 0:
        raise NotImplementedError(
            "Non-zero target_step_offset is reserved for transition ablations"
        )

    question_list = _as_question_batch(questions)
    step_batch = _as_step_batch(cot_steps, batch_size=len(question_list))
    answer_list = _as_optional_batch(answers, batch_size=len(question_list))

    extractions: list[StepBoundaryExtraction] = []
    filter_results: list[StepFilterResult] = []
    for question, steps, answer in zip(question_list, step_batch, answer_list, strict=True):
        if filter_answer_steps:
            filter_result = filter_answer_only_steps(
                steps,
                answer=answer,
                exclude_answer_prefix=exclude_answer_prefix,
            )
            target_steps = filter_result.steps
        else:
            filter_result = StepFilterResult(
                steps=[str(step) for step in steps],
                keep_mask=[True for _ in steps],
                kept_indices=list(range(len(steps))),
                dropped_indices=[],
                dropped_reasons=[None for _ in steps],
            )
            target_steps = filter_result.steps

        trailing_text = None
        if not exclude_answer_tokens and answer is not None and str(answer).strip():
            trailing_text = f"{answer_prefix}{answer}"

        extraction = extract_step_boundaries(
            tokenizer,
            question,
            target_steps,
            trailing_text=trailing_text,
            add_special_tokens=add_special_tokens,
            question_step_separator=question_step_separator,
            step_separator=step_separator,
        )
        extraction = StepBoundaryExtraction(
            text=extraction.text,
            input_ids=extraction.input_ids,
            attention_mask=extraction.attention_mask,
            step_boundaries=extraction.step_boundaries,
            step_starts=extraction.step_starts,
            step_mask=extraction.step_mask,
            filtered_steps=filter_result.steps,
            kept_step_indices=filter_result.kept_indices,
        )
        extractions.append(extraction)
        filter_results.append(filter_result)

    token_lengths = [len(extraction.input_ids) for extraction in extractions]
    padded_length = _resolve_padded_length(
        token_lengths,
        padding=padding,
        max_length=max_length,
    )
    max_steps = max((len(extraction.step_boundaries) for extraction in extractions), default=0)
    pad_token_id = _pad_token_id(tokenizer)

    input_ids_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    boundary_rows: list[list[int]] = []
    start_rows: list[list[int]] = []
    step_mask_rows: list[list[bool]] = []

    for extraction in extractions:
        input_ids = list(extraction.input_ids)
        attention_mask = list(extraction.attention_mask)
        if max_length is not None and len(input_ids) > max_length:
            if not truncation:
                raise ValueError(
                    "teacher input exceeds max_length; pass truncation=True "
                    "to mark truncated step targets invalid"
                )
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        if len(input_ids) < padded_length:
            pad_count = padded_length - len(input_ids)
            input_ids = input_ids + [pad_token_id] * pad_count
            attention_mask = attention_mask + [0] * pad_count
        elif len(input_ids) > padded_length:
            input_ids = input_ids[:padded_length]
            attention_mask = attention_mask[:padded_length]

        boundaries = list(extraction.step_boundaries)
        starts = list(extraction.step_starts)
        valid_steps = list(extraction.step_mask)
        for idx, boundary in enumerate(boundaries):
            is_valid = (
                valid_steps[idx]
                and 0 <= boundary < len(attention_mask)
                and bool(attention_mask[boundary])
            )
            valid_steps[idx] = bool(is_valid)
            if idx < len(starts):
                starts[idx] = starts[idx] if is_valid else 0
            boundaries[idx] = boundary if is_valid else 0

        if len(boundaries) < max_steps:
            missing = max_steps - len(boundaries)
            boundaries.extend([0] * missing)
            starts.extend([0] * missing)
            valid_steps.extend([False] * missing)

        input_ids_rows.append(input_ids)
        attention_rows.append(attention_mask)
        boundary_rows.append(boundaries)
        start_rows.append(starts)
        step_mask_rows.append(valid_steps)

    output: dict[str, Any]
    if return_tensors == "pt":
        output = {
            "input_ids": torch.tensor(input_ids_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "step_boundaries": torch.tensor(boundary_rows, dtype=torch.long),
            "step_starts": torch.tensor(start_rows, dtype=torch.long),
            "step_mask": torch.tensor(step_mask_rows, dtype=torch.bool),
        }
    elif return_tensors is None:
        output = {
            "input_ids": input_ids_rows,
            "attention_mask": attention_rows,
            "step_boundaries": boundary_rows,
            "step_starts": start_rows,
            "step_mask": step_mask_rows,
        }
    else:
        raise ValueError("target_builder only supports return_tensors='pt' or None")

    output.update(
        {
            "texts": [extraction.text for extraction in extractions],
            "filtered_steps": [extraction.filtered_steps for extraction in extractions],
            "kept_step_indices": [
                extraction.kept_step_indices for extraction in extractions
            ],
            "debug": {
                "exclude_answer_tokens": exclude_answer_tokens,
                "exclude_answer_prefix": exclude_answer_prefix,
                "filter_answer_steps": filter_answer_steps,
                "target_step_offset": target_step_offset,
                "dropped_step_indices": [
                    filter_result.dropped_indices for filter_result in filter_results
                ],
                "dropped_step_reasons": [
                    filter_result.dropped_reasons for filter_result in filter_results
                ],
            },
        }
    )
    return output


def extract_step_boundaries(
    tokenizer: Any,
    question: str,
    steps: Sequence[str],
    *,
    trailing_text: str | None = None,
    add_special_tokens: bool = True,
    question_step_separator: str = "\n",
    step_separator: str = "\n",
) -> StepBoundaryExtraction:
    """Tokenize one ``Question + CoT`` sequence and locate step boundaries.

    Boundaries are the token positions of the last token in each retained
    reasoning step after tokenizer special tokens have been applied. The
    optional ``trailing_text`` can be used for explicit answer-leakage
    ablations without moving CoT step boundaries.
    """

    clean_question = str(question)
    clean_steps = [str(step).strip() for step in steps if str(step).strip()]
    text_without_trailing = _join_question_and_steps(
        clean_question,
        clean_steps,
        question_step_separator=question_step_separator,
        step_separator=step_separator,
    )
    full_text = text_without_trailing
    if trailing_text:
        full_text = f"{full_text}{trailing_text}"

    bare_full_ids = _encode(tokenizer, full_text, add_special_tokens=False)
    input_ids = _encode_with_optional_special_tokens(
        tokenizer,
        full_text,
        bare_ids=bare_full_ids,
        add_special_tokens=add_special_tokens,
    )
    content_start = _find_contiguous_subsequence(input_ids, bare_full_ids)
    if content_start is None:
        content_start = 0

    step_boundaries: list[int] = []
    step_starts: list[int] = []
    previous_prefix = clean_question
    for step_index in range(len(clean_steps)):
        prefix_text = _join_question_and_steps(
            clean_question,
            clean_steps[: step_index + 1],
            question_step_separator=question_step_separator,
            step_separator=step_separator,
        )
        prefix_ids = _encode(tokenizer, prefix_text, add_special_tokens=False)
        previous_prefix_ids = _encode(
            tokenizer,
            previous_prefix,
            add_special_tokens=False,
        )
        if len(prefix_ids) <= len(previous_prefix_ids):
            step_boundaries.append(0)
            step_starts.append(0)
        else:
            step_starts.append(content_start + len(previous_prefix_ids))
            step_boundaries.append(content_start + len(prefix_ids) - 1)
        previous_prefix = prefix_text

    return StepBoundaryExtraction(
        text=full_text,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        step_boundaries=step_boundaries,
        step_starts=step_starts,
        step_mask=[True] * len(step_boundaries),
        filtered_steps=clean_steps,
        kept_step_indices=list(range(len(clean_steps))),
    )


def filter_answer_only_steps(
    steps: Sequence[str],
    *,
    answer: str | None = None,
    exclude_answer_prefix: bool = True,
) -> StepFilterResult:
    """Drop answer-only or answer-formatting reasoning steps.

    Steps such as ``"#### 42"``, ``"The answer is 42"``, or a bare answer
    string are not valid LSP-Core teacher targets. If a reasoning step contains
    useful reasoning followed by an answer marker, the reasoning prefix is kept
    when ``exclude_answer_prefix=True``.
    """

    kept_steps: list[str] = []
    keep_mask: list[bool] = []
    kept_indices: list[int] = []
    dropped_indices: list[int] = []
    dropped_reasons: list[str | None] = []
    normalized_answer = _normalize_answer(answer)

    for index, raw_step in enumerate(steps):
        step = str(raw_step).strip()
        reason: str | None = None
        if not step:
            reason = "empty"
        elif _is_answer_only_step(step, normalized_answer=normalized_answer):
            reason = "answer_only"
        elif exclude_answer_prefix:
            stripped_step, stripped_reason = _strip_answer_suffix(step)
            if stripped_step != step:
                step = stripped_step
                reason = stripped_reason if not step else None
            if not step:
                reason = reason or "answer_prefix"
            elif _is_answer_only_step(step, normalized_answer=normalized_answer):
                reason = "answer_only"

        if reason is None:
            kept_steps.append(step)
            keep_mask.append(True)
            kept_indices.append(index)
            dropped_reasons.append(None)
        else:
            keep_mask.append(False)
            dropped_indices.append(index)
            dropped_reasons.append(reason)

    return StepFilterResult(
        steps=kept_steps,
        keep_mask=keep_mask,
        kept_indices=kept_indices,
        dropped_indices=dropped_indices,
        dropped_reasons=dropped_reasons,
    )


def gather_step_boundary_hidden_states(
    teacher_output: Any,
    step_boundaries: torch.Tensor,
    step_mask: torch.Tensor | None = None,
    *,
    attention_mask: torch.Tensor | None = None,
    step_starts: torch.Tensor | None = None,
    target_layer: int | str = -1,
    target_pooling: TargetPooling = "step_last_token",
    detach: bool = True,
    allow_embedding_targets: bool = False,
) -> TeacherTargetBatch:
    """Gather teacher targets from contextual hidden states at step boundaries.

    The default path rejects explicit embedding-layer targets because raw token
    embeddings are only valid for negative-control ablations.
    """

    hidden = _select_contextual_hidden_states(
        teacher_output,
        target_layer=target_layer,
        allow_embedding_targets=allow_embedding_targets,
    )
    if hidden.ndim != 3:
        raise ValueError("teacher hidden states must have shape [batch, seq_len, dim]")
    if step_boundaries.ndim != 2:
        raise ValueError("step_boundaries must have shape [batch, T]")
    if hidden.shape[0] != step_boundaries.shape[0]:
        raise ValueError("hidden states and step_boundaries batch size must match")
    if step_mask is None:
        step_mask = torch.ones_like(step_boundaries, dtype=torch.bool)
    else:
        if step_mask.shape != step_boundaries.shape:
            raise ValueError("step_mask must match step_boundaries shape")
        step_mask = step_mask.to(device=step_boundaries.device, dtype=torch.bool)

    step_boundaries = step_boundaries.to(device=hidden.device, dtype=torch.long)
    step_mask = step_mask.to(device=hidden.device, dtype=torch.bool)
    if step_starts is not None:
        if step_starts.shape != step_boundaries.shape:
            raise ValueError("step_starts must match step_boundaries shape")
        step_starts = step_starts.to(device=hidden.device, dtype=torch.long)
    valid_mask = step_mask.clone()
    seq_len = hidden.shape[1]
    valid_mask &= step_boundaries >= 0
    valid_mask &= step_boundaries < seq_len

    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, seq_len]")
        if attention_mask.shape[0] != hidden.shape[0] or attention_mask.shape[1] != seq_len:
            raise ValueError("attention_mask must match hidden state batch and seq_len")
        safe_boundaries = step_boundaries.clamp(min=0, max=max(seq_len - 1, 0))
        boundary_attention = attention_mask.to(
            device=hidden.device,
            dtype=torch.bool,
        ).gather(1, safe_boundaries)
        valid_mask &= boundary_attention

    if target_pooling in {"step_last_token", "prefix_last_token"}:
        target_states = _gather_positions(hidden, step_boundaries)
    elif target_pooling == "step_mean":
        if step_starts is None:
            step_starts = _infer_step_starts(step_boundaries)
        target_states = _mean_pool_spans(hidden, step_starts, step_boundaries, valid_mask)
    else:
        raise ValueError(f"Unsupported target_pooling: {target_pooling}")

    target_states = torch.where(
        valid_mask.unsqueeze(-1),
        target_states,
        torch.zeros_like(target_states),
    )
    if detach:
        target_states = target_states.detach()

    step_indices = torch.where(
        valid_mask,
        step_boundaries.to(device=hidden.device, dtype=torch.long),
        torch.zeros_like(step_boundaries, dtype=torch.long),
    )
    return TeacherTargetBatch(
        target_states=target_states,
        target_mask=valid_mask,
        step_indices=step_indices,
        debug={
            "target_layer": target_layer,
            "target_pooling": target_pooling,
            "source": "contextual_hidden_states",
        },
    )


def _as_question_batch(questions: str | Sequence[str]) -> list[str]:
    if isinstance(questions, str):
        return [questions]
    return [str(question) for question in questions]


def _as_step_batch(
    cot_steps: Sequence[str] | Sequence[Sequence[str]],
    *,
    batch_size: int,
) -> list[list[str]]:
    if batch_size == 1 and (
        not cot_steps or isinstance(next(iter(cot_steps)), str)  # type: ignore[arg-type]
    ):
        return [[str(step) for step in cot_steps]]  # type: ignore[arg-type]

    step_batch: list[list[str]] = []
    for sample_steps in cot_steps:  # type: ignore[assignment]
        if isinstance(sample_steps, str):
            step_batch.append([sample_steps])
        else:
            step_batch.append([str(step) for step in sample_steps])
    if len(step_batch) != batch_size:
        raise ValueError("cot_steps batch size must match questions batch size")
    return step_batch


def _as_optional_batch(
    values: str | Sequence[str | None] | None,
    *,
    batch_size: int,
) -> list[str | None]:
    if values is None:
        return [None] * batch_size
    if isinstance(values, str):
        if batch_size != 1:
            raise ValueError("a scalar answer can only be used for a single question")
        return [values]
    output = [None if value is None else str(value) for value in values]
    if len(output) != batch_size:
        raise ValueError("answers batch size must match questions batch size")
    return output


def _resolve_padded_length(
    lengths: Sequence[int],
    *,
    padding: bool | Literal["longest", "max_length"],
    max_length: int | None,
) -> int:
    if not lengths:
        return max_length or 0
    if padding == "max_length":
        if max_length is None:
            raise ValueError("padding='max_length' requires max_length")
        return max_length
    if padding is True or padding == "longest":
        longest = max(lengths)
        if max_length is None:
            return longest
        return min(longest, max_length)
    if padding is False:
        if len(set(lengths)) != 1:
            raise ValueError("padding=False requires equal-length teacher inputs")
        return lengths[0]
    raise ValueError("padding must be True, False, 'longest', or 'max_length'")


def _join_question_and_steps(
    question: str,
    steps: Sequence[str],
    *,
    question_step_separator: str,
    step_separator: str,
) -> str:
    if not steps:
        return question
    return f"{question.rstrip()}{question_step_separator}{step_separator.join(steps)}"


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    tokenized = tokenizer(text, add_special_tokens=add_special_tokens)
    input_ids = tokenized["input_ids"] if isinstance(tokenized, Mapping) else tokenized.input_ids
    if input_ids and isinstance(input_ids[0], Sequence):
        input_ids = input_ids[0]
    return list(input_ids)


def _encode_with_optional_special_tokens(
    tokenizer: Any,
    text: str,
    *,
    bare_ids: list[int],
    add_special_tokens: bool,
) -> list[int]:
    if not add_special_tokens:
        return list(bare_ids)
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        try:
            return list(tokenizer.build_inputs_with_special_tokens(list(bare_ids)))
        except TypeError:
            pass
    return _encode(tokenizer, text, add_special_tokens=True)


def _pad_token_id(tokenizer: Any) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return 0
    return int(pad_token_id)


def _find_contiguous_subsequence(values: Sequence[int], pattern: Sequence[int]) -> int | None:
    if not pattern:
        return 0
    if len(pattern) > len(values):
        return None
    pattern_list = list(pattern)
    for start in range(0, len(values) - len(pattern) + 1):
        if list(values[start : start + len(pattern)]) == pattern_list:
            return start
    return None


_ANSWER_PREFIX_PATTERNS = [
    re.compile(r"(?i)#{4,}\s*"),
    re.compile(r"(?i)(?:the\s+)?answer\s+is\s*:?\s*"),
    re.compile(r"(?i)final\s+answer\s*:?\s*"),
    re.compile(r"(?i)therefore,?\s+the\s+answer\s+is\s*:?\s*"),
    re.compile(r"(?i)so,?\s+the\s+answer\s+is\s*:?\s*"),
]


def _strip_answer_suffix(step: str) -> tuple[str, str | None]:
    for pattern in _ANSWER_PREFIX_PATTERNS:
        match = pattern.search(step)
        if match is None:
            continue
        prefix = step[: match.start()].strip()
        if prefix:
            return prefix, "answer_suffix"
        return "", "answer_prefix"
    return step, None


def _is_answer_only_step(step: str, *, normalized_answer: str | None) -> bool:
    stripped = step.strip()
    lowered = stripped.lower()
    if not stripped:
        return True
    if any(pattern.match(stripped) is not None for pattern in _ANSWER_PREFIX_PATTERNS):
        return True
    normalized_step = _normalize_answer(stripped)
    if normalized_answer is not None and normalized_step == normalized_answer:
        return True
    answer_like = {
        "answer",
        "finalanswer",
        "theansweris",
        "thereforetheansweris",
        "sotheansweris",
    }
    return lowered.replace(" ", "").replace(":", "") in answer_like


def _normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    normalized = re.sub(r"\W+", "", str(answer).lower())
    return normalized or None


def _select_contextual_hidden_states(
    teacher_output: Any,
    *,
    target_layer: int | str,
    allow_embedding_targets: bool,
) -> torch.Tensor:
    hidden_states = _output_get(teacher_output, "hidden_states")
    if hidden_states is not None:
        if isinstance(target_layer, str) and target_layer.startswith("last_"):
            count = int(target_layer.split("_", maxsplit=1)[1])
            if count <= 0:
                raise ValueError("last_N target_layer must use a positive N")
            if len(hidden_states) < count:
                raise ValueError("not enough hidden state layers for target_layer")
            if len(hidden_states) == count and not allow_embedding_targets:
                raise ValueError(
                    "target_layer would include raw embeddings; set "
                    "allow_embedding_targets=True only for explicit ablations"
                )
            selected = torch.stack(list(hidden_states[-count:]), dim=0).mean(dim=0)
            return selected

        index = _resolve_layer_index(target_layer, len(hidden_states))
        if index == 0 and len(hidden_states) > 1 and not allow_embedding_targets:
            raise ValueError(
                "target_layer=0 selects raw embeddings; set allow_embedding_targets=True "
                "only for explicit ablations"
            )
        return hidden_states[index]

    last_hidden_state = _output_get(teacher_output, "last_hidden_state")
    if last_hidden_state is not None:
        return last_hidden_state

    if isinstance(teacher_output, torch.Tensor):
        return teacher_output

    for embedding_key in ("inputs_embeds", "input_embeddings", "embeddings"):
        if _output_get(teacher_output, embedding_key) is not None:
            raise ValueError(
                "teacher targets require contextual hidden states, not raw embeddings"
            )
    raise ValueError("teacher_output must provide hidden_states or last_hidden_state")


def _output_get(output: Any, key: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(key)
    return getattr(output, key, None)


def _resolve_layer_index(target_layer: int | str, num_layers: int) -> int:
    if isinstance(target_layer, str):
        if target_layer in {"last", "final"}:
            return num_layers - 1
        try:
            target_layer = int(target_layer)
        except ValueError as exc:
            raise ValueError(f"Unsupported target_layer: {target_layer}") from exc

    index = int(target_layer)
    if index < 0:
        index = num_layers + index
    if index < 0 or index >= num_layers:
        raise ValueError("target_layer index out of range")
    return index


def _gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    safe_positions = positions.clamp(min=0, max=max(hidden.shape[1] - 1, 0))
    gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return hidden.gather(dim=1, index=gather_index.to(device=hidden.device))


def _infer_step_starts(step_boundaries: torch.Tensor) -> torch.Tensor:
    starts = torch.zeros_like(step_boundaries)
    if step_boundaries.shape[1] > 1:
        starts[:, 1:] = step_boundaries[:, :-1] + 1
    return starts


def _mean_pool_spans(
    hidden: torch.Tensor,
    step_starts: torch.Tensor,
    step_boundaries: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    batch, steps = step_boundaries.shape
    output = torch.zeros(batch, steps, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
    for batch_idx in range(batch):
        for step_idx in range(steps):
            if not bool(valid_mask[batch_idx, step_idx]):
                continue
            start = int(step_starts[batch_idx, step_idx].item())
            end = int(step_boundaries[batch_idx, step_idx].item())
            start = max(0, min(start, hidden.shape[1] - 1))
            end = max(start, min(end, hidden.shape[1] - 1))
            output[batch_idx, step_idx] = hidden[batch_idx, start : end + 1].mean(dim=0)
    return output


__all__ = [
    "StepBoundaryExtraction",
    "StepFilterResult",
    "TargetPooling",
    "build_teacher_inputs",
    "extract_step_boundaries",
    "filter_answer_only_steps",
    "gather_step_boundary_hidden_states",
]
