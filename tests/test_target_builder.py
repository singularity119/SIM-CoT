from __future__ import annotations

import inspect
import re

import pytest
import torch

from lsp_jepa.core.target_builder import (
    build_teacher_inputs,
    filter_answer_only_steps,
    gather_step_boundary_hidden_states,
)


class ToyTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        tokens = re.findall(r"\S+", text)
        ids = [self._id_for(token) for token in tokens]
        if add_special_tokens:
            return self.build_inputs_with_special_tokens(ids)
        return ids

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [101, *token_ids, 102]

    def _id_for(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab) + 1
        return self._vocab[token]


def test_build_teacher_inputs_excludes_answer_tokens_by_default() -> None:
    signature = inspect.signature(build_teacher_inputs)
    assert signature.parameters["exclude_answer_tokens"].default is True

    tokenizer = ToyTokenizer()
    batch = build_teacher_inputs(
        tokenizer,
        ["Q"],
        [["think one", "think two", "#### 4"]],
        answers=["4"],
    )

    assert batch["debug"]["exclude_answer_tokens"] is True
    assert batch["filtered_steps"] == [["think one", "think two"]]
    assert "4" not in batch["texts"][0]
    assert batch["step_mask"].tolist() == [[True, True]]


def test_answer_only_steps_are_filtered_without_dropping_reasoning_steps() -> None:
    result = filter_answer_only_steps(
        ["combine facts", "The answer is 12", "derive value. #### 12"],
        answer="12",
    )

    assert result.steps == ["combine facts", "derive value."]
    assert result.keep_mask == [True, False, True]
    assert result.dropped_indices == [1]


def test_step_boundary_positions_are_stable_when_answer_tokens_are_added() -> None:
    tokenizer = ToyTokenizer()
    default_batch = build_teacher_inputs(
        tokenizer,
        ["Q"],
        [["think one", "think two"]],
        answers=["4"],
    )
    leakage_ablation_batch = build_teacher_inputs(
        tokenizer,
        ["Q"],
        [["think one", "think two"]],
        answers=["4"],
        exclude_answer_tokens=False,
    )

    assert default_batch["step_boundaries"].tolist() == [[3, 5]]
    assert leakage_ablation_batch["step_boundaries"].tolist() == [[3, 5]]
    assert default_batch["step_mask"].tolist() == [[True, True]]
    assert leakage_ablation_batch["step_mask"].tolist() == [[True, True]]


def test_padding_positions_have_zero_masks_for_steps_and_targets() -> None:
    tokenizer = ToyTokenizer()
    batch = build_teacher_inputs(
        tokenizer,
        ["Q1", "Q2"],
        [["first step", "second step"], ["only step"]],
    )

    assert batch["step_mask"].tolist() == [[True, True], [True, False]]
    assert batch["step_boundaries"].tolist()[1][1] == 0
    assert batch["attention_mask"][1, -1].item() == 0

    hidden = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
    targets = gather_step_boundary_hidden_states(
        {"last_hidden_state": hidden},
        torch.tensor([[2, 4]]),
        torch.tensor([[True, True]]),
        attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
    )

    assert targets.target_mask.tolist() == [[True, False]]
    assert torch.equal(targets.target_states[0, 1], torch.zeros(2))
    assert targets.step_indices.tolist() == [[2, 0]]


def test_targets_use_contextual_hidden_states_not_raw_token_embeddings() -> None:
    raw_embeddings = torch.full((1, 6, 3), -100.0)
    contextual = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    boundaries = torch.tensor([[2, 4]])
    step_mask = torch.tensor([[True, True]])

    targets = gather_step_boundary_hidden_states(
        {"hidden_states": (raw_embeddings, contextual)},
        boundaries,
        step_mask,
    )

    expected = contextual[:, [2, 4], :]
    assert torch.equal(targets.target_states, expected)
    assert not torch.equal(targets.target_states, raw_embeddings[:, [2, 4], :])
    assert targets.debug["source"] == "contextual_hidden_states"

    with pytest.raises(ValueError, match="raw embeddings"):
        gather_step_boundary_hidden_states(
            {"hidden_states": (raw_embeddings, contextual)},
            boundaries,
            step_mask,
            target_layer=0,
        )
