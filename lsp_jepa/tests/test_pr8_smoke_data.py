from __future__ import annotations

import pytest
import torch

from lsp_jepa.train_lsp_coconut_minimal import (
    Sample,
    final_valid_step_position,
    sample_from_record,
    validate_no_answer_leakage,
)


def test_gsm8k_answer_field_parses_question_cot_and_answer() -> None:
    sample = sample_from_record(
        {
            "question": "How many clips?",
            "answer": "Natalia sold 48/2 = 24 clips.\nShe sold 48+24 = 72 clips.\n#### 72",
        },
        source="unit",
    )

    assert sample is not None
    assert sample.question == "How many clips?"
    assert sample.cot_steps == [
        "Natalia sold 48/2 = 24 clips.",
        "She sold 48+24 = 72 clips.",
    ]
    assert sample.answer == "72"


def test_gsm8k_aug_explicit_cot_parses_without_answer_marker() -> None:
    sample = sample_from_record(
        {
            "question": "Q",
            "cot": "First compute the total. Then subtract the known amount.",
            "answer": "#### 5",
        },
        source="unit",
    )

    assert sample is not None
    assert sample.cot_steps == [
        "First compute the total.",
        "Then subtract the known amount.",
    ]
    assert sample.answer == "5"


def test_final_valid_step_position_uses_last_true_boundary() -> None:
    positions = final_valid_step_position(
        torch.tensor([[3, 7, 0], [2, 0, 0]]),
        torch.tensor([[True, True, False], [False, False, False]]),
    )

    assert positions.tolist() == [7, 0]


def test_answer_leakage_check_rejects_answer_markers() -> None:
    with pytest.raises(RuntimeError, match="answer leakage"):
        validate_no_answer_leakage(
            [Sample("Q", ["#### 4"], "4", "unit")],
            {"texts": ["Q\n#### 4"], "filtered_steps": [["#### 4"]]},
            include_answer_tokens=False,
            include_answer_prefix=False,
        )
