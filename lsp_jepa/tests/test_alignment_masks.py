from __future__ import annotations

import pytest
import torch

from lsp_jepa.core.mapping import map_one_to_one, map_sequence, map_trajectories


def test_sequence_mapping_uses_final_valid_steps_and_joint_mask():
    student_states = torch.tensor(
        [
            [[1.0], [2.0], [99.0]],
            [[10.0], [20.0], [30.0]],
            [[100.0], [200.0], [300.0]],
        ]
    )
    target_states = torch.tensor(
        [
            [[3.0], [4.0], [5.0], [999.0]],
            [[40.0], [50.0], [60.0], [70.0]],
            [[400.0], [500.0], [600.0], [700.0]],
        ]
    )
    student_mask = torch.tensor(
        [
            [True, True, False],
            [False, False, False],
            [True, False, False],
        ]
    )
    target_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [False, False, False, False],
        ]
    )

    result = map_sequence(student_states, target_states, student_mask, target_mask)

    assert result.student_states.shape == (3, 1, 1)
    assert result.target_states.shape == (3, 1, 1)
    assert result.mask.tolist() == [[True], [False], [False]]
    assert result.student_indices.tolist() == [[1], [0], [0]]
    assert result.target_indices.tolist() == [[2], [1], [0]]
    assert result.student_states[:, 0, 0].tolist() == [2.0, 10.0, 100.0]
    assert result.target_states[:, 0, 0].tolist() == [5.0, 50.0, 400.0]


def test_one_to_one_mapping_intersects_masks_and_truncates_to_shared_length():
    student_states = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    target_states = (torch.arange(2 * 3 * 1, dtype=torch.float32) + 100).reshape(2, 3, 1)
    student_mask = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, True],
        ]
    )
    target_mask = torch.tensor(
        [
            [True, True, False],
            [False, True, True],
        ]
    )

    result = map_one_to_one(student_states, target_states, student_mask, target_mask)

    assert result.student_states.shape == (2, 3, 1)
    assert result.target_states.shape == (2, 3, 1)
    assert result.mask.tolist() == [
        [True, False, False],
        [False, True, False],
    ]
    assert result.student_indices.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert result.target_indices.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
    ]


def test_mapping_accepts_numeric_masks_but_returns_bool_joint_mask():
    student_states = torch.randn(1, 2, 3)
    target_states = torch.randn(1, 2, 3)
    student_mask = torch.tensor([[1, 0]])
    target_mask = torch.tensor([[1.0, 1.0]])

    result = map_one_to_one(student_states, target_states, student_mask, target_mask)

    assert result.mask.dtype is torch.bool
    assert result.mask.tolist() == [[True, False]]


def test_map_trajectories_dispatches_supported_strategies():
    student_states = torch.randn(1, 2, 3)
    target_states = torch.randn(1, 3, 3)
    student_mask = torch.ones(1, 2, dtype=torch.bool)
    target_mask = torch.ones(1, 3, dtype=torch.bool)

    assert map_trajectories(
        student_states,
        target_states,
        student_mask,
        target_mask,
        strategy="sequence",
    ).mask.shape == (1, 1)
    assert map_trajectories(
        student_states,
        target_states,
        student_mask,
        target_mask,
        strategy="one_to_one",
    ).mask.shape == (1, 2)


def test_map_trajectories_rejects_unimplemented_strategy():
    with pytest.raises(ValueError, match="Unsupported mapping strategy"):
        map_trajectories(
            torch.randn(1, 2, 3),
            torch.randn(1, 2, 3),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
            strategy="soft_dtw",  # type: ignore[arg-type]
        )


def test_mapping_rejects_shape_mismatches_before_loss_code_exists():
    with pytest.raises(ValueError, match="hidden dim"):
        map_one_to_one(
            torch.randn(1, 2, 3),
            torch.randn(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="student_mask"):
        map_sequence(
            torch.randn(1, 2, 3),
            torch.randn(1, 2, 3),
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )
