from __future__ import annotations

import pytest
import torch

from lsp_jepa.adapters.base import HostAdapter
from lsp_jepa.core.latent_interface import HostModelOutput, LSPBatch, TeacherTargetBatch


class FakeAdapter:
    def prepare_batch(self, raw_batch):
        return {"student": raw_batch["question"]}

    def forward_student(self, model, batch, *, output_latent_states: bool):
        del model, batch, output_latent_states
        return HostModelOutput(
            latent_states=torch.zeros(2, 3, 4),
            latent_mask=torch.tensor(
                [
                    [True, True, False],
                    [True, False, False],
                ]
            ),
            host_losses={},
            debug={"adapter": "fake"},
        )

    def build_teacher_input(self, batch):
        return {"teacher": batch["student"]}

    def extract_answer_loss(self, host_output, batch):
        del host_output, batch
        return None


def test_host_model_output_shape_contract_accepts_valid_tensors():
    output = HostModelOutput(
        latent_states=torch.randn(2, 3, 4),
        latent_mask=torch.ones(2, 3, dtype=torch.bool),
        answer_logits=torch.randn(2, 5, 7),
        answer_labels=torch.ones(2, 5, dtype=torch.long),
        host_losses={"answer_ce": torch.tensor(0.0)},
    )

    assert output.latent_states.shape == (2, 3, 4)
    assert output.latent_mask.shape == (2, 3)
    assert output.latent_mask.dtype is torch.bool
    assert output.host_losses["answer_ce"].shape == ()


def test_host_model_output_rejects_bad_mask_shape():
    with pytest.raises(ValueError, match="latent_mask"):
        HostModelOutput(
            latent_states=torch.randn(2, 3, 4),
            latent_mask=torch.ones(2, 4, dtype=torch.bool),
        )


def test_teacher_target_batch_shape_contract_accepts_step_indices():
    targets = TeacherTargetBatch(
        target_states=torch.randn(2, 4, 8),
        target_mask=torch.tensor(
            [
                [True, True, True, False],
                [True, False, False, False],
            ]
        ),
        step_indices=torch.tensor(
            [
                [3, 7, 11, 0],
                [4, 0, 0, 0],
            ]
        ),
        debug={"invalid_count": 0},
    )

    assert targets.target_states.shape == (2, 4, 8)
    assert targets.target_mask.shape == (2, 4)
    assert targets.step_indices.shape == (2, 4)


def test_teacher_target_batch_rejects_bad_step_indices_shape():
    with pytest.raises(ValueError, match="step_indices"):
        TeacherTargetBatch(
            target_states=torch.randn(2, 4, 8),
            target_mask=torch.ones(2, 4, dtype=torch.bool),
            step_indices=torch.ones(2, 5, dtype=torch.long),
        )


def test_lsp_batch_allows_host_neutral_payloads():
    batch = LSPBatch(
        student_inputs={"input_ids": torch.ones(2, 5, dtype=torch.long)},
        teacher_inputs={"input_ids": torch.ones(2, 9, dtype=torch.long)},
        metadata={"sample_ids": ["a", "b"]},
    )

    assert batch.student_inputs["input_ids"].shape == (2, 5)
    assert batch.teacher_inputs is not None
    assert batch.teacher_inputs["input_ids"].shape == (2, 9)
    assert batch.metadata["sample_ids"] == ["a", "b"]


def test_fake_adapter_satisfies_protocol_and_returns_host_output():
    adapter = FakeAdapter()

    assert isinstance(adapter, HostAdapter)
    batch = adapter.prepare_batch({"question": "q"})
    output = adapter.forward_student(None, batch, output_latent_states=True)
    teacher_input = adapter.build_teacher_input(batch)

    assert output.latent_states.shape == (2, 3, 4)
    assert output.latent_mask.tolist() == [[True, True, False], [True, False, False]]
    assert teacher_input == {"teacher": "q"}
    assert adapter.extract_answer_loss(output, batch) is None
