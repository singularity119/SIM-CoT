from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lsp_jepa.adapters.coconut_lsp_adapter import (
    CoconutLSPAdapter,
    forward_coconut_with_optional_lsp,
)
from lsp_jepa.core.trainer_mixin import combine_losses


class BaselineHost(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return {"baseline": True, "kwargs": kwargs}


def _adapter_factory_that_must_not_run():
    raise AssertionError("adapter factory should not run when use_lsp_jepa is false")


def test_flag_off_calls_baseline_forward_without_constructing_adapter():
    model = BaselineHost()
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}

    output = forward_coconut_with_optional_lsp(
        model,
        batch,
        use_lsp_jepa=False,
        adapter_factory=_adapter_factory_that_must_not_run,
    )

    assert output["baseline"] is True
    assert model.calls == 1
    assert torch.equal(output["kwargs"]["input_ids"], batch["input_ids"])


def test_adapter_prefers_host_output_latent_states_and_preserves_autograd():
    latent_states = torch.randn(2, 3, 4, requires_grad=True)
    loss = torch.tensor(1.25)

    class Host(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(
                latent_states=latent_states,
                logits=torch.ones(2, 5, 7),
                loss=loss,
            )

    adapter = CoconutLSPAdapter(latent_token_id=99)
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "labels": torch.tensor([[1, -100], [2, -100]]),
    }
    output = adapter.forward_student(Host(), batch, output_latent_states=True)

    assert output.latent_states is latent_states
    assert output.latent_states.requires_grad is True
    assert output.latent_mask.tolist() == [
        [True, True, True],
        [True, True, True],
    ]
    assert output.answer_logits.shape == (2, 5, 7)
    assert output.answer_labels is batch["labels"]
    assert output.host_losses == {"host_answer_ce": loss}
    assert output.debug["latent_source"] == "host_output.latent_states"


def test_adapter_falls_back_to_inputs_embeds_and_pads_variable_latent_counts():
    input_ids = torch.tensor(
        [
            [7, 1, 7, 0],
            [7, 2, 3, 0],
        ]
    )
    inputs_embeds = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    inputs_embeds.requires_grad_()

    class Host(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(inputs_embeds=inputs_embeds)

    output = CoconutLSPAdapter(latent_token_id=7).forward_student(
        Host(),
        {"input_ids": input_ids},
        output_latent_states=True,
    )

    expected = torch.stack(
        [
            torch.stack([inputs_embeds[0, 0], inputs_embeds[0, 2]]),
            torch.stack([inputs_embeds[1, 0], torch.zeros_like(inputs_embeds[1, 0])]),
        ]
    )
    assert output.latent_states.shape == (2, 2, 3)
    assert torch.equal(output.latent_states, expected)
    assert output.latent_mask.tolist() == [[True, True], [True, False]]
    assert output.latent_states.requires_grad is True
    assert output.debug["latent_source"] == "host_output.inputs_embeds"


def test_adapter_returns_empty_latent_sequence_when_no_latent_tokens_are_present():
    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    inputs_embeds = torch.randn(2, 3, 5)

    class Host(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(inputs_embeds=inputs_embeds)

    output = CoconutLSPAdapter(latent_token_id=7).forward_student(
        Host(),
        {"input_ids": input_ids},
        output_latent_states=True,
    )

    assert output.latent_states.shape == (2, 0, 5)
    assert output.latent_mask.shape == (2, 0)
    assert output.latent_mask.dtype is torch.bool


def test_adapter_raises_clear_error_when_host_exposes_no_latent_payload():
    class Host(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(logits=torch.ones(1, 2, 3), loss=torch.tensor(0.0))

    with pytest.raises(NotImplementedError, match="latent_states.*inputs_embeds.*use_lsp_jepa"):
        CoconutLSPAdapter(latent_token_id=7).forward_student(
            Host(),
            {"input_ids": torch.tensor([[1, 2]])},
            output_latent_states=True,
        )


def test_adapter_raises_value_error_when_latent_token_id_is_missing_for_fallback():
    inputs_embeds = torch.randn(1, 3, 2)

    class Host(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(inputs_embeds=inputs_embeds)

    with pytest.raises(ValueError, match="latent_token_id"):
        CoconutLSPAdapter().forward_student(
            Host(),
            {"input_ids": torch.tensor([[1, 2, 3]])},
            output_latent_states=True,
        )


def test_host_answer_ce_is_named_once_and_combined_once_by_convention():
    host_answer_ce = torch.tensor(3.0)

    class Host(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(
                latent_states=torch.zeros(1, 1, 2),
                loss=host_answer_ce,
            )

    adapter = CoconutLSPAdapter()
    output = adapter.forward_student(
        Host(),
        {"input_ids": torch.tensor([[1]])},
        output_latent_states=True,
    )

    assert set(output.host_losses) == {"host_answer_ce"}
    assert adapter.extract_answer_loss(output, {}) is host_answer_ce

    combined = combine_losses(
        None,
        torch.tensor(2.0),
        torch.tensor(0.0),
        output.host_losses,
        _adapter_config(),
    )

    assert torch.allclose(combined.total_loss, host_answer_ce)
    assert set(combined.loss_terms) == {"host/host_answer_ce"}


def _adapter_config():
    return {
        "experiment": {"mode": "adapter"},
        "student": {"detach_between_steps": False},
        "lsp_objective": {"type": "state", "state_offset": 0},
        "loss": {
            "align_weight": 0.0,
            "anti_collapse_weight": 0.0,
            "answer_readout_weight": 0.0,
        },
        "host_losses": {
            "host_answer_ce_weight": 1.0,
            "codi_distill_weight": 0.0,
            "simcot_decoder_weight": 0.0,
            "intermediate_cot_ce_weight": 0.0,
        },
    }
