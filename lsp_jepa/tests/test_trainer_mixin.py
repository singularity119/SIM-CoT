from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
import yaml

import lsp_jepa.core.trainer_mixin as trainer_mixin
from lsp_jepa.core.latent_interface import HostModelOutput, TeacherTargetBatch
from lsp_jepa.core.trainer_mixin import combine_losses, compute_lsp_losses


def _config(**overrides):
    config = {
        "experiment": {"mode": "core"},
        "student": {"detach_between_steps": False},
        "lsp_objective": {"type": "state", "state_offset": 0},
        "mapping": {"strategy": "one_to_one"},
        "loss": {
            "alignment": "mse",
            "align_weight": 1.0,
            "anti_collapse": "none",
            "anti_collapse_weight": 0.0,
            "answer_readout_weight": 0.0,
        },
        "host_losses": {
            "host_answer_ce_weight": 0.0,
            "codi_distill_weight": 0.0,
            "simcot_decoder_weight": 0.0,
            "intermediate_cot_ce_weight": 0.0,
        },
    }
    for section, values in overrides.items():
        config.setdefault(section, {}).update(values)
    return config


def test_single_backward_contract_uses_one_total_loss_and_no_optimizer_step_calls():
    student = torch.nn.Linear(3, 3, bias=False)
    inputs = torch.randn(2, 3, 3)
    latent_states = student(inputs)
    teacher_states = torch.randn(2, 3, 3, requires_grad=True)
    host_output = HostModelOutput(
        latent_states=latent_states,
        latent_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    teacher_targets = TeacherTargetBatch(
        target_states=teacher_states,
        target_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    losses = compute_lsp_losses(host_output, teacher_targets, _config())
    combined = combine_losses(
        None,
        losses.lsp_loss,
        losses.anti_collapse_loss,
        host_output.host_losses,
        _config(),
    )

    assert losses.per_step_lsp_loss.shape == (3,)
    assert combined.total_loss.shape == ()
    combined.total_loss.backward()

    assert student.weight.grad is not None
    assert student.weight.grad.abs().sum().item() > 0.0
    assert teacher_states.grad is None

    tree = ast.parse(inspect.getsource(trainer_mixin))
    forbidden_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in {"backward", "step", "zero_grad"}:
            forbidden_calls.append(node.func.attr)
    assert forbidden_calls == []


def test_host_losses_only_contribute_in_adapter_mode():
    lsp_loss = torch.tensor(2.0)
    anti_loss = torch.tensor(0.0)
    host_losses = {"answer_ce": torch.tensor(4.0)}

    with pytest.raises(ValueError, match="adapter mode"):
        combine_losses(
            None,
            lsp_loss,
            anti_loss,
            host_losses,
            _config(host_losses={"host_answer_ce_weight": 0.5}),
        )

    adapter_config = _config(
        experiment={"mode": "adapter"},
        host_losses={"host_answer_ce_weight": 0.5},
    )
    combined = combine_losses(None, lsp_loss, anti_loss, host_losses, adapter_config)

    assert torch.allclose(combined.total_loss, torch.tensor(4.0))
    assert torch.allclose(combined.loss_terms["host/answer_ce"], torch.tensor(2.0))


@pytest.mark.parametrize("weight_key", ["simcot_decoder_weight", "codi_distill_weight"])
def test_core_mode_forbids_simcot_decoder_and_codi_distill_weights(weight_key):
    with pytest.raises(ValueError, match=f"host_losses.{weight_key} must be zero"):
        combine_losses(
            None,
            torch.tensor(1.0),
            torch.tensor(0.0),
            {},
            _config(host_losses={weight_key: 0.1}),
        )


def test_core_sequence_config_keeps_forbidden_host_weights_zero():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "core"
        / "gsm8k_lsp_state_seq.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["experiment"]["mode"] == "core"
    assert config["mapping"]["strategy"] == "sequence"
    assert config["student"]["detach_between_steps"] is False
    assert config["host_losses"]["simcot_decoder_weight"] == 0.0
    assert config["host_losses"]["codi_distill_weight"] == 0.0
    assert config["host_losses"]["intermediate_cot_ce_weight"] == 0.0


def test_step_trajectory_full_train_config_uses_raw_mse_alignment():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "core"
        / "lsp_step_trajectory_full_train.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["lsp_objective"]["type"] == "step_trajectory"
    assert config["loss"]["alignment"] == "mse"


def test_loss_composition_preserves_alignment_mask_and_invalid_gradients_are_zero():
    student_states = torch.tensor(
        [
            [[1.0, 2.0], [1000.0, 1000.0], [3.0, 4.0]],
            [[-999.0, -999.0], [5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )
    target_states = torch.tensor(
        [
            [[2.0, 0.0], [-1000.0, -1000.0], [1.0, 1.0]],
            [[999.0, 999.0], [5.0, 7.0], [70.0, 80.0]],
        ]
    )
    host_output = HostModelOutput(
        latent_states=student_states,
        latent_mask=torch.tensor(
            [
                [True, False, True],
                [False, True, True],
            ]
        ),
    )
    teacher_targets = TeacherTargetBatch(
        target_states=target_states,
        target_mask=torch.tensor(
            [
                [True, True, False],
                [True, True, False],
            ]
        ),
    )

    losses = compute_lsp_losses(host_output, teacher_targets, _config())
    combined = combine_losses(
        None,
        losses.lsp_loss,
        losses.anti_collapse_loss,
        {},
        _config(),
    )

    expected_mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
        ]
    )
    expected = torch.stack(
        [
            (student_states[0, 0] - target_states[0, 0]).square(),
            (student_states[1, 1] - target_states[1, 1]).square(),
        ]
    ).mean()

    assert losses.mapped_mask.tolist() == expected_mask.tolist()
    assert torch.allclose(combined.total_loss, expected)

    combined.total_loss.backward()
    assert student_states.grad is not None
    assert student_states.grad[0, 1].abs().sum().item() == 0.0
    assert student_states.grad[1, 0].abs().sum().item() == 0.0
    assert student_states.grad[:, 2].abs().sum().item() == 0.0
