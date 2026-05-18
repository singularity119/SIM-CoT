from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_lsp_jepa_core.py"
SPEC = importlib.util.spec_from_file_location("train_lsp_jepa_core_for_tests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_lsp_jepa_core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_lsp_jepa_core
SPEC.loader.exec_module(train_lsp_jepa_core)

load_coconut_initial_checkpoint = train_lsp_jepa_core.load_coconut_initial_checkpoint
lsp_weight_for_step = train_lsp_jepa_core.lsp_weight_for_step
lsp_weight_warmup_progress = train_lsp_jepa_core.lsp_weight_warmup_progress
is_post_lsp_weight_warmup_epoch = train_lsp_jepa_core.is_post_lsp_weight_warmup_epoch
maybe_update_best_post_warmup_eval_checkpoint = (
    train_lsp_jepa_core.maybe_update_best_post_warmup_eval_checkpoint
)
prepare_coconut_initialization = train_lsp_jepa_core.prepare_coconut_initialization
prepare_lsp_weight_warmup = train_lsp_jepa_core.prepare_lsp_weight_warmup
safe_float_ratio = train_lsp_jepa_core.safe_float_ratio
load_best_checkpoint_metric = train_lsp_jepa_core.load_best_checkpoint_metric
maybe_save_checkpoint = train_lsp_jepa_core.maybe_save_checkpoint
maybe_run_epoch_eval = train_lsp_jepa_core.maybe_run_epoch_eval
checkpoint_epoch_metadata = train_lsp_jepa_core.checkpoint_epoch_metadata
write_checkpoint_alias = train_lsp_jepa_core.write_checkpoint_alias
align_student_teacher_states = train_lsp_jepa_core.align_student_teacher_states
build_final_answer_eval_input_ids = train_lsp_jepa_core.build_final_answer_eval_input_ids
latent_counts_for_objective = train_lsp_jepa_core.latent_counts_for_objective
select_teacher_step_indices = train_lsp_jepa_core.select_teacher_step_indices
TeacherTargetBatch = train_lsp_jepa_core.TeacherTargetBatch
Sample = train_lsp_jepa_core.Sample


def test_lsp_weight_warmup_uses_zero_first_step_and_target_after_warmup():
    args = SimpleNamespace(
        max_steps=100,
        lsp_weight=0.2,
        lsp_weight_warmup_ratio=0.10,
        lsp_weight_warmup_steps=0,
    )

    prepare_lsp_weight_warmup(args)

    assert args.resolved_lsp_weight_warmup_steps == 10
    assert lsp_weight_warmup_progress(args, 1) == 0.0
    assert lsp_weight_for_step(args, 1) == 0.0
    assert lsp_weight_for_step(args, 6) == 0.1
    assert lsp_weight_for_step(args, 11) == 0.2


def test_post_warmup_eval_starts_after_warmup_is_complete():
    args = SimpleNamespace(
        max_steps=100,
        lsp_weight=0.2,
        lsp_weight_warmup_ratio=0.10,
        lsp_weight_warmup_steps=0,
    )

    prepare_lsp_weight_warmup(args)

    assert not is_post_lsp_weight_warmup_epoch(args, 10)
    assert is_post_lsp_weight_warmup_epoch(args, 11)

    args.resolved_lsp_weight_warmup_steps = 0
    assert is_post_lsp_weight_warmup_epoch(args, 1)


def test_safe_float_ratio_reports_weighted_loss_balance():
    assert safe_float_ratio(0.3, 0.6) == 0.5
    assert safe_float_ratio(0.3, 0.0) is None


def test_checkpoint_alias_metadata_survives_source_pruning(tmp_path):
    source = tmp_path / "step_000010.pt"
    source.write_bytes(b"checkpoint")

    alias = write_checkpoint_alias(
        tmp_path,
        alias_name="latest.pt",
        checkpoint_path=source,
        metadata={"step": 10, "total_loss": 0.25},
    )

    assert alias.read_bytes() == b"checkpoint"
    metric_value, metric_path = load_best_checkpoint_metric(
        tmp_path / "latest.json",
        "total_loss",
    )
    assert metric_value == 0.25
    assert metric_path == alias
    source.unlink()
    assert alias.read_bytes() == b"checkpoint"


def test_best_post_warmup_eval_checkpoint_alias_updates_only_after_warmup(tmp_path):
    args = SimpleNamespace(
        epoch_steps=10,
        max_steps=100,
        lsp_weight=0.05,
        lsp_weight_warmup_ratio=0.10,
        lsp_weight_warmup_steps=0,
    )
    prepare_lsp_weight_warmup(args)
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"post-warmup")
    epoch_eval = {
        "accuracy": 0.35,
        "exact_match": 0.35,
        "invalid_answer_rate": 0.0,
        "num_eval_samples": 10,
        "metrics_path": str(tmp_path / "evals" / "metrics.jsonl"),
        "step_metrics_path": str(tmp_path / "evals" / "step_000011.json"),
    }

    best_acc, best_path, updated = maybe_update_best_post_warmup_eval_checkpoint(
        args,
        checkpoint_dir=tmp_path,
        checkpoint_path=checkpoint,
        step=10,
        epoch_eval=epoch_eval,
        best_accuracy=None,
        best_checkpoint_path=None,
    )

    assert best_acc is None
    assert best_path is None
    assert not updated
    assert not (tmp_path / "best_post_warmup_eval_accuracy.pt").exists()

    best_acc, best_path, updated = maybe_update_best_post_warmup_eval_checkpoint(
        args,
        checkpoint_dir=tmp_path,
        checkpoint_path=checkpoint,
        step=11,
        epoch_eval=epoch_eval,
        best_accuracy=None,
        best_checkpoint_path=None,
    )

    assert best_acc == 0.35
    assert best_path == tmp_path / "best_post_warmup_eval_accuracy.pt"
    assert updated
    metadata = json.loads((tmp_path / "best_post_warmup_eval_accuracy.json").read_text())
    assert metadata["accuracy"] == 0.35
    assert metadata["post_warmup_eval_eligible"] is True
    assert metadata["lsp_weight_warmup_steps"] == 10


def test_coconut_initialization_json_resolves_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"placeholder")
    metadata = tmp_path / "best.json"
    metadata.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint),
                "eval_accuracy": 0.3586,
                "completed_epoch": 25,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        init_coconut_checkpoint=metadata,
        resolved_init_coconut_checkpoint=None,
        init_coconut_checkpoint_metadata_path=None,
        init_coconut_checkpoint_metadata={},
    )

    prepare_coconut_initialization(args)

    assert args.resolved_init_coconut_checkpoint == str(checkpoint)
    assert args.init_coconut_checkpoint_metadata_path == str(metadata)
    assert args.init_coconut_checkpoint_metadata["eval_accuracy"] == 0.3586
    assert args.init_coconut_checkpoint_metadata["completed_epoch"] == 25


def test_checkpoint_epoch_metadata_labels_epoch_boundaries():
    metadata = checkpoint_epoch_metadata(step=6426, epoch_steps=3213)

    assert metadata["step"] == 6426
    assert metadata["global_step"] == 6426
    assert metadata["epoch_index"] == 2
    assert metadata["epoch_step"] == 3213
    assert metadata["epoch_label"] == "epoch_0002"
    assert metadata["checkpoint_label"] == "epoch_0002_step_006426"


def test_maybe_save_checkpoint_writes_latest_only(tmp_path):
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_causallm = nn.Linear(2, 2)

    student = TinyStudent()
    ema_teacher = SimpleNamespace(teacher_model=nn.Linear(2, 2))
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    args = SimpleNamespace(
        save_checkpoints=True,
        save_every=10,
        max_steps=100,
        objective="step_trajectory",
        ema_decay=0.995,
        lsp_weight=0.05,
        lsp_weight_warmup_ratio=0.3,
        resolved_lsp_weight_warmup_steps=30,
        host_answer_ce_weight=1.0,
        anti_collapse_type="variance",
        anti_collapse_weight=0.01,
        resolved_init_coconut_checkpoint="coconut.pt",
    )

    checkpoint = maybe_save_checkpoint(
        args,
        step=10,
        checkpoint_dir=tmp_path,
        student=student,
        ema_teacher=ema_teacher,
        optimizer=optimizer,
    )

    assert checkpoint == tmp_path / "latest.pt"
    assert checkpoint.exists()
    assert not list(tmp_path.glob("step_*.pt"))


def test_forced_step0_eval_writes_baseline_metrics(tmp_path):
    class TinyTokenizer:
        eos_token_id = 99

        def convert_tokens_to_ids(self, token):
            return {
                "<|start-latent|>": 10,
                "<|latent|>": 11,
                "<|end-latent|>": 12,
            }[token]

        def encode(self, text, add_special_tokens=True):
            return [2, 3]

        def decode(self, token_ids, skip_special_tokens=True):
            return "4"

    class TinyStudent:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def generate(self, input_ids, attention_mask, max_new_tokens, synced_gpus):
            del attention_mask, max_new_tokens, synced_gpus
            generated = torch.tensor([[42, 99]], dtype=input_ids.dtype, device=input_ids.device)
            return torch.cat([input_ids, generated], dim=1)

    args = SimpleNamespace(
        eval_every=0,
        max_steps=10,
        epoch_steps=5,
        num_latent_steps=4,
        resolved_eval_num_latent_steps=10,
        latent_tokens_per_teacher_step=2,
        max_teacher_step_groups=5,
        teacher_step_selection="uniform_first_last",
        mapping_strategy="grouped_step_end",
        latent_count_mode="fixed_full",
        eval_max_new_tokens=4,
        eval_limit_samples=0,
        eval_json=None,
        eval_split="test",
        eval_dataset_id="openai/gsm8k",
        eval_dataset_config="main",
        eval_save_examples=1,
    )
    eval_output_dir = tmp_path / "evals"
    eval_output_dir.mkdir()
    eval_metrics_path = eval_output_dir / "metrics.jsonl"

    metrics = maybe_run_epoch_eval(
        args,
        step=0,
        student=TinyStudent(),
        tokenizer=TinyTokenizer(),
        eval_samples=[
            Sample(question="q", cot_steps=["step"], answer="#### 4", source="unit"),
        ],
        checkpoint_path=tmp_path / "coconut.pt",
        eval_output_dir=eval_output_dir,
        eval_metrics_path=eval_metrics_path,
        device=torch.device("cpu"),
        force=True,
        extra_metrics={
            "is_step0_baseline": True,
            "baseline_checkpoint_path": str(tmp_path / "coconut.pt"),
        },
    )

    assert metrics["step"] == 0
    assert metrics["epoch_index"] == 0
    assert metrics["is_step0_baseline"] is True
    assert metrics["eval_num_latent_steps"] == 10
    assert metrics["latent_tokens_per_teacher_step"] == 2
    assert metrics["mapping_strategy"] == "grouped_step_end"
    assert metrics["accuracy"] == 1.0
    step0 = json.loads((eval_output_dir / "step_000000.json").read_text())
    rows = [json.loads(line) for line in eval_metrics_path.read_text().splitlines()]
    assert step0["baseline_checkpoint_path"].endswith("coconut.pt")
    assert rows == [step0]


def test_grouped_step_end_maps_two_latents_to_one_teacher_step():
    student_states = torch.arange(1 * 6 * 1, dtype=torch.float32).view(1, 6, 1)
    teacher_states = torch.tensor([[[10.0], [20.0], [30.0]]])
    mask = torch.ones((1, 6), dtype=torch.bool)
    teacher_mask = torch.ones((1, 3), dtype=torch.bool)
    args = SimpleNamespace(
        objective="step_trajectory",
        mapping_strategy="grouped_step_end",
        latent_tokens_per_teacher_step=2,
        max_teacher_step_groups=5,
        num_latent_steps=10,
        teacher_step_selection="uniform_first_last",
    )

    aligned_student, aligned_teacher, aligned_mask = align_student_teacher_states(
        student_states,
        mask,
        teacher_states,
        teacher_mask,
        args=args,
        alignment="normalized_mse",
    )

    assert aligned_student.squeeze(-1).tolist() == [[1.0, 3.0, 5.0]]
    assert aligned_teacher.squeeze(-1).tolist() == [[10.0, 20.0, 30.0]]
    assert aligned_mask.tolist() == [[True, True, True]]


def test_uniform_first_last_selects_five_boundaries_from_seven_steps():
    selected = select_teacher_step_indices(
        7,
        max_groups=5,
        selection="uniform_first_last",
    )

    assert selected[0] == 0
    assert selected[-1] == 6
    assert len(selected) == 5
    assert selected == sorted(set(selected))


def test_fixed_full_latent_count_keeps_ten_student_latents():
    teacher_targets = TeacherTargetBatch(
        target_states=torch.zeros((2, 7, 3)),
        target_mask=torch.tensor(
            [
                [True, True, True, False, False, False, False],
                [True, True, True, True, True, True, True],
            ]
        ),
    )
    args = SimpleNamespace(
        objective="step_trajectory",
        latent_count_mode="fixed_full",
        num_latent_steps=10,
    )

    assert latent_counts_for_objective(teacher_targets, args) == [10, 10]


def test_final_answer_eval_input_uses_configured_eval_latent_count():
    class TinyTokenizer:
        eos_token_id = 99

        def convert_tokens_to_ids(self, token):
            return {
                "<|start-latent|>": 10,
                "<|latent|>": 11,
                "<|end-latent|>": 12,
            }[token]

        def encode(self, text, add_special_tokens=True):
            return [2, 3]

    input_ids = build_final_answer_eval_input_ids(
        TinyTokenizer(),
        "q",
        num_latent_steps=10,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist()[0].count(11) == 10


def test_load_coconut_initial_checkpoint_strict_loads_student_state(tmp_path):
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_causallm = nn.Linear(2, 2)

    student = TinyStudent()
    state = {
        key: torch.ones_like(value)
        for key, value in student.state_dict().items()
    }
    checkpoint = tmp_path / "coconut.pt"
    torch.save(state, checkpoint)

    info = load_coconut_initial_checkpoint(
        checkpoint,
        student=student,
        device=torch.device("cpu"),
    )

    assert info == {"loaded_keys": len(state), "load_target": "coconut_student"}
    for value in student.state_dict().values():
        assert torch.allclose(value, torch.ones_like(value))
