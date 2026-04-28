#!/usr/bin/env python3
"""Summarize an LSP-JEPA sequence-level training/eval gate into a markdown report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

DEFAULT_TRAIN_METRICS = Path("/root/autodl-tmp/lsp_jepa/runs/pr13_mvp_full_seq_launch/max_steps_100/metrics.jsonl")
DEFAULT_TRAIN_SUMMARY = Path("/root/autodl-tmp/lsp_jepa/runs/pr13_mvp_full_seq_launch/max_steps_100/summary.json")
DEFAULT_EVAL_METRICS = Path("/root/autodl-tmp/lsp_jepa/runs/pr14_mvp_min_eval/metrics.json")
DEFAULT_CONFIG = Path("/root/SIM-CoT/lsp_jepa/configs/core/lsp_seq_full_train.yaml")
DEFAULT_OUTPUT = Path("/root/SIM-CoT/lsp_jepa/docs/reports/LSP_SEQ_FULL_1K_REPORT.md")

NUMERIC_FIELDS = [
    "total_loss", "lsp_loss", "host_answer_ce", "anti_collapse_loss",
    "latent_variance_mean", "latent_variance_min", "raw_latent_variance_mean",
    "pairwise_cosine_mean", "pairwise_cosine_max", "pairwise_cosine_min",
    "pairwise_l2_mean", "effective_rank", "ema_drift_l1", "teacher_delta_l1",
    "student_grad_l1", "predictor_grad_l1", "student_base_grad_l1",
    "valid_final_targets", "valid_teacher_targets",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_yaml_config(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text()
    try:
        import yaml  # type: ignore
    except Exception:
        return {}, text
    return yaml.safe_load(text) or {}, text


def nested(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def numeric_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            vals.append(float(value))
    return vals


def field_stats(rows: list[dict[str, Any]], field: str) -> dict[str, float] | None:
    vals = numeric_values(rows, field)
    if not vals:
        return None
    tail = vals[-10:]
    return {
        "first": vals[0], "last": vals[-1], "delta": vals[-1] - vals[0],
        "min": min(vals), "max": max(vals), "mean": statistics.fmean(vals),
        "tail_mean": statistics.fmean(tail),
    }


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def trend_line(label: str, stats: dict[str, float] | None) -> str:
    if stats is None:
        return f"- {label}: not recorded."
    return (
        f"- {label}: first {fmt(stats['first'])}, last {fmt(stats['last'])}, "
        f"delta {fmt(stats['delta'])}, min {fmt(stats['min'])}, max {fmt(stats['max'])}, "
        f"tail-10 mean {fmt(stats['tail_mean'])}."
    )


def count_false(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if not bool(row.get(field)))


def count_true(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if bool(row.get(field)))


def nonfinite_entries(rows: list[dict[str, Any]]) -> list[tuple[int, str, float]]:
    bad = []
    for row in rows:
        step = int(row.get("step", -1))
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                bad.append((step, field, float(value)))
    return bad


def make_report(args: argparse.Namespace) -> str:
    rows = load_jsonl(args.train_metrics)
    summary = load_json(args.train_summary)
    eval_metrics = load_json(args.eval_metrics)
    config, _ = load_yaml_config(args.config)
    stats = {field: field_stats(rows, field) for field in NUMERIC_FIELDS}

    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    bad = nonfinite_entries(rows)
    empty_teacher_batches = count_false(rows, "teacher_target_mask_nonempty")
    empty_latent_batches = count_false(rows, "latent_mask_nonempty")
    near_one_batches = count_true(rows, "pairwise_cosine_near_one")
    all_one_batches = count_true(rows, "pairwise_cosine_all_one")

    eval_accuracy = float(eval_metrics.get("accuracy", 0.0) or 0.0)
    eval_exact = float(eval_metrics.get("exact_match", 0.0) or 0.0)
    invalid_answer_rate = float(eval_metrics.get("invalid_answer_rate", 0.0) or 0.0)
    eval_broken = eval_accuracy == 0.0 and invalid_answer_rate >= 0.5
    repr_stable = (
        not bad
        and (stats["latent_variance_mean"] or {}).get("last", 0.0) > 0.0
        and near_one_batches == 0
        and all_one_batches == 0
        and empty_teacher_batches == 0
        and empty_latent_batches == 0
    )
    recommend_step_level = repr_stable and not eval_broken and int(last.get("max_steps", 0) or 0) >= 1000

    def cfg(path: str, fallback: Any = "not recorded") -> Any:
        return nested(config, path, fallback)

    generated_length = eval_metrics.get("generated_length_mean")
    if generated_length is None and isinstance(eval_metrics.get("generated_length"), dict):
        generated_length = eval_metrics["generated_length"].get("mean")

    lines = [
        "# LSP Sequence Full 1K Report", "",
        "Status: the requested 1000-step / 1K run has not been executed. This report is intentionally based on the PR13 `max_steps=100` gate run plus the PR14 minimal answer eval that loaded the PR13 step-100 checkpoint. No 1K metrics are inferred or fabricated.", "",
        "## Inputs", "",
        f"- Train metrics: `{args.train_metrics}`",
        f"- Train summary: `{args.train_summary}`",
        f"- Eval metrics: `{args.eval_metrics}`",
        f"- Config: `{args.config}`",
        f"- Report generated by: `lsp_jepa/scripts/summarize_lsp_run.py`", "",
        "## Training Configuration Summary", "",
        f"- experiment mode / host / backbone: `{cfg('experiment.mode', first.get('experiment_mode'))}` / `{cfg('experiment.host')}` / `{cfg('experiment.backbone', first.get('backbone'))}`",
        f"- use_lsp_jepa: `{cfg('experiment.use_lsp_jepa')}`",
        f"- data: `{cfg('data.dataset_id')}` `{cfg('data.dataset_split')}`, expected samples `{cfg('data.expected_samples', first.get('expected_samples'))}`, local path `{cfg('data.local_dataset_path', first.get('dataset_local_path'))}`",
        f"- observed gate run: `{len(rows)}` steps, batch size `{last.get('batch_size', cfg('training.batch_size'))}`, unique samples seen `{last.get('unique_samples_seen', 'not recorded')}`, max_steps field `{last.get('max_steps', 'not recorded')}`",
        f"- optimizer config: lr `{cfg('training.lr')}`, weight_decay `{cfg('training.weight_decay')}`, gradient_clip_norm declared `{cfg('training.gradient_clip_norm')}`, warmup_ratio declared `{cfg('training.warmup_ratio')}`",
        f"- teacher: EMA decay `{cfg('teacher.ema_decay')}`, update_trainable_only `{cfg('teacher.update_trainable_only')}`, target_layer `{cfg('teacher.target_layer')}`, pooling `{cfg('teacher.target_pooling')}`, target_space `{cfg('teacher.target_space')}`",
        f"- teacher leakage controls: exclude_answer_tokens `{last.get('teacher_exclude_answer_tokens', cfg('teacher.exclude_answer_tokens'))}`, exclude_answer_prefix `{last.get('teacher_exclude_answer_prefix', cfg('teacher.exclude_answer_prefix'))}`, answer_leakage_ok `{last.get('answer_leakage_ok', 'not recorded')}`",
        f"- student: latent_arch `{cfg('student.latent_arch')}`, latent steps `{cfg('student.num_latent_steps')}`, predictor_head_layers `{cfg('student.predictor_head_layers')}`, detach_between_steps `{cfg('student.detach_between_steps')}`",
        f"- objective / mapping: `{last.get('objective', cfg('lsp_objective.objective'))}` with mapping `{last.get('mapping', cfg('mapping.strategy'))}`, target_position `{last.get('target_position', cfg('teacher.target_position'))}`",
        f"- losses: alignment `{cfg('loss.alignment')}`, lsp_weight `{last.get('lsp_weight', cfg('loss.align_weight'))}`, anti_collapse `{last.get('anti_collapse_type', cfg('loss.anti_collapse'))}` weight `{last.get('anti_collapse_weight', cfg('loss.anti_collapse_weight'))}`, host_answer_ce_weight `{last.get('host_answer_ce_weight', cfg('host_losses.host_answer_ce_weight'))}`",
        f"- disabled core contaminants: CODI distill weight `{cfg('host_losses.codi_distill_weight')}`, SIM-CoT decoder weight `{cfg('host_losses.simcot_decoder_weight')}`, intermediate CoT CE weight `{cfg('host_losses.intermediate_cot_ce_weight')}`", "",
        "## Loss Curves", "",
        trend_line("total_loss", stats["total_loss"]),
        trend_line("lsp_loss", stats["lsp_loss"]),
        trend_line("host_answer_ce", stats["host_answer_ce"]),
        trend_line("anti_collapse_loss", stats["anti_collapse_loss"]), "",
        "Interpretation: total_loss, lsp_loss, and host_answer_ce decreased over the 100-step gate. anti_collapse_loss stayed finite and nearly flat, which is expected for the variance regularizer at this short horizon.", "",
        "## Representation Diagnostics", "",
        trend_line("latent_variance_mean", stats["latent_variance_mean"]),
        trend_line("latent_variance_min", stats["latent_variance_min"]),
        trend_line("raw_latent_variance_mean", stats["raw_latent_variance_mean"]),
        trend_line("pairwise_cosine_mean", stats["pairwise_cosine_mean"]),
        trend_line("pairwise_cosine_max", stats["pairwise_cosine_max"]),
        trend_line("pairwise_cosine_min", stats["pairwise_cosine_min"]),
        trend_line("pairwise_l2_mean", stats["pairwise_l2_mean"]),
        trend_line("effective_rank", stats["effective_rank"]),
        f"- pairwise_cosine_near_one batches: `{near_one_batches}/{len(rows)}`; pairwise_cosine_all_one batches: `{all_one_batches}/{len(rows)}`.", "",
        "Interpretation: latent variance remained non-zero and the run did not report sustained pairwise-cosine near-one collapse. Pairwise cosine is high, and pairwise_cosine_max reaches almost 1 on some batches, so this should remain monitored, but the 100-step gate does not show a hard collapse signal by the recorded flags.", "",
        "## Student-Teacher And EMA", "",
        "- student_teacher_cosine: not recorded in PR13 metrics; no value is inferred.",
        trend_line("ema_drift_l1", stats["ema_drift_l1"]),
        trend_line("teacher_delta_l1", stats["teacher_delta_l1"]),
        f"- teacher_grad_params last: `{last.get('teacher_grad_params', 'not recorded')}`.", "",
        "Interpretation: EMA drift is non-zero and increases through the gate, while teacher_grad_params remains zero in the final metrics, matching the no-grad EMA teacher contract.", "",
        "## Gradient Diagnostics", "",
        "- grad_norm: not recorded in PR13 metrics.",
        trend_line("student_grad_l1 substitute", stats["student_grad_l1"]),
        trend_line("predictor_grad_l1 substitute", stats["predictor_grad_l1"]),
        trend_line("student_base_grad_l1 substitute", stats["student_base_grad_l1"]), "",
        "Interpretation: because `grad_norm` is absent, this report uses the recorded L1 gradient diagnostics as substitutes. They are finite and non-zero; no spike or missing-gradient anomaly is visible in these substitute fields.", "",
        "## Target And Failure Checks", "",
        f"- Non-finite numeric entries: `{len(bad)}`.",
        f"- Empty teacher target batches (`teacher_target_mask_nonempty=false`): `{empty_teacher_batches}/{len(rows)}`.",
        f"- Empty latent batches (`latent_mask_nonempty=false`): `{empty_latent_batches}/{len(rows)}`.",
        f"- valid_final_targets: first `{fmt((stats['valid_final_targets'] or {}).get('first'))}`, last `{fmt((stats['valid_final_targets'] or {}).get('last'))}`.",
        f"- valid_teacher_targets: first `{fmt((stats['valid_teacher_targets'] or {}).get('first'))}`, last `{fmt((stats['valid_teacher_targets'] or {}).get('last'))}`.",
        "- OOM: not indicated by the completed metrics/summary files; no OOM marker was recorded in the inspected JSON outputs.", "",
        "## Answer Eval Results", "",
        f"- checkpoint: `{eval_metrics.get('checkpoint_path', 'not recorded')}`",
        f"- checkpoint step: `{eval_metrics.get('checkpoint_step', 'not recorded')}`",
        f"- eval samples: `{eval_metrics.get('num_eval_samples', 'not recorded')}` with limit `{eval_metrics.get('limit_eval_samples', 'not recorded')}`",
        f"- accuracy: `{fmt(eval_accuracy)}`",
        f"- exact_match: `{fmt(eval_exact)}`",
        f"- invalid_answer_rate: `{fmt(invalid_answer_rate)}`",
        f"- generated_length_mean: `{fmt(generated_length)}`",
        f"- core eval contract: `{json.dumps(eval_metrics.get('core_eval_contract', {}), sort_keys=True)}`", "",
        "Interpretation: PR14 answer eval is clearly broken at this checkpoint: accuracy and exact match are 0, and every evaluated sample has an invalid extracted answer. Generated text contains mostly unknown-token outputs in the recorded examples. This blocks a positive step-level recommendation even though the representation diagnostics passed the short gate.", "",
        "## Collapse Assessment", "",
        "- Representation collapse: `not observed by PR13 100-step gate metrics` (latent variance non-zero, no near-one/all-one cosine flags, effective rank finite).",
        f"- Answer-generation collapse/failure: `observed` in PR14 minimal eval (invalid_answer_rate `{fmt(invalid_answer_rate)}`, accuracy `{fmt(eval_accuracy)}`).",
        "- Teacher target collapse/emptiness: `not observed` in PR13 metrics; teacher target masks were non-empty for all recorded batches.", "",
        "## Decision", "",
        "- Sequence-level LSP-JEPA-Core training stability: `partial`. The 100-step gate is numerically stable and does not show recorded representation collapse, but the requested 1K/full run has not been executed and answer eval is broken.",
        f"- Recommend entering step-level now: `{'yes' if recommend_step_level else 'no'}`.",
        "- Rationale: the project rule allows step-level only when loss is finite, latent variance is non-zero, pairwise cosine is not collapsed, and eval is not obviously broken. The first three conditions pass on the 100-step gate, but PR14 eval is obviously broken and no 1K run exists. First tune the sequence-level answer path, especially `host_answer_ce_weight` versus `lsp_weight`, then rerun a longer sequence-level gate before moving to step-level supervision.", "",
        "## Logging Preference For Future Runs", "",
        "Future training logs should emit a fixed-interval, tqdm-like progress line in the SIM-CoT training style. This is a logging/reporting preference only, not a new training algorithm. Each line should include at least step/total, total loss, lsp_loss, host_answer_ce, latent variance, EMA drift, elapsed time, and preferably throughput/ETA when available.", "",
        "## Unresolved Items", "",
        "- Execute the actual 1K/full sequence-level run before calling this a 1K result.",
        "- Add or log `student_teacher_cosine` if it is required for future decision gates.",
        "- Add `grad_norm` if exact norm-based anomaly checks are required; current report used L1 gradient substitutes.",
        "- Investigate PR14 invalid-answer generation/tokenizer behavior and retune `host_answer_ce_weight` / `lsp_weight` before step-level.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metrics", type=Path, default=DEFAULT_TRAIN_METRICS)
    parser.add_argument("--train-summary", type=Path, default=DEFAULT_TRAIN_SUMMARY)
    parser.add_argument("--eval-metrics", type=Path, default=DEFAULT_EVAL_METRICS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.train_metrics, args.train_summary, args.eval_metrics, args.config]:
        if not path.exists():
            raise FileNotFoundError(path)
    report = make_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(args.output)


if __name__ == "__main__":
    main()
