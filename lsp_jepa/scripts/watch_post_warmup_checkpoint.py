from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        status = update_best_post_warmup_checkpoint(
            run_dir=args.run_dir,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
        )
        print(json.dumps(status, sort_keys=True), flush=True)
        if args.once:
            return
        time.sleep(max(1.0, args.interval_seconds))


def update_best_post_warmup_checkpoint(
    *,
    run_dir: Path,
    warmup_steps: int,
    warmup_ratio: float | None,
) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    latest_pt = checkpoint_dir / "latest.pt"
    latest_json = checkpoint_dir / "latest.json"
    eval_metrics_path = run_dir / "evals" / "metrics.jsonl"
    best_pt = checkpoint_dir / "best_post_warmup_eval_accuracy.pt"
    best_json = checkpoint_dir / "best_post_warmup_eval_accuracy.json"

    if not latest_pt.exists() or not latest_json.exists():
        return {"status": "waiting_for_latest_checkpoint"}
    if not eval_metrics_path.exists():
        return {"status": "waiting_for_eval_metrics"}

    latest_metadata = load_json(latest_json)
    latest_step = int(latest_metadata.get("step", -1))
    if latest_step < 0:
        return {"status": "latest_step_missing"}
    if not is_post_warmup_step(latest_step, warmup_steps):
        return {
            "status": "waiting_for_post_warmup_step",
            "latest_step": latest_step,
            "warmup_steps": warmup_steps,
        }

    eval_by_step = {
        int(record["step"]): record
        for record in iter_eval_records(eval_metrics_path)
        if isinstance(record.get("step"), int)
        and isinstance(record.get("accuracy"), (int, float))
    }
    latest_eval = eval_by_step.get(latest_step)
    if latest_eval is None:
        return {
            "status": "waiting_for_matching_eval",
            "latest_step": latest_step,
            "latest_eval_step": max(eval_by_step) if eval_by_step else None,
        }

    latest_accuracy = float(latest_eval["accuracy"])
    best_accuracy = None
    if best_json.exists():
        best_metadata = load_json(best_json)
        metric = best_metadata.get("accuracy")
        if isinstance(metric, (int, float)):
            best_accuracy = float(metric)
    if best_accuracy is not None and latest_accuracy <= best_accuracy:
        return {
            "status": "no_update",
            "latest_step": latest_step,
            "latest_accuracy": latest_accuracy,
            "best_accuracy": best_accuracy,
        }

    replace_checkpoint_alias(latest_pt, best_pt)
    metadata = {
        "step": latest_step,
        "global_step": latest_step,
        "epoch_index": latest_eval.get("epoch_index"),
        "epoch_steps": latest_eval.get("epoch_steps"),
        "accuracy": latest_accuracy,
        "exact_match": latest_eval.get("exact_match"),
        "invalid_answer_rate": latest_eval.get("invalid_answer_rate"),
        "num_eval_samples": latest_eval.get("num_eval_samples"),
        "eval_metrics_path": str(eval_metrics_path),
        "step_metrics_path": latest_eval.get("step_metrics_path"),
        "source_checkpoint_path": str(latest_pt),
        "checkpoint_path": str(best_pt),
        "lsp_weight": latest_eval.get("lsp_weight"),
        "lsp_weight_target": latest_eval.get("lsp_weight_target"),
        "lsp_weight_warmup_ratio": warmup_ratio,
        "lsp_weight_warmup_steps": warmup_steps,
        "lsp_weight_warmup_progress": latest_eval.get("lsp_weight_warmup_progress"),
        "post_warmup_eval_eligible": True,
        "purpose": "highest eval accuracy checkpoint after LSP weight warmup",
        "managed_by": "post_warmup_checkpoint_watcher",
        "note": (
            "For already-running jobs this watcher can only preserve checkpoints "
            "that still exist when it observes them."
        ),
    }
    write_json_atomic(best_json, metadata)
    return {
        "status": "updated",
        "latest_step": latest_step,
        "latest_accuracy": latest_accuracy,
        "previous_best_accuracy": best_accuracy,
        "checkpoint_path": str(best_pt),
    }


def is_post_warmup_step(step: int, warmup_steps: int) -> bool:
    if warmup_steps <= 0:
        return step > 0
    return step > warmup_steps


def iter_eval_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def replace_checkpoint_alias(source: Path, alias: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    tmp = alias.with_name(f".{alias.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        try:
            os.link(source, tmp)
        except OSError:
            os.symlink(source, tmp)
        os.replace(tmp, alias)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
