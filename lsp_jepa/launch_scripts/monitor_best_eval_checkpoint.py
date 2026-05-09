#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    "/root/autodl-tmp/lsp_jepa/runs/"
    "lsp_step_trajectory_gpt2_epoch35_gsm8k_aug_full35/"
    "steptraj_epoch35_gpt2_gsm8k_aug_full35_20260430"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preserve the best available eval checkpoint from a running LSP-JEPA run."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "accuracy" in row and "checkpoint_path" in row:
                rows.append(row)
    return rows


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def better(left: dict[str, Any], right: dict[str, Any] | None) -> bool:
    if right is None:
        return True
    left_acc = float(left.get("accuracy", -1.0))
    right_acc = float(right.get("accuracy", -1.0))
    if left_acc != right_acc:
        return left_acc > right_acc
    return int(left.get("step", -1)) > int(right.get("step", -1))


def select_best(rows: list[dict[str, Any]], *, require_existing_checkpoint: bool) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in rows:
        checkpoint_path = Path(str(row.get("checkpoint_path", "")))
        if require_existing_checkpoint and not checkpoint_path.exists():
            continue
        if better(row, best):
            best = row
    return best


def copy_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def summarize_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "step": int(row["step"]),
        "epoch_index": row.get("epoch_index"),
        "epoch_steps": row.get("epoch_steps"),
        "accuracy": float(row["accuracy"]),
        "exact_match": float(row.get("exact_match", row["accuracy"])),
        "invalid_answer_rate": float(row.get("invalid_answer_rate", 0.0)),
        "num_eval_samples": int(row.get("num_eval_samples", 0)),
        "checkpoint_path": str(row.get("checkpoint_path", "")),
        "step_metrics_path": str(row.get("step_metrics_path", "")),
    }


def monitor_once(run_dir: Path) -> str:
    eval_metrics = run_dir / "evals" / "metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    best_ckpt = checkpoint_dir / "best_eval_accuracy.pt"
    state_path = checkpoint_dir / "best_eval_accuracy.json"
    rows = read_jsonl(eval_metrics)
    if not rows:
        return "no eval rows yet"

    historical_best = select_best(rows, require_existing_checkpoint=False)
    best_available = select_best(rows, require_existing_checkpoint=True)
    if best_available is None:
        metadata = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "no eval checkpoint currently exists to copy",
            "historical_best": summarize_row(historical_best),
            "best_available": None,
            "preserved_checkpoint_path": str(best_ckpt),
        }
        state_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return "no existing eval checkpoint to copy"

    state = load_state(state_path)
    copied = state.get("best_available") if isinstance(state.get("best_available"), dict) else None
    should_copy = not best_ckpt.exists()
    if copied is None:
        should_copy = True
    else:
        should_copy = better(best_available, copied)
    if copied is not None and int(copied.get("step", -1)) == int(best_available["step"]):
        should_copy = should_copy or not best_ckpt.exists()

    source = Path(str(best_available["checkpoint_path"]))
    action = "kept"
    if should_copy:
        copy_checkpoint(source, best_ckpt)
        action = "copied"

    metadata = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": action,
        "historical_best": summarize_row(historical_best),
        "historical_best_checkpoint_exists": Path(str(historical_best.get("checkpoint_path", ""))).exists()
        if historical_best
        else False,
        "best_available": summarize_row(best_available),
        "source_checkpoint_path": str(source),
        "preserved_checkpoint_path": str(best_ckpt),
        "preserved_checkpoint_size": best_ckpt.stat().st_size if best_ckpt.exists() else None,
    }
    state_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return (
        f"{action} best available eval checkpoint: "
        f"step={best_available['step']} acc={float(best_available['accuracy']):.6f}"
    )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    while True:
        try:
            message = monitor_once(run_dir)
        except Exception as exc:  # noqa: BLE001 - this is a long-running guard process.
            message = f"error: {exc!r}"
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
