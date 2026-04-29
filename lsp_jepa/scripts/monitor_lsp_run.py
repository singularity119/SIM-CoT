#!/usr/bin/env python3
"""Monitor an LSP-JEPA training run and surface likely interruption causes.

The script is intentionally dependency-free. It reads local run artifacts,
process state, GPU state, and recent logs, then prints a compact status report.
Use --watch SEC for repeated polling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/lsp_jepa/runs/lsp_step_trajectory_gpt2_epoch35")
ERROR_PATTERNS = [
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "out of memory",
    "OutOfMemoryError",
    "Killed",
    "SIGTERM",
    "SIGKILL",
    "KeyboardInterrupt",
    "No space left",
    "Input/output error",
    "NCCL",
    "IndexError",
    "ValueError",
    "AssertionError",
    "nan",
    "NaN",
    "inf",
]


@dataclass
class ProcessInfo:
    pid: int
    ppid: int | None
    stat: str
    command: str


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "missing"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def fmt_float(value: Any, digits: int = 6) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}g}"


def run_command(args: list[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()

    root = args.output_root.resolve()
    latest_main = root / "latest_main.log"
    if latest_main.exists():
        resolved = latest_main.resolve()
        if resolved.parent.name == "logs":
            return resolved.parent.parent

    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith("steptraj_")
    ] if root.exists() else []
    if not candidates:
        raise SystemExit(
            f"could not infer run dir; pass --run-dir or create {latest_main}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_jsonl_tail(path: Path, max_lines: int = 2000) -> tuple[int, dict[str, Any] | None, list[str]]:
    if not path.exists():
        return 0, None, []
    lines = tail_lines(path, max_lines)
    parsed_rows: list[dict[str, Any]] = []
    bad_lines: list[str] = []
    total_rows = 0
    # Count rows without loading the whole file into memory.
    with path.open("rb") as fh:
        for _ in fh:
            total_rows += 1
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            bad_lines.append(text[:240])
            continue
        if isinstance(row, dict):
            parsed_rows.append(row)
    return total_rows, (parsed_rows[-1] if parsed_rows else None), bad_lines[-5:]


def tail_lines(path: Path, max_lines: int = 100) -> list[str]:
    if not path.exists():
        return []
    # Simple bounded tail implementation that avoids reading very large logs.
    block_size = 8192
    data = b""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        while pos > 0 and data.count(b"\n") <= max_lines:
            read_size = min(block_size, pos)
            pos -= read_size
            fh.seek(pos)
            data = fh.read(read_size) + data
    return data.decode(errors="replace").splitlines()[-max_lines:]


def find_main_log(run_dir: Path, output_root: Path) -> Path | None:
    latest_main = output_root / "latest_main.log"
    if latest_main.exists():
        resolved = latest_main.resolve()
        if run_dir in resolved.parents:
            return resolved
    log_dir = run_dir / "logs"
    logs = sorted(log_dir.glob("run_*_max_steps_*.log"), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def scan_log(log_path: Path | None, max_lines: int = 1200) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {"progress": [], "suspicious": [], "tail": []}
    lines = tail_lines(log_path, max_lines)
    progress = [line for line in lines if line.startswith("step_trajectory:") or line.startswith("sequence:")]
    suspicious = [
        line for line in lines
        if any(pattern in line for pattern in ERROR_PATTERNS)
    ]
    return {
        "progress": progress[-8:],
        "suspicious": suspicious[-20:],
        "tail": lines[-20:],
    }


def find_processes(run_dir: Path, run_name: str, explicit_pid: int | None) -> list[ProcessInfo]:
    output = run_command(["ps", "-eo", "pid=,ppid=,stat=,args="], timeout=5.0)
    processes: list[ProcessInfo] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        stat = parts[2]
        command = parts[3]
        matched = False
        if explicit_pid is not None and pid == explicit_pid:
            matched = True
        if "train_lsp_jepa_core.py" in command and str(run_dir) in command:
            matched = True
        if run_name and run_name in command and (
            "train_lsp_jepa_core.py" in command or "launch_step_trajectory_train.sh" in command
        ):
            matched = True
        if matched:
            processes.append(ProcessInfo(pid=pid, ppid=ppid, stat=stat, command=command))
    return processes


def gpu_status() -> list[dict[str, str]]:
    output = run_command([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"pid": parts[0], "process": parts[1], "used_memory_mib": parts[2]})
    return rows


def checkpoints_status(run_dir: Path) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    step_files = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda path: path.stat().st_mtime)
    best_meta = load_json(checkpoint_dir / "best_total_loss.json")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "step_checkpoint_count": len(step_files),
        "latest_checkpoint": str(step_files[-1]) if step_files else None,
        "latest_checkpoint_age_sec": age_seconds(step_files[-1]) if step_files else None,
        "best_total_loss": best_meta,
    }


def eval_status(run_dir: Path) -> dict[str, Any]:
    eval_metrics = run_dir / "evals" / "metrics.jsonl"
    total, last, bad = load_jsonl_tail(eval_metrics, max_lines=200)
    return {
        "metrics_path": str(eval_metrics),
        "eval_rows": total,
        "latest_eval": last,
        "bad_lines": bad,
    }


def disk_status(path: Path) -> dict[str, Any]:
    target = path if path.exists() else path.parent
    usage = shutil.disk_usage(target)
    return {
        "path": str(target),
        "free_gb": usage.free / (1024 ** 3),
        "used_gb": usage.used / (1024 ** 3),
        "total_gb": usage.total / (1024 ** 3),
        "free_pct": usage.free / usage.total * 100.0 if usage.total else 0.0,
    }


def infer_state(
    *,
    processes: list[ProcessInfo],
    metrics_age: float | None,
    log_scan: dict[str, Any],
    summary: dict[str, Any] | None,
    stale_seconds: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if summary is not None and summary.get("acceptance", {}).get("passed"):
        return "completed", ["summary acceptance passed"]
    if processes:
        state = "running"
    else:
        state = "stopped"
        reasons.append("no matching train_lsp_jepa_core.py process found")
    if metrics_age is None:
        reasons.append("metrics.jsonl missing")
    elif metrics_age > stale_seconds:
        state = "stale" if processes else "stopped"
        reasons.append(f"metrics stale for {fmt_age(metrics_age)}")
    suspicious = log_scan.get("suspicious") or []
    if suspicious:
        reasons.append(f"{len(suspicious)} suspicious recent log lines")
    return state, reasons


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = resolve_run_dir(args)
    output_root = args.output_root.resolve()
    run_name = run_dir.name
    main_log = find_main_log(run_dir, output_root)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.yaml"
    row_count, latest_metrics, bad_metric_lines = load_jsonl_tail(metrics_path)
    log_scan = scan_log(main_log, max_lines=args.log_scan_lines)
    processes = find_processes(run_dir, run_name, args.pid)
    process_pids = {str(proc.pid) for proc in processes}
    gpus = gpu_status()
    matching_gpus = [row for row in gpus if row.get("pid") in process_pids]
    summary = load_json(summary_path)
    stale_seconds = args.stale_minutes * 60.0
    state, reasons = infer_state(
        processes=processes,
        metrics_age=age_seconds(metrics_path),
        log_scan=log_scan,
        summary=summary,
        stale_seconds=stale_seconds,
    )
    return {
        "timestamp_utc": utc_now().isoformat(),
        "state": state,
        "reasons": reasons,
        "run": {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "main_log": str(main_log) if main_log else None,
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "config_path": str(config_path),
        },
        "progress": {
            "metrics_rows": row_count,
            "latest_metrics": latest_metrics,
            "metrics_age_sec": age_seconds(metrics_path),
            "bad_metric_lines": bad_metric_lines,
            "recent_progress_lines": log_scan["progress"][-args.tail_progress:],
        },
        "processes": [proc.__dict__ for proc in processes],
        "gpu": {
            "matching": matching_gpus,
            "all_compute": gpus,
        },
        "checkpoints": checkpoints_status(run_dir),
        "eval": eval_status(run_dir),
        "disk": {
            "run_dir": disk_status(run_dir),
            "autodl_tmp": disk_status(Path("/root/autodl-tmp")),
        },
        "logs": {
            "main_log_age_sec": age_seconds(main_log) if main_log else None,
            "suspicious": log_scan["suspicious"],
            "tail": log_scan["tail"][-args.tail_log:],
        },
    }


def progress_line(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "progress: no metrics yet"
    step = int(metrics.get("step") or 0)
    max_steps = int(metrics.get("max_steps") or 0)
    pct = step / max_steps * 100.0 if max_steps else 0.0
    epoch = metrics.get("epoch_index", "n/a")
    epoch_steps = int(metrics.get("epoch_steps") or 0)
    epoch_step = ((step - 1) % epoch_steps + 1) if step and epoch_steps else "n/a"
    return (
        f"progress: step {step}/{max_steps} ({pct:.2f}%), "
        f"epoch {epoch}, epoch_step {epoch_step}/{epoch_steps or 'n/a'}, "
        f"total_loss={fmt_float(metrics.get('total_loss'))}, "
        f"lsp_loss={fmt_float(metrics.get('lsp_loss'))}, "
        f"host_answer_ce={fmt_float(metrics.get('host_answer_ce'))}, "
        f"latent_var={fmt_float(metrics.get('latent_variance_mean'))}"
    )


def print_report(snapshot: dict[str, Any]) -> None:
    run = snapshot["run"]
    progress = snapshot["progress"]
    checkpoints = snapshot["checkpoints"]
    latest_eval = snapshot["eval"]["latest_eval"]
    print("=" * 100)
    print(f"LSP-JEPA monitor @ {snapshot['timestamp_utc']}")
    print(f"state: {snapshot['state']}  reasons: {', '.join(snapshot['reasons']) or 'none'}")
    print(f"run_dir: {run['run_dir']}")
    print(f"main_log: {run['main_log']}")
    print(f"metrics: {run['metrics_path']} age={fmt_age(progress['metrics_age_sec'])} rows={progress['metrics_rows']}")
    print(progress_line(progress["latest_metrics"]))
    print(
        "processes: "
        + (
            ", ".join(f"{proc['pid']}({proc['stat']})" for proc in snapshot["processes"])
            if snapshot["processes"]
            else "none"
        )
    )
    if snapshot["gpu"]["matching"]:
        gpu_bits = [
            f"pid={row['pid']} mem={row['used_memory_mib']}MiB"
            for row in snapshot["gpu"]["matching"]
        ]
        print("gpu: " + ", ".join(gpu_bits))
    else:
        print("gpu: no matching compute process")
    disk = snapshot["disk"]["autodl_tmp"]
    print(f"disk /root/autodl-tmp: free={disk['free_gb']:.1f}GB ({disk['free_pct']:.1f}%)")
    print(
        "checkpoints: "
        f"count={checkpoints['step_checkpoint_count']} "
        f"latest={checkpoints['latest_checkpoint']} "
        f"latest_age={fmt_age(checkpoints['latest_checkpoint_age_sec'])}"
    )
    if checkpoints["best_total_loss"]:
        print(f"best_total_loss: {json.dumps(checkpoints['best_total_loss'], sort_keys=True)}")
    if latest_eval:
        print(
            "latest_eval: "
            f"step={latest_eval.get('step')} "
            f"accuracy={fmt_float(latest_eval.get('accuracy'))} "
            f"exact_match={fmt_float(latest_eval.get('exact_match'))} "
            f"invalid={fmt_float(latest_eval.get('invalid_answer_rate'))}"
        )
    if progress["recent_progress_lines"]:
        print("recent progress:")
        for line in progress["recent_progress_lines"]:
            print(f"  {line}")
    if snapshot["logs"]["suspicious"]:
        print("suspicious recent log lines:")
        for line in snapshot["logs"]["suspicious"][-10:]:
            print(f"  {line}")
    if progress["bad_metric_lines"]:
        print("bad metric json lines:")
        for line in progress["bad_metric_lines"]:
            print(f"  {line}")


def write_snapshot(snapshot: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--watch", type=float, default=0.0, help="Poll every N seconds until interrupted.")
    parser.add_argument("--stale-minutes", type=float, default=15.0)
    parser.add_argument("--tail-progress", type=int, default=6)
    parser.add_argument("--tail-log", type=int, default=12)
    parser.add_argument("--log-scan-lines", type=int, default=1200)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal between --watch reports.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the text report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        snapshot = build_snapshot(args)
        json_out = args.json_out
        if json_out is None:
            json_out = Path(snapshot["run"]["run_dir"]) / "monitor" / "status.json"
        write_snapshot(snapshot, json_out)
        if args.watch > 0 and not args.no_clear and not args.json:
            print("\033[2J\033[H", end="")
        if args.json:
            print(json.dumps(snapshot, indent=2, sort_keys=True))
        else:
            print_report(snapshot)
            print(f"status_json: {json_out}")
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
