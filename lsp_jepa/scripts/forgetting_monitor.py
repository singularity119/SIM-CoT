#!/usr/bin/env python3
"""Conservative catastrophic-forgetting monitor for Coconut -> LSP-JEPA runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INIT_COCONUT_CHECKPOINT = Path(
    "/root/autodl-tmp/simcot_coconut/ckpts/"
    "gsm_coconut_from_best_accuracy_cachefix2_bs32_restart_20260513_110520/"
    "best_full_latent_eval_accuracy.pt"
)
ACTION_EXIT_CODE = 20


@dataclass(frozen=True)
class ForgettingThresholds:
    warn_accuracy_drop: float = 0.05
    warn_lsp_to_ce_ratio: float = 1.5
    min_completed_epochs: int = 5
    restart_accuracy_drop: float = 0.06
    restart_best_drop: float = 0.04
    recovery_tolerance: float = 0.01
    max_auto_restarts: int = 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def row_accuracy(row: dict[str, Any]) -> float | None:
    return numeric(row.get("accuracy"))


def epoch_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if not row.get("is_step0_baseline")
        and isinstance(row.get("step"), int)
        and int(row["step"]) > 0
        and row_accuracy(row) is not None
    ]
    return sorted(filtered, key=lambda row: (int(row.get("step") or 0), int(row.get("epoch_index") or 0)))


def find_step0_baseline(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if row.get("is_step0_baseline") and row_accuracy(row) is not None:
            return row
    step0 = read_json(run_dir / "evals" / "step_000000.json")
    if step0.get("is_step0_baseline") and row_accuracy(step0) is not None:
        return step0
    return None


def best_eval_accuracy(run_dir: Path, epoch_rows: list[dict[str, Any]]) -> float | None:
    metadata = read_json(run_dir / "checkpoints" / "best_eval_accuracy.json")
    value = numeric(metadata.get("accuracy"))
    if value is not None and int(metadata.get("step") or 0) > 0:
        return value
    values = [value for value in (row_accuracy(row) for row in epoch_rows) if value is not None]
    return max(values) if values else None


def compact_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": row.get("step"),
        "epoch_index": row.get("epoch_index"),
        "accuracy": row.get("accuracy"),
        "effective_rank": row.get("effective_rank"),
        "lsp_to_ce_ratio": row.get("lsp_to_ce_ratio"),
        "latent_variance_mean": row.get("latent_variance_mean"),
    }


def evaluate_forgetting(
    *,
    baseline_accuracy: float | None,
    eval_rows: list[dict[str, Any]],
    best_accuracy: float | None,
    generation: int,
    thresholds: ForgettingThresholds = ForgettingThresholds(),
) -> dict[str, Any]:
    epoch_rows = epoch_eval_rows(eval_rows)
    warnings: list[dict[str, Any]] = []
    if baseline_accuracy is None:
        return {
            "status": "missing_baseline",
            "action": "none",
            "baseline_accuracy": None,
            "completed_epoch_evals": len(epoch_rows),
            "warnings": warnings,
        }

    for row in epoch_rows:
        accuracy = row_accuracy(row)
        if accuracy is not None and accuracy <= baseline_accuracy - thresholds.warn_accuracy_drop:
            warnings.append(
                {
                    "kind": "accuracy_drop",
                    "step": row.get("step"),
                    "epoch_index": row.get("epoch_index"),
                    "accuracy": accuracy,
                    "drop": baseline_accuracy - accuracy,
                    "threshold": thresholds.warn_accuracy_drop,
                }
            )
        ratio = numeric(row.get("lsp_to_ce_ratio"))
        if ratio is not None and ratio > thresholds.warn_lsp_to_ce_ratio:
            warnings.append(
                {
                    "kind": "lsp_to_ce_ratio_high",
                    "step": row.get("step"),
                    "epoch_index": row.get("epoch_index"),
                    "lsp_to_ce_ratio": ratio,
                    "threshold": thresholds.warn_lsp_to_ce_ratio,
                }
            )

    status = "warning" if warnings else "ok"
    action = "none"
    reason = None
    last_three = epoch_rows[-3:]
    last_three_accuracies = [row_accuracy(row) for row in last_three]
    restart_threshold = baseline_accuracy - thresholds.restart_accuracy_drop
    best_threshold = baseline_accuracy - thresholds.restart_best_drop
    sustained_low = (
        len(epoch_rows) >= thresholds.min_completed_epochs
        and len(last_three_accuracies) == 3
        and all(value is not None and value < restart_threshold for value in last_three_accuracies)
    )
    best_still_low = best_accuracy is not None and best_accuracy < best_threshold
    no_recovery = (
        len(last_three_accuracies) == 3
        and last_three_accuracies[-1] is not None
        and last_three_accuracies[0] is not None
        and last_three_accuracies[-1] <= last_three_accuracies[0] + thresholds.recovery_tolerance
    )
    should_act = sustained_low and best_still_low and no_recovery
    if should_act:
        action = "stop_only" if generation >= thresholds.max_auto_restarts else "restart"
        status = "restart_required" if action == "restart" else "stop_required"
        reason = {
            "sustained_low": sustained_low,
            "best_still_low": best_still_low,
            "no_recovery": no_recovery,
            "restart_threshold": restart_threshold,
            "best_threshold": best_threshold,
        }

    return {
        "status": status,
        "action": action,
        "baseline_accuracy": baseline_accuracy,
        "completed_epoch_evals": len(epoch_rows),
        "best_eval_accuracy": best_accuracy,
        "latest_eval": compact_eval_row(epoch_rows[-1]) if epoch_rows else None,
        "last_three_evals": [compact_eval_row(row) for row in last_three],
        "last_three_accuracies": last_three_accuracies,
        "warnings": warnings,
        "reason": reason,
        "thresholds": thresholds.__dict__,
        "generation": generation,
    }


def pid_alive(pid: int | None) -> bool:
    return pid is not None and pid > 0 and Path(f"/proc/{pid}").exists()


def direct_children(pid: int) -> list[int]:
    children: list[int] = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            child_pid = int(proc.name)
            status = (proc / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        except ValueError:
            continue
        ppid = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                try:
                    ppid = int(line.split()[1])
                except (IndexError, ValueError):
                    ppid = None
                break
        if ppid == pid:
            children.append(child_pid)
    return children


def process_tree(pid: int) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []

    def visit(current: int) -> None:
        if current in seen:
            return
        seen.add(current)
        ordered.append(current)
        for child in direct_children(current):
            visit(child)

    visit(pid)
    return ordered


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def stop_pid(pid: int | None, *, timeout_sec: float = 30.0) -> bool:
    if not pid_alive(pid):
        return False
    assert pid is not None
    targets = list(reversed(process_tree(pid)))
    try:
        for target in targets:
            if pid_alive(target):
                os.kill(target, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not any(pid_alive(target) for target in targets):
            return True
        time.sleep(0.5)
    for target in targets:
        if not pid_alive(target):
            continue
        try:
            os.kill(target, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return not any(pid_alive(target) for target in targets)


def stop_run_processes(run_dir: Path) -> dict[str, Any]:
    stopped: dict[str, Any] = {}
    for name in ("train", "checkpoint_alias_watcher"):
        pid = read_pid(run_dir / f"{name}.pid")
        stopped[name] = {"pid": pid, "stopped": stop_pid(pid)}
    return stopped


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is present in training env
        raise RuntimeError("PyYAML is required to create safer restart configs") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is present in training env
        raise RuntimeError("PyYAML is required to create safer restart configs") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def ensure_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        value = {}
        payload[key] = value
    return value


def create_safer_config(
    *,
    config_path: Path,
    current_run_dir: Path,
    init_checkpoint: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    payload = load_yaml(config_path)
    stamp = run_timestamp()
    run_name = f"steptraj_from_coconut_full_latent_aw001_ce10_warmup50_lr1e5_{stamp}"
    new_run_dir = current_run_dir.parent / run_name
    new_config_path = config_path.with_name(
        f"lsp_step_trajectory_from_coconut_full_latent_aw001_ce10_warmup50_lr1e5_{stamp}.yaml"
    )

    experiment = ensure_mapping(payload, "experiment")
    initialization = ensure_mapping(payload, "initialization")
    training = ensure_mapping(payload, "training")
    loss = ensure_mapping(payload, "loss")
    host_losses = ensure_mapping(payload, "host_losses")
    eval_config = ensure_mapping(payload, "eval")

    experiment["run_kind"] = "lsp_core_step_trajectory_from_coconut_full_latent_aw001_ce10_warmup50_lr1e5"
    initialization["coconut_checkpoint"] = str(init_checkpoint)
    training["output_dir"] = str(new_run_dir)
    training["log_dir"] = str(new_run_dir / "logs")
    training["lr"] = 1.0e-5
    loss["align_weight"] = 0.01
    loss["lsp_weight"] = 0.01
    loss["lsp_weight_warmup_ratio"] = 0.50
    loss["lsp_weight_warmup_steps"] = 0
    loss["anti_collapse_weight"] = 0.01
    host_losses["host_answer_ce_weight"] = 1.0
    eval_config["before_train"] = True
    eval_config["every_epoch"] = True

    dump_yaml(new_config_path, payload)
    return new_config_path, new_run_dir, payload


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_resume_monitor(
    *,
    run_dir: Path,
    config_path: Path,
    repo: Path,
    python: Path,
    cuda_devices: str,
    nproc: int,
    generation: int,
) -> Path:
    script_path = run_dir / "resume_monitor.sh"
    text = f"""#!/usr/bin/env bash
set -uo pipefail

RUN_DIR={shell_quote(str(run_dir))}
CFG={shell_quote(str(config_path))}
REPO={shell_quote(str(repo))}
PYTHON={shell_quote(str(python))}
PIDFILE="$RUN_DIR/train.pid"
MONITOR_LOG="$RUN_DIR/resume_monitor.log"
FORGETTING_MONITOR="$REPO/lsp_jepa/scripts/forgetting_monitor.py"
INTERVAL_SECONDS=120
NPROC={int(nproc)}
CUDA_DEVICES={shell_quote(cuda_devices)}
FORGETTING_GENERATION={int(generation)}

log() {{
  printf '[%s] %s\\n' "$(date '+%F %T %z')" "$*" >> "$MONITOR_LOG"
}}

pid_alive() {{
  local pid="$1"
  [ -n "$pid" ] && [ -d "/proc/$pid" ]
}}

train_complete() {{
  "$PYTHON" - "$RUN_DIR/metrics.jsonl" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists() or p.stat().st_size == 0:
    raise SystemExit(1)
last = None
for line in p.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    try:
        last = json.loads(line)
    except json.JSONDecodeError:
        pass
if not last:
    raise SystemExit(1)
step = int(last.get('step') or 0)
max_steps = int(last.get('max_steps') or 0)
raise SystemExit(0 if max_steps > 0 and step >= max_steps else 1)
PY
}}

current_train_pid() {{
  if [ -f "$PIDFILE" ]; then
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if pid_alive "$pid"; then
      echo "$pid"
      return 0
    fi
  fi
  pgrep -f "train_lsp_jepa_core.py .*--config $CFG" | sed -n '1p' || true
}}

start_resume() {{
  local latest="$RUN_DIR/checkpoints/latest.pt"
  if [ ! -f "$latest" ]; then
    log "cannot resume: latest checkpoint missing at $latest"
    return 1
  fi
  local ts log_path pid
  ts="$(date '+%Y%m%dT%H%M%S')"
  log_path="$RUN_DIR/logs/resume_${{ts}}_from_latest.log"
  mkdir -p "$RUN_DIR/logs"
  (
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    export PYTHONPATH="$REPO"
    cd "$REPO"
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \\
      lsp_jepa/scripts/train_lsp_jepa_core.py \\
      --config "$CFG" \\
      --resume-from-checkpoint "$latest" \\
      --append-metrics
  ) > "$log_path" 2>&1 &
  pid="$!"
  echo "$pid" > "$PIDFILE"
  log "resumed training pid=$pid from $latest log=$log_path"
}}

log "resume monitor started interval=${{INTERVAL_SECONDS}}s generation=${{FORGETTING_GENERATION}}"
while true; do
  if [ -f "$RUN_DIR/forgetting_stop_reason.json" ]; then
    log "forgetting stop reason exists; monitor exiting"
    exit 0
  fi
  "$PYTHON" "$FORGETTING_MONITOR" \\
    --run-dir "$RUN_DIR" \\
    --config "$CFG" \\
    --repo "$REPO" \\
    --python "$PYTHON" \\
    --cuda-devices "$CUDA_DEVICES" \\
    --nproc "$NPROC" \\
    --generation "$FORGETTING_GENERATION" \\
    --auto-restart >> "$RUN_DIR/forgetting_monitor.log" 2>&1
  rc="$?"
  if [ "$rc" -eq {ACTION_EXIT_CODE} ]; then
    log "forgetting monitor requested stop or restart; monitor exiting"
    exit 0
  fi
  if train_complete; then
    log "training complete; monitor exiting"
    exit 0
  fi
  pid="$(current_train_pid)"
  if [ -n "$pid" ] && pid_alive "$pid"; then
    echo "$pid" > "$PIDFILE"
  else
    log "training process missing; attempting resume from latest.pt"
    start_resume
  fi
  sleep "$INTERVAL_SECONDS"
done
"""
    run_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(text, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def launch_training(
    *,
    run_dir: Path,
    config_path: Path,
    repo: Path,
    python: Path,
    cuda_devices: str,
    nproc: int,
) -> dict[str, Any]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = logs_dir / f"run_{timestamp}_from_coconut.log"
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": cuda_devices,
            "PYTHONPATH": str(repo),
        }
    )
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "lsp_jepa/scripts/train_lsp_jepa_core.py",
        "--config",
        str(config_path),
    ]
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    log_handle.close()
    (run_dir / "train.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    (run_dir / "launch_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    return {"pid": process.pid, "log_path": str(log_path), "command": command}


def start_resume_monitor(
    *,
    run_dir: Path,
    config_path: Path,
    repo: Path,
    python: Path,
    cuda_devices: str,
    nproc: int,
    generation: int,
) -> dict[str, Any]:
    script_path = write_resume_monitor(
        run_dir=run_dir,
        config_path=config_path,
        repo=repo,
        python=python,
        cuda_devices=cuda_devices,
        nproc=nproc,
        generation=generation,
    )
    log_path = run_dir / "resume_monitor.nohup.log"
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", str(script_path)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    log_handle.close()
    (run_dir / "resume_monitor.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return {"pid": process.pid, "script_path": str(script_path), "log_path": str(log_path)}


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.run_dir / "evals" / "metrics.jsonl")
    baseline = find_step0_baseline(args.run_dir, rows)
    epoch_rows = epoch_eval_rows(rows)
    baseline_accuracy = row_accuracy(baseline) if baseline is not None else None
    best_accuracy = best_eval_accuracy(args.run_dir, epoch_rows)
    decision = evaluate_forgetting(
        baseline_accuracy=baseline_accuracy,
        eval_rows=rows,
        best_accuracy=best_accuracy,
        generation=args.generation,
        thresholds=ForgettingThresholds(
            warn_accuracy_drop=args.warn_accuracy_drop,
            warn_lsp_to_ce_ratio=args.warn_lsp_to_ce_ratio,
            min_completed_epochs=args.min_completed_epochs,
            restart_accuracy_drop=args.restart_accuracy_drop,
            restart_best_drop=args.restart_best_drop,
            recovery_tolerance=args.recovery_tolerance,
            max_auto_restarts=args.max_auto_restarts,
        ),
    )
    return {
        "timestamp_utc": utc_timestamp(),
        "run_dir": str(args.run_dir),
        "config": str(args.config),
        "baseline": compact_eval_row(baseline) if baseline is not None else None,
        "decision": decision,
    }


def write_stop_reason(
    *,
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    process_stop: dict[str, Any],
    action: str,
    new_run_dir: Path | None = None,
    new_config_path: Path | None = None,
    launch: dict[str, Any] | None = None,
    monitor: dict[str, Any] | None = None,
) -> None:
    payload = {
        "timestamp_utc": utc_timestamp(),
        "action": action,
        "baseline": snapshot.get("baseline"),
        "decision": snapshot.get("decision"),
        "process_stop": process_stop,
        "old_run_dir": str(args.run_dir),
        "new_run_dir": str(new_run_dir) if new_run_dir is not None else None,
        "new_config_path": str(new_config_path) if new_config_path is not None else None,
        "launch": launch,
        "resume_monitor": monitor,
    }
    write_json(args.run_dir / "forgetting_stop_reason.json", payload)


def maybe_take_action(args: argparse.Namespace, snapshot: dict[str, Any]) -> int:
    decision = snapshot["decision"]
    action = decision.get("action")
    if not args.auto_restart or action == "none":
        return 0
    if (args.run_dir / "forgetting_stop_reason.json").exists():
        return ACTION_EXIT_CODE

    process_stop = stop_run_processes(args.run_dir)
    if action == "stop_only":
        write_stop_reason(
            args=args,
            snapshot=snapshot,
            process_stop=process_stop,
            action="stop_only",
        )
        return ACTION_EXIT_CODE

    init_checkpoint = args.safer_init_checkpoint
    if init_checkpoint is None:
        init_checkpoint = DEFAULT_INIT_COCONUT_CHECKPOINT
    new_config_path, new_run_dir, _payload = create_safer_config(
        config_path=args.config,
        current_run_dir=args.run_dir,
        init_checkpoint=init_checkpoint,
    )
    launch = launch_training(
        run_dir=new_run_dir,
        config_path=new_config_path,
        repo=args.repo,
        python=args.python,
        cuda_devices=args.cuda_devices,
        nproc=args.nproc,
    )
    monitor = start_resume_monitor(
        run_dir=new_run_dir,
        config_path=new_config_path,
        repo=args.repo,
        python=args.python,
        cuda_devices=args.cuda_devices,
        nproc=args.nproc,
        generation=args.generation + 1,
    )
    write_stop_reason(
        args=args,
        snapshot=snapshot,
        process_stop=process_stop,
        action="restart",
        new_run_dir=new_run_dir,
        new_config_path=new_config_path,
        launch=launch,
        monitor=monitor,
    )
    return ACTION_EXIT_CODE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/root/SIM-CoT"))
    parser.add_argument("--python", type=Path, default=Path("/root/miniconda3/bin/python"))
    parser.add_argument("--cuda-devices", default="0,1,2,3,4")
    parser.add_argument("--nproc", type=int, default=5)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--max-auto-restarts", type=int, default=1)
    parser.add_argument("--warn-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--warn-lsp-to-ce-ratio", type=float, default=1.5)
    parser.add_argument("--min-completed-epochs", type=int, default=5)
    parser.add_argument("--restart-accuracy-drop", type=float, default=0.06)
    parser.add_argument("--restart-best-drop", type=float, default=0.04)
    parser.add_argument("--recovery-tolerance", type=float, default=0.01)
    parser.add_argument("--safer-init-checkpoint", type=Path, default=DEFAULT_INIT_COCONUT_CHECKPOINT)
    parser.add_argument("--auto-restart", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_snapshot(args)
    write_json(args.run_dir / "forgetting_monitor_status.json", snapshot)
    print(json.dumps(snapshot, sort_keys=True), flush=True)
    return maybe_take_action(args, snapshot)


if __name__ == "__main__":
    sys.exit(main())
