#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="lsp_jepa/configs/core/lsp_seq_full_train.yaml"
READINESS_REPORT="lsp_jepa/runs/pr10_mvp/full_data_readiness_report.json"
OUTPUT_ROOT="/root/autodl-tmp/lsp_jepa/runs/pr13_mvp_full_seq_launch"
RUN_NAME=""
MAX_STEPS=""
DRY_RUN=0
DEVICE=""
MODEL_ID=""
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  lsp_jepa/scripts/launch_full_seq_train.sh --dry-run [options] [-- extra train args]
  lsp_jepa/scripts/launch_full_seq_train.sh --max-steps N [options] [-- extra train args]

PR13 launch gate for full-data sequence-level LSP-JEPA-Core training.

Options:
  --config PATH             Config path. Default: lsp_jepa/configs/core/lsp_seq_full_train.yaml
  --readiness-report PATH   Full-data readiness report. Default: lsp_jepa/runs/pr10_mvp/full_data_readiness_report.json
  --output-root PATH        Output root. Default: /root/autodl-tmp/lsp_jepa/runs/pr13_mvp_full_seq_launch
  --run-name NAME           Output subdirectory name.
  --dry-run                 Run a 1-step smoke check with 32 samples.
  --max-steps N             Run N optimizer steps on the configured full training split.
  --device DEVICE           Forward to train_lsp_jepa_core.py.
  --model-id ID             Forward to train_lsp_jepa_core.py.
  -h, --help                Show this help.

Examples:
  lsp_jepa/scripts/launch_full_seq_train.sh --dry-run
  lsp_jepa/scripts/launch_full_seq_train.sh --max-steps 10
  lsp_jepa/scripts/launch_full_seq_train.sh --max-steps 100
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --readiness-report)
      READINESS_REPORT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$REPO_ROOT"

if [[ "$DRY_RUN" == "1" && -n "$MAX_STEPS" ]]; then
  echo "--dry-run and --max-steps are mutually exclusive" >&2
  exit 2
fi
if [[ "$DRY_RUN" == "0" && -z "$MAX_STEPS" ]]; then
  echo "Provide --dry-run or --max-steps N" >&2
  exit 2
fi
if [[ "$DRY_RUN" == "1" ]]; then
  MAX_STEPS=1
  RUN_NAME="${RUN_NAME:-dry_run}"
else
  RUN_NAME="${RUN_NAME:-max_steps_${MAX_STEPS}}"
fi

OUTPUT_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}"
LOG_DIR="${OUTPUT_DIR%/}/logs"
CHECKPOINT_DIR="${OUTPUT_DIR%/}/checkpoints"
METRICS_PATH="${OUTPUT_DIR%/}/metrics.jsonl"
SUMMARY_PATH="${OUTPUT_DIR%/}/summary.json"
CONFIG_SNAPSHOT_PATH="${OUTPUT_DIR%/}/config.yaml"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
rm -f "$METRICS_PATH" "$SUMMARY_PATH" "$CONFIG_SNAPSHOT_PATH"

python - "$CONFIG_PATH" "$READINESS_REPORT" <<'PY'
from pathlib import Path
import json
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required; install lsp_jepa/requirements-mvp.txt") from exc

config_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text()) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"config must be a mapping: {config_path}")

def get(dotted, default=None):
    value = cfg
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value

required = {
    "experiment.mode": "core",
    "lsp_objective.objective": "lsp_state",
    "mapping.strategy": "sequence",
    "host_losses.simcot_decoder_weight": 0.0,
    "host_losses.codi_distill_weight": 0.0,
    "host_losses.intermediate_cot_ce_weight": 0.0,
    "teacher.exclude_answer_tokens": True,
    "teacher.exclude_answer_prefix": True,
    "teacher.target_position": "final_valid_reasoning_step",
}
for key, expected in required.items():
    actual = get(key)
    if actual != expected:
        raise SystemExit(f"config gate failed: {key}={actual!r}, expected {expected!r}")

if get("student.detach_between_steps") is not False:
    raise SystemExit("config gate failed: student.detach_between_steps must be false")
if get("optimization_contract.ema_update_after_optimizer_step") is not True:
    raise SystemExit("config gate failed: EMA update must be after optimizer.step")
if not report_path.exists():
    raise SystemExit(f"readiness report missing: {report_path}")
report = json.loads(report_path.read_text())
checks = {
    "empty_teacher_target_rate": report.get("teacher_target", {}).get("empty_teacher_target_rate"),
    "empty_latent_batch_rate": report.get("latent", {}).get("latent_empty_batch_rate"),
    "answer_leakage_count": report.get("answer_leakage", {}).get("answer_leakage_count"),
    "skipped_sample_count": report.get("skips", {}).get("skipped_sample_count"),
}
expected = {
    "empty_teacher_target_rate": 0.0,
    "empty_latent_batch_rate": 0.0,
    "answer_leakage_count": 0,
    "skipped_sample_count": 0,
}
for key, expected_value in expected.items():
    if checks[key] != expected_value:
        raise SystemExit(f"readiness gate failed: {key}={checks[key]!r}, expected {expected_value!r}")
if not all(report.get("acceptance", {}).values()):
    raise SystemExit(f"readiness acceptance has failures: {report.get('acceptance', {})}")
print(json.dumps({"config_gate": "passed", "readiness_gate": "passed", **checks}, sort_keys=True))
PY

RUN_LOG="$LOG_DIR/run_$(date -u +%Y%m%dT%H%M%SZ)_max_steps_${MAX_STEPS}.log"
CMD=(
  python lsp_jepa/scripts/train_lsp_jepa_core.py
  --config "$CONFIG_PATH"
  --max-steps "$MAX_STEPS"
  --output-dir "$OUTPUT_DIR"
  --metrics-path "$METRICS_PATH"
  --summary-path "$SUMMARY_PATH"
  --config-snapshot-path "$CONFIG_SNAPSHOT_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --save-checkpoints
)
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--num-samples 32 --min-samples 1)
fi
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
if [[ -n "$MODEL_ID" ]]; then
  CMD+=(--model-id "$MODEL_ID")
fi
CMD+=("${EXTRA_ARGS[@]}")

{
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
} 2>&1 | while IFS= read -r line; do
  printf '%s\n' "$line" | tee -a "$RUN_LOG"
  if [[ "$line" == \{*\"step\"*\"total_loss\"* ]]; then
    python - "$line" <<'PY' | tee -a "$RUN_LOG"
import json
import sys

try:
    row = json.loads(sys.argv[1])
except json.JSONDecodeError:
    raise SystemExit(0)
required = {"step", "max_steps", "total_loss", "lsp_loss", "host_answer_ce"}
if not required <= row.keys():
    raise SystemExit(0)
step = int(row["step"])
max_steps = int(row["max_steps"])
width = 28
filled = min(width, max(0, round(width * step / max_steps)))
bar = "#" * filled + "." * (width - filled)
print(
    f"progress [{bar}] {step}/{max_steps} "
    f"total_loss={float(row['total_loss']):.6f} "
    f"lsp_loss={float(row['lsp_loss']):.6f} "
    f"host_answer_ce={float(row['host_answer_ce']):.6f} "
    f"latent_var={float(row['latent_variance_mean']):.6g} "
    f"ema_drift={float(row['ema_drift_l1']):.6g}"
)
PY
  fi
done

python - "$SUMMARY_PATH" "$METRICS_PATH" "$RUN_LOG" "$DRY_RUN" <<'PY'
from pathlib import Path
import json
import math
import sys

summary_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
run_log = Path(sys.argv[3])
dry_run = sys.argv[4] == "1"
summary = json.loads(summary_path.read_text())
metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
if not metrics:
    raise SystemExit("PR13 gate failed: no metrics were written")

def finite_row(row):
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if isinstance(value, int) and not math.isfinite(float(value)):
            return False
    return True

tail_count = max(1, len(metrics) // 4)
tail = metrics[-tail_count:]
pairwise_tail_mean = sum(float(row["pairwise_cosine_mean"]) for row in tail) / len(tail)
threshold = float(metrics[-1]["max_steps"] and metrics[-1].get("pairwise_cosine_near_one") is not None and 0.999)
gate = {
    "mode_core": all(row.get("experiment_mode") == "core" for row in metrics),
    "objective_lsp_state": all(row.get("objective") == "lsp_state" for row in metrics),
    "mapping_sequence": all(row.get("mapping") == "sequence" for row in metrics),
    "target_final_valid_reasoning_step": all(row.get("target_position") == "final_valid_reasoning_step" for row in metrics),
    "completed_steps": len(metrics) == int(metrics[-1]["max_steps"]),
    "no_nan_or_inf": all(finite_row(row) for row in metrics),
    "teacher_no_grad": all(int(row["teacher_grad_params"]) == 0 for row in metrics),
    "student_has_grad": all(float(row["student_grad_l1"]) > 0.0 and float(row["student_base_grad_l1"]) > 0.0 for row in metrics),
    "ema_drift_nonzero": all(float(row["ema_drift_l1"]) > 0.0 for row in metrics),
    "latent_variance_nonzero": all(float(row["latent_variance_mean"]) > 1e-10 for row in metrics),
    "pairwise_cosine_not_long_near_one": pairwise_tail_mean < threshold and not any(row.get("pairwise_cosine_all_one") for row in metrics),
    "host_answer_ce_not_double_counted": all(bool(row["answer_ce_double_count_ok"]) and row["answer_ce_terms_in_total"] in ([], ["host_answer_ce"]) for row in metrics),
    "teacher_targets_nonempty": all(bool(row["teacher_target_mask_nonempty"]) for row in metrics),
    "student_latents_nonempty": all(bool(row["latent_mask_nonempty"]) for row in metrics),
    "answer_leakage_ok": all(bool(row["answer_leakage_ok"]) for row in metrics),
}
gate["passed"] = all(gate.values())
summary["pr13_mvp"] = {
    "dry_run": dry_run,
    "run_log": str(run_log),
    "metrics_path": str(metrics_path),
    "summary_path": str(summary_path),
    "steps_completed": len(metrics),
    "max_steps": int(metrics[-1]["max_steps"]),
    "final_total_loss": metrics[-1]["total_loss"],
    "final_lsp_loss": metrics[-1]["lsp_loss"],
    "host_answer_ce_weight": metrics[-1]["host_answer_ce_weight"],
    "answer_ce_terms_in_total": metrics[-1]["answer_ce_terms_in_total"],
    "teacher_grad_params_last": metrics[-1]["teacher_grad_params"],
    "student_grad_l1_last": metrics[-1]["student_grad_l1"],
    "ema_drift_l1_last": metrics[-1]["ema_drift_l1"],
    "latent_variance_mean_last": metrics[-1]["latent_variance_mean"],
    "pairwise_cosine_tail_mean": pairwise_tail_mean,
    "gate": gate,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["pr13_mvp"], sort_keys=True))
if not gate["passed"]:
    raise SystemExit("PR13 gate failed")
PY
