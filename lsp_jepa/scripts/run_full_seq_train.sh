#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="lsp_jepa/configs/core/lsp_seq_full_train.yaml"
OUTPUT_DIR="outputs/lsp_jepa/full_seq"
MAX_STEPS=""
ONE_EPOCH=0
DRY_RUN=0
RESUME=0
NUM_SAMPLES=""
MIN_SAMPLES=""
MODEL_ID=""
DEVICE=""
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  lsp_jepa/scripts/run_full_seq_train.sh [options] [-- extra train args]

Options:
  --config PATH          Config path. Default: lsp_jepa/configs/core/lsp_seq_full_train.yaml
  --output-dir PATH      Output directory. Default: outputs/lsp_jepa/full_seq
  --max-steps N          Train for N optimizer steps.
  --one-epoch            Train one full configured epoch. Default when --max-steps is omitted.
  --dry-run              Run a 1-step smoke check in outputs/lsp_jepa/full_seq/dry_run.
  --resume               Keep existing metrics and continue writing into the same output dir.
  --num-samples N        Forward to train_lsp_jepa_core.py.
  --min-samples N        Forward to train_lsp_jepa_core.py.
  --model-id ID          Forward to train_lsp_jepa_core.py.
  --device DEVICE        Forward to train_lsp_jepa_core.py.
  -h, --help             Show this help.

Examples:
  lsp_jepa/scripts/run_full_seq_train.sh --dry-run
  lsp_jepa/scripts/run_full_seq_train.sh --max-steps 100
  lsp_jepa/scripts/run_full_seq_train.sh --one-epoch
  lsp_jepa/scripts/run_full_seq_train.sh --max-steps 100 --resume
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --one-epoch)
      ONE_EPOCH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --min-samples)
      MIN_SAMPLES="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
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

if [[ "$DRY_RUN" == "1" ]]; then
  MAX_STEPS="${MAX_STEPS:-1}"
  OUTPUT_DIR="${OUTPUT_DIR%/}/dry_run"
  NUM_SAMPLES="${NUM_SAMPLES:-32}"
  MIN_SAMPLES="${MIN_SAMPLES:-1}"
elif [[ -z "$MAX_STEPS" || "$ONE_EPOCH" == "1" ]]; then
  MAX_STEPS="$(python - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required; install lsp_jepa/requirements-mvp.txt") from exc
cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
training = cfg.get("training", {})
print(int(training.get("one_epoch_steps") or training.get("max_steps") or 1))
PY
)"
fi

LOG_DIR="${OUTPUT_DIR%/}/logs"
CHECKPOINT_DIR="${OUTPUT_DIR%/}/checkpoints"
METRICS_PATH="${OUTPUT_DIR%/}/metrics.jsonl"
SUMMARY_PATH="${OUTPUT_DIR%/}/summary.json"
CONFIG_SNAPSHOT_PATH="${OUTPUT_DIR%/}/config.yaml"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

if [[ "$RESUME" == "1" ]]; then
  latest_checkpoint="$(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'step_*.pt' -type f | sort | tail -n 1 || true)"
  if [[ -z "$latest_checkpoint" ]]; then
    echo "resume requested but no checkpoint found under $CHECKPOINT_DIR" >&2
    exit 2
  fi
  echo "resume requested; preserving prior metrics and using latest checkpoint marker: $latest_checkpoint"
else
  rm -f "$METRICS_PATH" "$SUMMARY_PATH" "$CONFIG_SNAPSHOT_PATH"
fi

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

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--append-metrics)
fi
if [[ -n "$NUM_SAMPLES" ]]; then
  CMD+=(--num-samples "$NUM_SAMPLES")
fi
if [[ -n "$MIN_SAMPLES" ]]; then
  CMD+=(--min-samples "$MIN_SAMPLES")
fi
if [[ -n "$MODEL_ID" ]]; then
  CMD+=(--model-id "$MODEL_ID")
fi
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi
CMD+=("${EXTRA_ARGS[@]}")

{
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}"
} 2>&1 | tee "$RUN_LOG"

python - "$SUMMARY_PATH" "$METRICS_PATH" "$RUN_LOG" "$RESUME" <<'PY'
from pathlib import Path
import json
import math
import sys

summary_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
run_log = Path(sys.argv[3])
resume_requested = sys.argv[4] == "1"

summary = json.loads(summary_path.read_text())
metrics = [
    json.loads(line)
    for line in metrics_path.read_text().splitlines()
    if line.strip()
]
last = metrics[-1] if metrics else {}
has_nan = any(
    isinstance(value, float) and math.isnan(value)
    for row in metrics
    for value in row.values()
)
effective_samples = len({
    sample_id
    for row in metrics
    for sample_id in row.get("batch_sample_ids", [])
})
target_empty = sum(
    1
    for row in metrics
    if not row.get("teacher_target_mask_nonempty", False)
)
skip_count = 0
final_loss = last.get("total_loss")
final_lsp_loss = last.get("lsp_loss")
finite_final_loss = isinstance(final_loss, (int, float)) and math.isfinite(float(final_loss))

summary["pr12_mvp"] = {
    "final_loss": final_loss,
    "final_lsp_loss": final_lsp_loss,
    "effective_sample_count": effective_samples,
    "skip_sample_count": skip_count,
    "target_empty_sample_count": target_empty,
    "has_nan": has_nan,
    "acceptance_passed": bool(
        metrics
        and finite_final_loss
        and not has_nan
        and target_empty == 0
        and all(row.get("answer_leakage_ok", False) for row in metrics)
        and all(row.get("teacher_target_mask_nonempty", False) for row in metrics)
        and all(row.get("latent_mask_nonempty", False) for row in metrics)
    ),
    "resume_requested": resume_requested,
    "run_log": str(run_log),
    "outputs": {
        "config": str(summary_path.with_name("config.yaml")),
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "checkpoints": str(summary_path.with_name("checkpoints")),
        "logs": str(summary_path.with_name("logs")),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["pr12_mvp"], sort_keys=True))
PY
