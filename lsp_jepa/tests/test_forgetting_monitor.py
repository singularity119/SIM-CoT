from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "forgetting_monitor.py"
SPEC = importlib.util.spec_from_file_location("forgetting_monitor_for_tests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
forgetting_monitor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = forgetting_monitor
SPEC.loader.exec_module(forgetting_monitor)

evaluate_forgetting = forgetting_monitor.evaluate_forgetting


def eval_row(epoch: int, accuracy: float, ratio: float = 0.5) -> dict:
    return {
        "step": epoch * 100,
        "epoch_index": epoch,
        "accuracy": accuracy,
        "lsp_to_ce_ratio": ratio,
        "effective_rank": 20.0 + epoch,
        "latent_variance_mean": 0.7,
    }


def test_epoch_before_five_warns_without_restart():
    decision = evaluate_forgetting(
        baseline_accuracy=0.35,
        eval_rows=[eval_row(1, 0.28), eval_row(2, 0.27), eval_row(3, 0.26)],
        best_accuracy=0.28,
        generation=0,
    )

    assert decision["status"] == "warning"
    assert decision["action"] == "none"
    assert decision["completed_epoch_evals"] == 3
    assert any(item["kind"] == "accuracy_drop" for item in decision["warnings"])


def test_single_large_drop_only_warns():
    decision = evaluate_forgetting(
        baseline_accuracy=0.35,
        eval_rows=[
            eval_row(1, 0.34),
            eval_row(2, 0.33),
            eval_row(3, 0.32),
            eval_row(4, 0.31),
            eval_row(5, 0.28),
        ],
        best_accuracy=0.34,
        generation=0,
    )

    assert decision["status"] == "warning"
    assert decision["action"] == "none"


def test_sustained_last_three_low_restarts_after_five_epochs():
    decision = evaluate_forgetting(
        baseline_accuracy=0.35,
        eval_rows=[
            eval_row(1, 0.34),
            eval_row(2, 0.33),
            eval_row(3, 0.28),
            eval_row(4, 0.27),
            eval_row(5, 0.27),
        ],
        best_accuracy=0.30,
        generation=0,
    )

    assert decision["status"] == "restart_required"
    assert decision["action"] == "restart"


def test_recovery_trend_prevents_restart():
    decision = evaluate_forgetting(
        baseline_accuracy=0.35,
        eval_rows=[
            eval_row(1, 0.34),
            eval_row(2, 0.33),
            eval_row(3, 0.25),
            eval_row(4, 0.26),
            eval_row(5, 0.285),
        ],
        best_accuracy=0.30,
        generation=0,
    )

    assert decision["status"] == "warning"
    assert decision["action"] == "none"


def test_generation_one_stops_instead_of_looping_restart():
    decision = evaluate_forgetting(
        baseline_accuracy=0.35,
        eval_rows=[
            eval_row(1, 0.34),
            eval_row(2, 0.33),
            eval_row(3, 0.28),
            eval_row(4, 0.27),
            eval_row(5, 0.27),
        ],
        best_accuracy=0.30,
        generation=1,
    )

    assert decision["status"] == "stop_required"
    assert decision["action"] == "stop_only"
