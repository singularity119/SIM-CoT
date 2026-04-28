"""PR14 minimal final-answer evaluation for LSP-JEPA-Core checkpoints.

The script intentionally evaluates only generated final answers. It does not
build teacher targets, does not generate or score CoT, does not run step-level
metrics, and does not update EMA or optimizer state.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
COCONUT_DIR = REPO_ROOT / "Coconut"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COCONUT_DIR) not in sys.path:
    sys.path.insert(0, str(COCONUT_DIR))

from lsp_jepa.scripts.train_lsp_jepa_core import (  # noqa: E402
    Coconut,
    LatentPredictor,
    MinimalTokenizer,
    TinyCausalLM,
    build_tokenizer_and_model,
    clean_answer,
    config_get,
    load_hf_dataset_samples,
    load_samples_from_path,
    load_yaml_config,
    model_hidden_size,
    resolve_device,
    resolve_repo_path,
)


DEFAULT_CHECKPOINT = Path(
    "/root/autodl-tmp/lsp_jepa/runs/pr13_mvp_full_seq_launch/"
    "max_steps_100/checkpoints/step_000100.pt"
)
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/lsp_jepa/runs/pr14_mvp_min_eval")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    config_path = resolve_repo_path(args.config)
    checkpoint_path = resolve_repo_path(args.checkpoint)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve_repo_path(args.metrics_path) if args.metrics_path else output_dir / "metrics.json"

    samples = load_eval_samples(args)
    checkpoint = load_checkpoint(checkpoint_path)
    tokenizer, base_model = build_eval_tokenizer_and_model(samples, args)
    student, predictor, load_status = build_and_load_student(
        checkpoint,
        tokenizer=tokenizer,
        base_model=base_model,
        model_id=args.model_id,
        num_latent_steps=args.num_latent_steps,
        predictor_head_layers=args.predictor_head_layers,
        use_predictor_head=args.use_predictor_head,
        device=device,
    )
    load_status["tokenizer"] = {
        "loaded_from_checkpoint": False,
        "checkpoint_contains_tokenizer": isinstance(checkpoint, Mapping) and "tokenizer" in checkpoint,
        "reconstructed_from_eval_samples": args.model_id == "tiny",
        "loaded_from_model_id": args.model_id != "tiny",
        "eval_tokenizer_vocab_size": len(tokenizer),
    }
    student.eval()
    predictor.eval()

    rows = []
    exact_matches = 0
    numeric_matches = 0
    invalid_answers = 0
    generated_lengths = []
    with torch.no_grad():
        for index, sample in enumerate(samples):
            input_ids = build_eval_input_ids(
                tokenizer,
                sample.question,
                num_latent_steps=args.num_latent_steps,
                device=device,
            )
            outputs = student.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, device=device),
                max_new_tokens=args.max_new_tokens,
                synced_gpus=False,
            )
            generated_ids = outputs[0, input_ids.shape[1] :].detach().cpu().tolist()
            generated_ids = trim_after_eos(generated_ids, int(tokenizer.eos_token_id))
            generated_text = decode_token_ids(tokenizer, generated_ids)
            normalized_prediction = normalize_baseline_answer(generated_text)
            normalized_gold = normalize_baseline_answer(sample.answer)
            prediction_number = extract_last_number(normalized_prediction)
            gold_number = extract_last_number(normalized_gold)
            exact = normalized_prediction == normalized_gold
            numeric = (
                prediction_number is not None
                and gold_number is not None
                and prediction_number == gold_number
            )
            invalid = prediction_number is None
            exact_matches += int(exact)
            numeric_matches += int(numeric)
            invalid_answers += int(invalid)
            generated_lengths.append(len(generated_ids))
            rows.append(
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "gold": normalized_gold,
                    "prediction": normalized_prediction,
                    "generated_text": generated_text,
                    "generated_length": len(generated_ids),
                    "exact_match": exact,
                    "accuracy_match": numeric,
                    "invalid_answer": invalid,
                }
            )

    total = len(samples)
    metrics = {
        "accuracy": safe_div(numeric_matches, total),
        "exact_match": safe_div(exact_matches, total),
        "invalid_answer_rate": safe_div(invalid_answers, total),
        "generated_length": {
            "mean": mean(generated_lengths),
            "min": min(generated_lengths) if generated_lengths else 0,
            "max": max(generated_lengths) if generated_lengths else 0,
        },
        "generated_length_mean": mean(generated_lengths),
        "num_eval_samples": total,
        "limit_eval_samples": args.limit_eval_samples,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step") if isinstance(checkpoint, Mapping) else None,
        "checkpoint_load_status": load_status,
        "config_path": str(config_path),
        "git_commit_hash": git_commit_hash(),
        "eval_json": str(args.eval_json) if args.eval_json else None,
        "eval_split": args.eval_split,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "answer_normalization": "Coconut baseline: split on '#', remove commas, strip whitespace",
        "generation": {
            "generate_cot": False,
            "final_answer_only": True,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
        },
        "core_eval_contract": {
            "no_teacher_branch": True,
            "ema_updated": False,
            "step_level_eval": False,
            "adapter_expansion": False,
        },
        "examples": rows[: args.save_examples],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_path": str(metrics_path), **metrics}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("lsp_jepa/configs/core/eval_lsp_seq.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default="tiny")
    parser.add_argument("--eval-json", type=Path)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--dataset-id", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--limit-eval-samples", type=int, default=20)
    parser.add_argument("--expected-samples", type=int, default=1319)
    parser.add_argument("--num-latent-steps", type=int, default=4)
    parser.add_argument("--use-predictor-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predictor-head-layers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--save-examples", type=int, default=5)
    args = parser.parse_args()
    provided = provided_cli_flags(sys.argv[1:])
    config_path = resolve_repo_path(args.config)
    if config_path.exists():
        apply_eval_config(args, load_yaml_config(config_path), provided)
    if args.limit_eval_samples is not None and args.limit_eval_samples < 1:
        raise ValueError("--limit-eval-samples must be positive")
    return args


def provided_cli_flags(argv: Sequence[str]) -> set[str]:
    flags = set()
    for token in argv:
        if token.startswith("--"):
            flags.add(token.split("=", maxsplit=1)[0][2:].replace("-", "_"))
    return flags


def apply_eval_config(args: argparse.Namespace, config: Mapping[str, Any], provided: set[str]) -> None:
    set_if_missing(args, provided, "checkpoint", config_get(config, "eval.checkpoint_path"), path=True)
    set_if_missing(args, provided, "output_dir", config_get(config, "eval.output_dir"), path=True)
    set_if_missing(args, provided, "metrics_path", config_get(config, "eval.metrics_path"), path=True)
    set_if_missing(args, provided, "limit_eval_samples", config_get(config, "eval.limit_eval_samples"))
    set_if_missing(args, provided, "max_new_tokens", config_get(config, "eval.max_new_tokens"))
    set_if_missing(args, provided, "save_examples", config_get(config, "eval.save_examples"))
    set_if_missing(args, provided, "model_id", config_get(config, "runtime.model_id"))
    set_if_missing(args, provided, "device", config_get(config, "runtime.device"))
    set_if_missing(args, provided, "seed", config_get(config, "runtime.seed"))
    set_if_missing(args, provided, "eval_json", config_get(config, "data.eval_json"), path=True)
    set_if_missing(args, provided, "eval_split", config_get(config, "data.eval_split"))
    set_if_missing(args, provided, "dataset_id", config_get(config, "data.dataset_id"))
    set_if_missing(args, provided, "dataset_config", config_get(config, "data.dataset_config"))
    set_if_missing(args, provided, "hf_endpoint", config_get(config, "data.hf_endpoint"))
    set_if_missing(args, provided, "expected_samples", config_get(config, "data.expected_samples"))
    set_if_missing(args, provided, "num_latent_steps", config_get(config, "student.num_latent_steps"))
    set_if_missing(args, provided, "use_predictor_head", config_get(config, "student.use_predictor_head"))
    set_if_missing(args, provided, "predictor_head_layers", config_get(config, "student.predictor_head_layers"))


def set_if_missing(
    args: argparse.Namespace,
    provided: set[str],
    attr: str,
    value: Any,
    *,
    path: bool = False,
) -> None:
    if value is None or attr in provided:
        return
    setattr(args, attr, Path(value) if path else value)


def load_eval_samples(args: argparse.Namespace) -> list[Any]:
    limit = args.limit_eval_samples
    if args.eval_json is not None:
        return load_samples_from_path(resolve_repo_path(args.eval_json), limit=limit)
    return load_hf_dataset_samples(
        dataset_id=args.dataset_id,
        config=args.dataset_config,
        split=args.eval_split,
        limit=limit,
        expected_samples=args.expected_samples,
        hf_endpoint=args.hf_endpoint,
    )


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping payload: {path}")
    return checkpoint


def build_eval_tokenizer_and_model(
    samples: Sequence[Any],
    args: argparse.Namespace,
) -> tuple[Any, torch.nn.Module]:
    if args.model_id != "tiny":
        model_args = argparse.Namespace(model_id=args.model_id)
        return build_tokenizer_and_model(model_args, list(samples))
    tokenizer = build_eval_tokenizer(samples, args.num_latent_steps)
    return tokenizer, TinyCausalLM(vocab_size=len(tokenizer))


def build_eval_tokenizer(samples: Sequence[Any], num_latent_steps: int) -> MinimalTokenizer:
    tokenizer = MinimalTokenizer()
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    for sample in samples:
        tokenizer.encode(sample.question + "\n", add_special_tokens=True)
        tokenizer.encode("### " + clean_answer(sample.answer), add_special_tokens=False)
    for _ in range(num_latent_steps):
        tokenizer.convert_tokens_to_ids("<|latent|>")
    return tokenizer


def build_and_load_student(
    checkpoint: Mapping[str, Any],
    *,
    tokenizer: Any,
    base_model: torch.nn.Module,
    model_id: str,
    num_latent_steps: int,
    predictor_head_layers: int,
    use_predictor_head: bool,
    device: torch.device,
) -> tuple[Coconut, LatentPredictor, dict[str, Any]]:
    state = checkpoint.get("student_base_causallm", {}) if isinstance(checkpoint, Mapping) else {}
    model_status = safe_load_state_dict(base_model, state)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    student = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    predictor = LatentPredictor(
        model_hidden_size(base_model),
        layers=predictor_head_layers if use_predictor_head else 0,
    )
    predictor_status = safe_load_state_dict(predictor, checkpoint.get("predictor", {}))
    student.to(device)
    predictor.to(device)
    return student, predictor, {
        "model_id": model_id,
        "student_base_causallm": model_status,
        "predictor": predictor_status,
    }


def checkpoint_vocab_size(state: Mapping[str, torch.Tensor], *, fallback: int) -> int:
    weight = state.get("embed_tokens.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return fallback


def safe_load_state_dict(module: torch.nn.Module, state: Any) -> dict[str, Any]:
    if not isinstance(state, Mapping) or not state:
        return {"loaded": False, "reason": "missing state_dict"}
    result = module.load_state_dict(state, strict=False)
    return {
        "loaded": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def build_eval_input_ids(
    tokenizer: MinimalTokenizer,
    question: str,
    *,
    num_latent_steps: int,
    device: torch.device,
) -> torch.Tensor:
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    token_ids = (
        list(tokenizer.encode(question + "\n", add_special_tokens=True))
        + [start_id]
        + [latent_id] * num_latent_steps
        + [end_id]
    )
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def trim_after_eos(token_ids: list[int], eos_token_id: int) -> list[int]:
    if eos_token_id in token_ids:
        return token_ids[: token_ids.index(eos_token_id)]
    return token_ids


def decode_token_ids(tokenizer: MinimalTokenizer, token_ids: Sequence[int]) -> str:
    if not isinstance(tokenizer, MinimalTokenizer) and hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    tokens = []
    for token_id in token_ids:
        if token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
            continue
        tokens.append(tokenizer._id_to_token.get(int(token_id), f"<unk:{int(token_id)}>"))
    return " ".join(tokens).strip()


def normalize_baseline_answer(text: Any) -> str:
    return str(text).split("#")[-1].replace(",", "").strip()


def extract_last_number(text: str) -> str | None:
    text = re.sub(r"<unk:\d+>", " ", text)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    value = matches[-1]
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return str(number)


def safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def mean(values: Sequence[int]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
