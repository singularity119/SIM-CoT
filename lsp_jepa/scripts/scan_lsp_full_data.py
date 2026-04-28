"""Full-data LSP-JEPA readiness scan.

This is a data and target-construction scanner, not a trainer. It does not
instantiate adapters, run backward, or call optimizer.step(). It loads
GSM8K-style records, builds default LSP-Core teacher inputs with answer tokens
excluded, simulates student latent-token input lengths, and writes a JSON report
with target/mask/leakage/length diagnostics for full-dataset runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsp_jepa.core.target_builder import build_teacher_inputs, filter_answer_only_steps  # noqa: E402


@dataclass(frozen=True)
class ScanSample:
    sample_id: str
    question: str
    cot_steps: list[str]
    answer: str
    source: str


@dataclass(frozen=True)
class ParseResult:
    sample: ScanSample | None
    reasons: list[str]


class WhitespaceTokenizer:
    """Dependency-free tokenizer matching the tiny debug tokenizer semantics."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self) -> None:
        self._token_to_id = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            "<bos>": self.bos_token_id,
        }

    def __len__(self) -> int:
        return len(self._token_to_id)

    def add_tokens(self, tokens: str | list[str]) -> int:
        if isinstance(tokens, str):
            tokens = [tokens]
        before = len(self)
        for token in tokens:
            self._id_for(token)
        return len(self) - before

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._id_for(token)

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [self.bos_token_id, *token_ids]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = [self._id_for(token) for token in re.findall(r"<\|[^>]+?\|>|\S+", str(text))]
        if add_special_tokens:
            return self.build_inputs_with_special_tokens(ids)
        return ids

    def _id_for(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._token_to_id)
        return self._token_to_id[token]


ANSWER_PREFIX_RE = re.compile(
    r"(?i)(?:^|\b)(?:the\s+)?answer\s+is\b|(?:^|\b)final\s+answer\b|####"
)


def main() -> None:
    args = parse_args()
    if args.num_latent_steps < 1:
        raise ValueError("--num-latent-steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    tokenizer = build_tokenizer(args)
    records, source_name = load_records_for_args(args)
    parse_results = records_to_samples(records, source_name=source_name, limit=args.max_samples)
    report = scan_samples(parse_results, tokenizer=tokenizer, args=args, source_name=source_name)

    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "acceptance": report["acceptance"]}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, help="Offline JSON/JSONL/Parquet GSM8K-style data path.")
    parser.add_argument(
        "--dataset-source",
        default="gsm8k",
        choices=("gsm8k", "gsm8k_smoke", "synthetic"),
        help="Used only when --data-path is not provided.",
    )
    parser.add_argument("--dataset-id", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=7473)
    parser.add_argument("--tokenizer-id", default="whitespace", help="whitespace or a HF tokenizer id.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-latent-steps", type=int, default=4)
    parser.add_argument("--configured-max-seq-len", type=int, default=512)
    parser.add_argument("--max-skipped-ids", type=int, default=100)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("lsp_jepa/runs/pr10_mvp/full_data_readiness_report.json"),
    )
    return parser.parse_args()


def build_tokenizer(args: argparse.Namespace) -> Any:
    if args.tokenizer_id == "whitespace":
        tokenizer = WhitespaceTokenizer()
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for non-whitespace tokenizers") from exc
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("tokenizer must define pad_token_id")
    if hasattr(tokenizer, "add_tokens"):
        tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    return tokenizer


def load_records_for_args(args: argparse.Namespace) -> tuple[list[Mapping[str, Any]], str]:
    if args.data_path is not None:
        path = resolve_repo_path(args.data_path)
        return read_records(path), str(path)
    if args.dataset_source == "gsm8k_smoke":
        path = REPO_ROOT / "lsp_jepa" / "data" / "gsm8k_smoke.jsonl"
        return read_records(path), str(path)
    if args.dataset_source == "synthetic":
        limit = args.max_samples or args.expected_samples
        return synthetic_records(limit), f"synthetic:{limit}"
    return load_hf_records(
        dataset_id=args.dataset_id,
        config=args.dataset_config,
        split=args.dataset_split,
        limit=args.max_samples,
        expected_samples=args.expected_samples,
        hf_endpoint=args.hf_endpoint,
    ), f"hf:{args.dataset_id}:{args.dataset_config}:{args.dataset_split}"


def read_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        records.append(value)
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, Mapping)]
        if isinstance(payload, Mapping):
            data = payload.get("data", payload.get("rows", []))
            return [item for item in data if isinstance(item, Mapping)]
        raise ValueError(f"Unsupported JSON payload in {path}")
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas/pyarrow is required to read parquet data") from exc
        frame = pd.read_parquet(path)
        return [dict(row) for row in frame.to_dict(orient="records")]
    raise ValueError(f"Unsupported data file suffix: {path.suffix}")


def load_hf_records(
    *,
    dataset_id: str,
    config: str,
    split: str,
    limit: int | None,
    expected_samples: int,
    hf_endpoint: str,
) -> list[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return load_hf_rows_via_dataset_server(
            dataset_id=dataset_id,
            config=config,
            split=split,
            limit=limit,
            expected_samples=expected_samples,
            hf_endpoint=hf_endpoint,
        )
    os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    dataset = load_dataset(dataset_id, config, split=split)
    row_count = len(dataset) if limit is None else min(limit, len(dataset))
    return [dataset[index] for index in range(row_count)]


def load_hf_rows_via_dataset_server(
    *,
    dataset_id: str,
    config: str,
    split: str,
    limit: int | None,
    expected_samples: int,
    hf_endpoint: str,
    page_size: int = 100,
) -> list[Mapping[str, Any]]:
    target = expected_samples if limit is None else limit
    rows: list[Mapping[str, Any]] = []
    offset = 0
    base_url = os.environ.get("HF_DATASETS_SERVER", "https://datasets-server.huggingface.co")
    while offset < target:
        length = min(page_size, target - offset)
        query = {"dataset": dataset_id, "config": config, "split": split, "offset": str(offset), "length": str(length)}
        url = base_url.rstrip("/") + "/rows?" + urllib.parse.urlencode(query)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(
                "datasets is not installed and dataset-server is not reachable; "
                f"install lsp_jepa/requirements-mvp.txt or pass --data-path. HF_ENDPOINT={hf_endpoint}"
            ) from exc
        page = [item.get("row", item) for item in payload.get("rows", []) if isinstance(item.get("row", item), Mapping)]
        if not page:
            break
        rows.extend(page)
        if len(page) < length:
            break
        offset += len(page)
    return rows


def synthetic_records(limit: int) -> list[Mapping[str, Any]]:
    records = []
    for index in range(limit):
        a = 2 + (index * 3) % 37
        b = 3 + (index * 5) % 29
        c = 1 + (index * 7) % 17
        answer = a + b + c
        records.append(
            {
                "question": f"A shelf has {a} red books, {b} blue books, and {c} green books. How many books are on the shelf?",
                "answer": f"Start with {a} red books.\nAdd {b} blue books to get {a + b} books.\nAdd {c} green books to get {answer} books.\n#### {answer}",
            }
        )
    return records


def records_to_samples(records: Sequence[Mapping[str, Any]], *, source_name: str, limit: int | None) -> list[ParseResult]:
    selected = records if limit is None else records[:limit]
    results = []
    for index, record in enumerate(selected):
        sample_id = str(record.get("id") or record.get("sample_id") or f"{source_name}#{index}")
        results.append(parse_record(record, sample_id=sample_id, source=f"{source_name}#{index}"))
    return results


def parse_record(record: Mapping[str, Any], *, sample_id: str, source: str) -> ParseResult:
    reasons = []
    question = first_text(record, ("question", "query", "problem"))
    if question is None or not question.strip():
        reasons.append("missing_question")

    answer_value = first_text(record, ("answer", "final_answer", "target", "label", "short_answer"))
    cot_value = first_value(record, ("cot", "steps", "rationale", "chain_of_thought", "solution", "response", "annotation"))
    cot_from_answer = None
    answer = clean_answer(answer_value)
    if answer_value is not None and "####" in str(answer_value):
        cot_from_answer, answer_from_answer = split_gsm8k_answer_preserving_final(answer_value)
        if answer_from_answer:
            answer = answer_from_answer

    if cot_value is not None:
        cot_steps = split_cot_steps_preserving_answer(cot_value)
    elif cot_from_answer is not None:
        cot_steps = split_cot_steps_preserving_answer(cot_from_answer)
    else:
        cot_steps = []

    if not cot_steps:
        reasons.append("missing_cot_steps")
    if not answer:
        reasons.append("missing_answer")
    if reasons:
        return ParseResult(sample=None, reasons=reasons)
    return ParseResult(sample=ScanSample(sample_id, str(question).strip(), cot_steps, answer, source), reasons=[])


def scan_samples(parse_results: Sequence[ParseResult], *, tokenizer: Any, args: argparse.Namespace, source_name: str) -> dict[str, Any]:
    valid_question_count = 0
    parsed_samples = 0
    skipped_samples: list[dict[str, Any]] = []
    skip_reason_counts: Counter[str] = Counter()
    empty_teacher_reasons: Counter[str] = Counter()
    parse_reason_counts: Counter[str] = Counter()
    leakage_reason_counts: Counter[str] = Counter()
    teacher_lengths: list[int] = []
    student_lengths: list[int] = []
    valid_step_counts: list[int] = []
    original_step_counts: list[int] = []
    final_token_positions: list[int] = []
    final_original_step_indices: list[int] = []
    answer_value_occurrences = 0
    final_answer_only_filtered = 0
    samples_with_original_steps = 0
    teacher_empty_count = 0
    leakage_count = 0
    over_configured_max = 0
    over_configured_ids: list[str] = []
    latent_nonempty_flags: list[bool] = []

    for index, result in enumerate(parse_results):
        if result.sample is None:
            parse_reason_counts.update(result.reasons)
            sample_id = f"{source_name}#{index}"
            skipped_samples.append({"id": sample_id, "reasons": result.reasons})
            skip_reason_counts.update(result.reasons)
            latent_nonempty_flags.append(False)
            continue

        sample = result.sample
        parsed_samples += 1
        valid_question_count += int(bool(sample.question.strip()))
        original_step_counts.append(len(sample.cot_steps))
        samples_with_original_steps += int(bool(sample.cot_steps))

        filter_result = filter_answer_only_steps(sample.cot_steps, answer=sample.answer, exclude_answer_prefix=True)
        final_answer_only_filtered += int(final_step_was_filtered(sample.cot_steps, filter_result.dropped_indices))
        valid_step_counts.append(len(filter_result.steps))
        student_lengths.append(student_input_length(tokenizer, sample, args.num_latent_steps))
        latent_nonempty_flags.append(args.num_latent_steps > 0 and bool(sample.question.strip()))

        sample_skip_reasons = []
        try:
            teacher_inputs = build_teacher_inputs(
                tokenizer,
                sample.question,
                sample.cot_steps,
                answers=sample.answer,
                padding=False,
                truncation=False,
                return_tensors=None,
                exclude_answer_tokens=True,
                exclude_answer_prefix=True,
                filter_answer_steps=True,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"teacher_build_error:{type(exc).__name__}"
            sample_skip_reasons.append(reason)
            empty_teacher_reasons[reason] += 1
            skipped_samples.append({"id": sample.sample_id, "reasons": sample_skip_reasons})
            skip_reason_counts.update(sample_skip_reasons)
            continue

        teacher_length = len(teacher_inputs["input_ids"][0])
        teacher_lengths.append(teacher_length)
        if teacher_length > args.configured_max_seq_len:
            over_configured_max += 1
            if len(over_configured_ids) < args.max_skipped_ids:
                over_configured_ids.append(sample.sample_id)

        step_mask = [bool(value) for value in teacher_inputs["step_mask"][0]]
        step_boundaries = [int(value) for value in teacher_inputs["step_boundaries"][0]]
        kept_indices = [int(value) for value in teacher_inputs["kept_step_indices"][0]]
        if sum(step_mask) == 0:
            teacher_empty_count += 1
            reason = empty_teacher_reason(sample, filter_result)
            empty_teacher_reasons[reason] += 1
            sample_skip_reasons.append(reason)

        final_position = last_valid_value(step_boundaries, step_mask)
        if final_position is not None:
            final_token_positions.append(final_position)
        if kept_indices:
            final_original_step_indices.append(kept_indices[-1])

        leakage_reasons = answer_leakage_reasons(sample, teacher_inputs)
        if leakage_reasons:
            leakage_count += 1
            leakage_reason_counts.update(leakage_reasons)
            sample_skip_reasons.extend([f"answer_leakage:{reason}" for reason in leakage_reasons])
        answer_value_occurrences += int(answer_value_occurs_in_reasoning(sample, teacher_inputs))

        if sample_skip_reasons:
            skipped_samples.append({"id": sample.sample_id, "reasons": sample_skip_reasons})
            skip_reason_counts.update(sample_skip_reasons)

    batch_count, latent_empty_batches = latent_batch_counts(latent_nonempty_flags, args.batch_size)
    recommended = recommend_settings(
        teacher_lengths=teacher_lengths,
        student_lengths=student_lengths,
        configured_max_seq_len=args.configured_max_seq_len,
    )
    total_records = len(parse_results)
    return {
        "run": {
            "name": "pr10_mvp_full_data_readiness",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": "lsp_jepa/scripts/scan_lsp_full_data.py",
            "no_training": True,
            "optimizer_step_executed": False,
        },
        "config": {
            "dataset_source": args.dataset_source,
            "dataset_id": args.dataset_id,
            "dataset_config": args.dataset_config,
            "dataset_split": args.dataset_split,
            "data_path": str(args.data_path) if args.data_path is not None else None,
            "hf_endpoint": args.hf_endpoint,
            "tokenizer_id": args.tokenizer_id,
            "batch_size": args.batch_size,
            "num_latent_steps": args.num_latent_steps,
            "configured_max_seq_len": args.configured_max_seq_len,
            "expected_samples": args.expected_samples,
            "max_samples": args.max_samples,
        },
        "data": {
            "source": source_name,
            "sample_total": total_records,
            "parsed_sample_count": parsed_samples,
            "valid_question_count": valid_question_count,
            "preparse_skip_count": sum(parse_reason_counts.values()),
            "preparse_skip_reason_counts": dict(sorted(parse_reason_counts.items())),
        },
        "cot_steps": {
            "original_step_count_distribution": distribution(original_step_counts),
            "valid_step_count_distribution": distribution(valid_step_counts),
            "valid_step_count_histogram": histogram(valid_step_counts),
            "answer_only_final_step_filtered_count": final_answer_only_filtered,
            "answer_only_final_step_filtered_rate": ratio(final_answer_only_filtered, samples_with_original_steps),
        },
        "teacher_target": {
            "empty_teacher_target_count": teacher_empty_count,
            "empty_teacher_target_rate": ratio(teacher_empty_count, parsed_samples),
            "empty_teacher_target_reason_counts": dict(sorted(empty_teacher_reasons.items())),
            "target_construction": {
                "exclude_answer_tokens": True,
                "exclude_answer_prefix": True,
                "filter_answer_steps": True,
                "target_pooling": "step_last_token",
            },
        },
        "latent": {
            "batch_size": args.batch_size,
            "batch_count": batch_count,
            "latent_empty_batch_count": latent_empty_batches,
            "latent_empty_batch_rate": ratio(latent_empty_batches, batch_count),
            "num_latent_steps": args.num_latent_steps,
        },
        "lengths": {
            "teacher_input_tokens": distribution(teacher_lengths),
            "student_input_tokens": distribution(student_lengths),
            "teacher_over_configured_max_seq_len_count": over_configured_max,
            "teacher_over_configured_max_seq_len_rate": ratio(over_configured_max, len(teacher_lengths)),
            "teacher_over_configured_max_seq_len_ids_first100": over_configured_ids,
        },
        "final_valid_reasoning_step_position": {
            "token_position_distribution": distribution(final_token_positions),
            "original_step_index_distribution": distribution(final_original_step_indices),
            "original_step_index_histogram": histogram(final_original_step_indices),
        },
        "answer_leakage": {
            "answer_leakage_count": leakage_count,
            "answer_leakage_rate": ratio(leakage_count, parsed_samples),
            "answer_leakage_reason_counts": dict(sorted(leakage_reason_counts.items())),
            "answer_value_occurs_in_reasoning_count": answer_value_occurrences,
            "answer_value_occurs_in_reasoning_note": "Informational only: final numeric values inside retained reasoning steps are not counted as leakage unless answer markers/prefixes or answer-only steps remain.",
        },
        "skips": {
            "skipped_sample_count": len(skipped_samples),
            "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
            "skipped_sample_ids_first100": [item["id"] for item in skipped_samples[: args.max_skipped_ids]],
            "skipped_samples_first100": skipped_samples[: args.max_skipped_ids],
        },
        "oom_risk": oom_risk_summary(
            teacher_lengths=teacher_lengths,
            student_lengths=student_lengths,
            configured_max_seq_len=args.configured_max_seq_len,
            batch_size=args.batch_size,
        ),
        "recommendations": recommended,
        "acceptance": {
            "script_completed": True,
            "json_report_written": True,
            "no_optimizer_step": True,
            "answer_leakage_default_zero": leakage_count == 0,
            "empty_teacher_target_rate_zero": teacher_empty_count == 0,
            "full_expected_sample_count_reached": args.expected_samples <= 0 or total_records >= args.expected_samples,
        },
    }


def final_step_was_filtered(raw_steps: Sequence[str], dropped_indices: Sequence[int]) -> bool:
    return bool(raw_steps) and len(raw_steps) - 1 in set(dropped_indices)


def empty_teacher_reason(sample: ScanSample, filter_result: Any) -> str:
    if not sample.cot_steps:
        return "no_raw_cot_steps"
    if not filter_result.steps:
        return "all_steps_filtered_as_answer_only"
    return "no_valid_step_boundaries"


def answer_leakage_reasons(sample: ScanSample, teacher_inputs: Mapping[str, Any]) -> list[str]:
    reasons = []
    text = str(teacher_inputs.get("texts", [""])[0])
    filtered_steps = teacher_inputs.get("filtered_steps", [[]])[0]
    if "####" in text:
        reasons.append("answer_marker_in_teacher_text")
    normalized_answer = normalize_answer(sample.answer)
    for step in filtered_steps:
        step_text = str(step)
        if ANSWER_PREFIX_RE.search(step_text):
            reasons.append("answer_prefix_in_filtered_step")
        if normalized_answer and normalize_answer(step_text) == normalized_answer:
            reasons.append("answer_only_token_in_filtered_step")
    return sorted(set(reasons))


def answer_value_occurs_in_reasoning(sample: ScanSample, teacher_inputs: Mapping[str, Any]) -> bool:
    normalized_answer = normalize_answer(sample.answer)
    if not normalized_answer:
        return False
    filtered_steps = teacher_inputs.get("filtered_steps", [[]])[0]
    return any(normalized_answer in (normalize_answer(str(step)) or "") for step in filtered_steps)


def student_input_length(tokenizer: Any, sample: ScanSample, num_latent_steps: int) -> int:
    question_ids = list(tokenizer.encode(sample.question + "\n", add_special_tokens=True))
    return len(question_ids) + num_latent_steps + 2


def latent_batch_counts(flags: Sequence[bool], batch_size: int) -> tuple[int, int]:
    if not flags:
        return 0, 0
    batch_count = math.ceil(len(flags) / batch_size)
    empty_count = 0
    for start in range(0, len(flags), batch_size):
        if not any(flags[start : start + batch_size]):
            empty_count += 1
    return batch_count, empty_count


def recommend_settings(*, teacher_lengths: Sequence[int], student_lengths: Sequence[int], configured_max_seq_len: int) -> dict[str, Any]:
    teacher_stats = distribution(teacher_lengths)
    student_stats = distribution(student_lengths)
    max_observed = int(max(teacher_stats["max"], student_stats["max"])) if teacher_lengths else 0
    p99_observed = int(max(teacher_stats["p99"], student_stats["p99"])) if teacher_lengths else 0
    no_truncation_max_seq_len = max(round_up(max_observed, 64), 64)
    practical_p99_max_seq_len = max(round_up(p99_observed, 64), 64)
    batch_size = batch_size_for_seq_len(no_truncation_max_seq_len)
    return {
        "recommended_max_seq_len": no_truncation_max_seq_len,
        "practical_p99_max_seq_len": practical_p99_max_seq_len,
        "no_truncation_max_seq_len": no_truncation_max_seq_len,
        "configured_max_seq_len": configured_max_seq_len,
        "configured_max_seq_len_covers_full_data": configured_max_seq_len >= max_observed,
        "recommended_batch_size": batch_size,
        "token_budget_at_recommended_batch": batch_size * no_truncation_max_seq_len,
    }


def oom_risk_summary(*, teacher_lengths: Sequence[int], student_lengths: Sequence[int], configured_max_seq_len: int, batch_size: int) -> dict[str, Any]:
    if not teacher_lengths:
        return {"risk_level": "unknown", "reason": "no teacher lengths"}
    max_len = max(max(teacher_lengths), max(student_lengths or [0]))
    reasons = []
    risk = "low"
    if max_len > configured_max_seq_len:
        risk = "medium"
        reasons.append("configured_max_seq_len_truncates_some_teacher_inputs")
    if batch_size * max(configured_max_seq_len, max_len) > 16384:
        risk = "medium"
        reasons.append("batch_token_budget_above_16k")
    if max_len > 1024 or batch_size * max(configured_max_seq_len, max_len) > 32768:
        risk = "high"
        reasons.append("long_sequence_or_large_batch_token_budget")
    return {
        "risk_level": risk,
        "reasons": reasons,
        "max_observed_length": max_len,
        "teacher_p99_length": percentile(teacher_lengths, 99),
        "teacher_over_512_count": sum(length > 512 for length in teacher_lengths),
        "teacher_over_1024_count": sum(length > 1024 for length in teacher_lengths),
        "configured_batch_token_budget": batch_size * configured_max_seq_len,
    }


def first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    value = first_value(record, keys)
    return None if value is None else str(value)


def split_gsm8k_answer_preserving_final(answer_text: str) -> tuple[str | None, str | None]:
    text = str(answer_text).strip()
    parts = text.rsplit("####", maxsplit=1)
    if len(parts) == 1:
        return text or None, clean_answer(text)
    return text, clean_answer(parts[1]) or None


def split_cot_steps_preserving_answer(value: Any | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(step).strip() for step in value if str(step).strip()]
    text = str(value).replace("\r\n", "\n").strip()
    line_steps = [line.strip() for line in text.split("\n") if line.strip()]
    if len(line_steps) > 1:
        return line_steps
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def clean_answer(value: Any | None) -> str:
    if value is None:
        return ""
    answer = str(value).strip()
    if "####" in answer:
        answer = answer.rsplit("####", maxsplit=1)[1]
    answer = answer.replace("####", "").strip()
    answer = re.sub(r"(?i)^(?:the\s+)?answer\s+is\s*:?\s*", "", answer).strip()
    answer = re.sub(r"(?i)^final\s+answer\s*:?\s*", "", answer).strip()
    return answer


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    normalized = re.sub(r"\W+", "", str(answer).lower())
    return normalized or None


def last_valid_value(values: Sequence[int], mask: Sequence[bool]) -> int | None:
    output = None
    for value, valid in zip(values, mask, strict=True):
        if valid:
            output = int(value)
    return output


def distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}
    sorted_values = sorted(int(value) for value in values)
    return {
        "count": len(sorted_values),
        "p50": percentile_sorted(sorted_values, 50),
        "p90": percentile_sorted(sorted_values, 90),
        "p95": percentile_sorted(sorted_values, 95),
        "p99": percentile_sorted(sorted_values, 99),
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / len(sorted_values),
    }


def histogram(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(key): counts[key] for key in sorted(counts)}


def percentile(values: Sequence[int], percent: float) -> int:
    if not values:
        return 0
    return percentile_sorted(sorted(int(value) for value in values), percent)


def percentile_sorted(sorted_values: Sequence[int], percent: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    rank = (len(sorted_values) - 1) * (percent / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return int(sorted_values[lower])
    fraction = rank - lower
    return int(math.ceil(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction))


def ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def round_up(value: int, multiple: int) -> int:
    return multiple if value <= 0 else int(math.ceil(value / multiple) * multiple)


def batch_size_for_seq_len(seq_len: int) -> int:
    if seq_len <= 256:
        return 32
    if seq_len <= 512:
        return 16
    if seq_len <= 1024:
        return 8
    if seq_len <= 2048:
        return 4
    return 2


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    main()
