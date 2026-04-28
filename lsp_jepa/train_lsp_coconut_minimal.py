"""Minimal sequence-level LSP-JEPA smoke training closure on Coconut.

This script is intentionally small: one student forward, one EMA-teacher target
forward, one scalar sequence-level alignment loss, optional host answer CE, one
backward/optimizer step, one EMA update, and JSONL metrics. It supports tiny
real GSM8K/GSM8K-Aug style samples without adding step-level objectives,
adapter expansion, distributed training, wandb, or checkpoint management.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
COCONUT_DIR = REPO_ROOT / "Coconut"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COCONUT_DIR) not in sys.path:
    sys.path.insert(0, str(COCONUT_DIR))

try:
    from transformers.models.gpt2 import GPT2LMHeadModel as _GPT2LMHeadModel
except ModuleNotFoundError:
    transformers_stub = types.ModuleType("transformers")
    models_stub = types.ModuleType("transformers.models")
    gpt2_stub = types.ModuleType("transformers.models.gpt2")

    class _GPT2LMHeadModel(nn.Module):
        pass

    gpt2_stub.GPT2LMHeadModel = _GPT2LMHeadModel
    transformers_stub.models = models_stub
    models_stub.gpt2 = gpt2_stub
    sys.modules.setdefault("transformers", transformers_stub)
    sys.modules.setdefault("transformers.models", models_stub)
    sys.modules.setdefault("transformers.models.gpt2", gpt2_stub)

from coconut import Coconut  # noqa: E402
from lsp_jepa.adapters.coconut_lsp_adapter import CoconutLSPAdapter  # noqa: E402
from lsp_jepa.core.ema_teacher import EMATeacher  # noqa: E402
from lsp_jepa.core.latent_interface import TeacherTargetBatch  # noqa: E402
from lsp_jepa.core.losses import compute_lsp_state_loss  # noqa: E402
from lsp_jepa.core.metrics import per_dim_variance  # noqa: E402
from lsp_jepa.core.target_builder import (  # noqa: E402
    build_teacher_inputs,
    gather_step_boundary_hidden_states,
)


@dataclass(frozen=True)
class Sample:
    question: str
    cot_steps: list[str]
    answer: str
    source: str = "unknown"


class MinimalTokenizer:
    """Whitespace tokenizer for dependency-free one-batch smoke training."""

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
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}

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
        ids = [self._id_for(token) for token in re.findall(r"<\|[^>]+?\|>|\S+", text)]
        if add_special_tokens:
            return self.build_inputs_with_special_tokens(ids)
        return ids

    def _id_for(self, token: str) -> int:
        if token not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        return self._token_to_id[token]


class TinyCausalLM(nn.Module):
    """Small causal LM with HF-like outputs for local MVP verification."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        max_positions: int = 256,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(max_positions, hidden_size)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            n_embd=hidden_size,
            num_hidden_layers=num_layers,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def resize_token_embeddings(self, new_size: int) -> nn.Embedding:
        if new_size == self.embed_tokens.num_embeddings:
            return self.embed_tokens
        if new_size < self.embed_tokens.num_embeddings:
            raise ValueError("TinyCausalLM only supports growing token embeddings")
        old_embed = self.embed_tokens
        old_head = self.lm_head
        self.embed_tokens = nn.Embedding(new_size, old_embed.embedding_dim).to(
            device=old_embed.weight.device,
            dtype=old_embed.weight.dtype,
        )
        self.lm_head = nn.Linear(old_head.in_features, new_size, bias=False).to(
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        with torch.no_grad():
            self.embed_tokens.weight[: old_embed.num_embeddings].copy_(old_embed.weight)
            self.lm_head.weight[: old_head.out_features].copy_(old_head.weight)
        self.config.vocab_size = new_size
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> Any:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.clamp(max=self.position_embed.num_embeddings - 1)
        token_states = inputs_embeds + self.position_embed(position_ids)

        previous_cache = None
        if past_key_values:
            previous_cache = past_key_values[0][0].to(device=token_states.device)
        previous_sum = None
        if previous_cache is not None and previous_cache.shape[2] > 0:
            previous_sum = previous_cache[:, 0, -1, :]

        cumulative = token_states.cumsum(dim=1)
        if previous_sum is not None:
            cumulative = cumulative + previous_sum.unsqueeze(1)
        denom = (position_ids.to(dtype=token_states.dtype) + 1.0).clamp_min(1.0)
        hidden = cumulative / denom.unsqueeze(-1)

        hidden_states = [hidden]
        for layer, norm in zip(self.layers, self.norms, strict=True):
            hidden = norm(hidden + layer(hidden))
            hidden_states.append(hidden)

        logits = self.lm_head(hidden)
        cache_len = (
            int(attention_mask.shape[1])
            if attention_mask is not None
            else int(position_ids.max().detach().item()) + 1
        )
        cache = token_states.new_zeros(batch_size, 1, cache_len, token_states.shape[-1])
        if previous_cache is not None:
            copy_len = min(previous_cache.shape[2], cache_len)
            cache[:, :, :copy_len, :] = previous_cache[:, :, :copy_len, :]
        scatter_index = position_ids.clamp(min=0, max=max(cache_len - 1, 0))
        for batch_idx in range(batch_size):
            cache[batch_idx, 0, scatter_index[batch_idx], :] = cumulative[batch_idx]

        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            last_hidden_state=hidden,
            past_key_values=((cache, cache),),
        )


def main() -> None:
    args = parse_args()
    if args.objective != "sequence":
        raise ValueError("PR8-MVP only supports objective=sequence")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.num_latent_steps < 1:
        raise ValueError("--num-latent-steps must be at least 1")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    samples = load_samples(args)
    if len(samples) < args.batch_size:
        raise RuntimeError(
            f"need at least batch_size={args.batch_size} samples, found {len(samples)}"
        )
    samples = samples[: args.batch_size]

    tokenizer, base_model = build_tokenizer_and_model(args, samples)
    base_model.to(device)

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    student = Coconut(
        base_model,
        latent_id,
        start_id,
        end_id,
        tokenizer.eos_token_id,
    ).to(device)
    student.train()

    ema_teacher = EMATeacher.from_student(
        student.base_causallm,
        decay=args.ema_decay,
        trainable_only=True,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    metrics_path = resolve_metrics_path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists() and not args.append_metrics:
        metrics_path.unlink()

    student_batch = build_student_batch(
        tokenizer,
        samples,
        latent_id=latent_id,
        start_id=start_id,
        end_id=end_id,
        num_latent_steps=args.num_latent_steps,
        device=device,
    )
    teacher_inputs = build_teacher_batch(tokenizer, samples, args, device)
    answer_leakage_ok = validate_no_answer_leakage(
        samples,
        teacher_inputs,
        include_answer_tokens=args.include_answer_tokens,
        include_answer_prefix=args.include_answer_prefix,
    )
    adapter = CoconutLSPAdapter(latent_token_id=latent_id)

    for train_step in range(1, args.max_steps + 1):
        teacher_output = ema_teacher(
            input_ids=teacher_inputs["teacher_input_ids"],
            attention_mask=teacher_inputs["attention_mask"],
            output_hidden_states=True,
        )
        teacher_targets = gather_step_boundary_hidden_states(
            teacher_output,
            teacher_inputs["step_boundaries"],
            teacher_inputs["step_mask"],
            attention_mask=teacher_inputs["attention_mask"],
            step_starts=teacher_inputs["step_starts"],
            target_layer=args.target_layer,
            target_pooling="step_last_token",
            detach=True,
        )
        teacher_target_mask_nonempty = bool(
            teacher_targets.target_mask.any().detach().cpu().item()
        )
        if not teacher_target_mask_nonempty:
            raise RuntimeError("teacher target mask is empty")
        final_teacher = select_last_valid_target(teacher_targets)

        host_output = adapter.forward_student(
            student,
            {"student_inputs": student_batch},
            output_latent_states=True,
        )
        latent_mask_nonempty = bool(host_output.latent_mask.any().detach().cpu().item())
        if not latent_mask_nonempty:
            raise RuntimeError("student latent mask is empty")
        h_final, h_final_mask = select_last_valid_state(
            host_output.latent_states,
            host_output.latent_mask,
        )
        z_final = final_teacher.target_states.to(device=h_final.device, dtype=h_final.dtype)
        z_final_mask = final_teacher.target_mask.to(device=h_final.device)
        final_mask = h_final_mask & z_final_mask

        lsp_loss = compute_lsp_state_loss(
            h_final,
            z_final,
            final_mask,
            alignment="normalized_mse",
        )
        host_answer_ce = host_output.host_losses.get("host_answer_ce")
        if host_answer_ce is None:
            raise RuntimeError("Coconut host output did not provide host_answer_ce")
        total_loss = args.lsp_weight * lsp_loss + args.host_answer_ce_weight * host_answer_ce
        latent_variance = per_dim_variance(h_final, final_mask)["mean"]

        validate_finite("total_loss", total_loss)
        validate_finite("lsp_loss", lsp_loss)
        validate_finite("host_answer_ce", host_answer_ce)
        validate_finite("latent_variance", latent_variance)
        if not bool(final_mask.any().detach().cpu().item()):
            raise RuntimeError("no valid final sequence alignment targets were found")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        student_grad_l1 = grad_l1(student)
        if student_grad_l1 <= 0.0 or not math.isfinite(student_grad_l1):
            raise RuntimeError("student parameters did not receive finite gradients")
        if any(param.grad is not None for param in ema_teacher.parameters()):
            raise RuntimeError("EMA teacher parameters received gradients")

        teacher_before = clone_params(ema_teacher.teacher_model)
        optimizer.step()
        ema_teacher.update(student.base_causallm)
        teacher_delta_l1 = param_delta_l1(teacher_before, ema_teacher.teacher_model)
        if teacher_delta_l1 <= 0.0 or not math.isfinite(teacher_delta_l1):
            raise RuntimeError("EMA update did not change teacher parameters")

        metrics = {
            "step": train_step,
            "max_steps": args.max_steps,
            "data_source": args.dataset_source if args.data_path is None else str(args.data_path),
            "sample_sources": [sample.source for sample in samples],
            "total_loss": to_float(total_loss),
            "lsp_loss": to_float(lsp_loss),
            "host_answer_ce": to_float(host_answer_ce),
            "host_answer_ce_weight": args.host_answer_ce_weight,
            "latent_variance": to_float(latent_variance),
            "student_grad_l1": student_grad_l1,
            "teacher_grad_params": sum(
                param.grad is not None for param in ema_teacher.parameters()
            ),
            "teacher_delta_l1": teacher_delta_l1,
            "teacher_target_mask_nonempty": teacher_target_mask_nonempty,
            "latent_mask_nonempty": latent_mask_nonempty,
            "valid_teacher_targets": int(
                teacher_targets.target_mask.sum().detach().cpu().item()
            ),
            "valid_final_targets": int(final_mask.sum().detach().cpu().item()),
            "teacher_input_ids_shape": list(teacher_inputs["teacher_input_ids"].shape),
            "final_valid_step_position": tensor_to_int_list(
                teacher_inputs["final_valid_step_position"]
            ),
            "answer_leakage_ok": answer_leakage_ok,
        }
        append_metrics(metrics_path, metrics)
        print(json.dumps(metrics, sort_keys=True))

    print(json.dumps({"metrics_path": str(metrics_path), "completed_steps": args.max_steps}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="tiny", help="'tiny' or a local/HF model id")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--objective", default="sequence")
    parser.add_argument("--lsp-weight", type=float, default=1.0)
    parser.add_argument("--host-answer-ce-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--target-layer", default="last_2")
    parser.add_argument("--num-latent-steps", type=int, default=2)
    parser.add_argument("--max-teacher-length", type=int, default=256)
    parser.add_argument("--metrics-path", default="lsp_jepa/runs/pr8_mvp/metrics.jsonl")
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--dataset-source",
        default="gsm8k_smoke",
        choices=("gsm8k_smoke", "gsm8k", "gsm8k_aug", "synthetic"),
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--include-answer-tokens", action="store_true")
    parser.add_argument("--include-answer-prefix", action="store_true")
    parser.add_argument(
        "--reasoning-step-filter",
        default="exclude_answer_only_steps",
        choices=("exclude_answer_only_steps", "none"),
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def default_samples() -> list[Sample]:
    return [
        Sample(
            question="What is two plus three?",
            cot_steps=[
                "Represent two as 2.",
                "Add three more to make five.",
                "#### 5",
            ],
            answer="5",
            source="synthetic",
        ),
        Sample(
            question="A box has four red pens and three blue pens. How many pens?",
            cot_steps=[
                "Start with four red pens.",
                "Add three blue pens for a total of seven pens.",
                "The answer is 7",
            ],
            answer="7",
            source="synthetic",
        ),
    ]


def load_samples(args: argparse.Namespace) -> list[Sample]:
    if args.data_path is not None:
        return load_samples_from_path(args.data_path, limit=args.num_samples)
    if args.dataset_source == "synthetic":
        return default_samples()[: args.num_samples]
    if args.dataset_source == "gsm8k_smoke":
        return load_samples_from_path(
            REPO_ROOT / "lsp_jepa" / "data" / "gsm8k_smoke.jsonl",
            limit=args.num_samples,
        )
    dataset_id = "gsm8k" if args.dataset_source == "gsm8k" else "zen-E/GSM8k-Aug"
    config = args.dataset_config if args.dataset_source == "gsm8k" else None
    return load_hf_dataset_samples(
        dataset_id=dataset_id,
        config=config,
        split=args.dataset_split,
        limit=args.num_samples,
    )


def load_samples_from_path(path: Path, *, limit: int) -> list[Sample]:
    path = resolve_repo_path(path)
    records = read_records(path)
    samples = samples_from_records(records, source=str(path))
    return require_enough_samples(samples[:limit], path)


def load_hf_dataset_samples(
    *,
    dataset_id: str,
    config: str | None,
    split: str,
    limit: int,
) -> list[Sample]:
    try:
        from datasets import load_dataset
    except ImportError:
        rows = load_hf_rows_via_dataset_server(
            dataset_id=dataset_id,
            config=config,
            split=split,
            limit=limit,
        )
    else:
        if config is None:
            dataset = load_dataset(dataset_id, split=split)
        else:
            dataset = load_dataset(dataset_id, config, split=split)
        rows = [dataset[idx] for idx in range(min(limit, len(dataset)))]
    return require_enough_samples(
        samples_from_records(rows, source=f"hf:{dataset_id}:{split}")[:limit],
        dataset_id,
    )


def load_hf_rows_via_dataset_server(
    *,
    dataset_id: str,
    config: str | None,
    split: str,
    limit: int,
) -> list[Mapping[str, Any]]:
    query: dict[str, str] = {
        "dataset": dataset_id,
        "split": split,
        "offset": "0",
        "length": str(limit),
    }
    if config is not None:
        query["config"] = config
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(query)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            "datasets is not installed and Hugging Face dataset-server rows "
            f"could not be fetched for {dataset_id}; pass --data-path instead"
        ) from exc
    if "rows" not in payload:
        raise RuntimeError(f"unexpected dataset-server response for {dataset_id}: {payload}")
    rows = []
    for item in payload["rows"]:
        row = item.get("row", item)
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def read_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping):
            records = payload.get("data", [])
        else:
            records = []
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list or JSONL records")
    return [record for record in records if isinstance(record, Mapping)]


def samples_from_records(records: Sequence[Mapping[str, Any]], *, source: str) -> list[Sample]:
    samples = []
    for index, record in enumerate(records):
        sample = sample_from_record(record, source=f"{source}#{index}")
        if sample is not None:
            samples.append(sample)
    return samples


def sample_from_record(record: Mapping[str, Any], *, source: str) -> Sample | None:
    question = first_text(record, ("question", "query", "problem"))
    if question is None:
        return None

    answer_value = first_text(record, ("answer", "final_answer", "target", "label"))
    cot_value = first_value(
        record,
        ("cot", "steps", "rationale", "chain_of_thought", "solution", "response"),
    )
    cot_from_answer = None
    if answer_value is not None:
        cot_from_answer, answer_from_answer = split_gsm8k_answer(answer_value)
        if answer_from_answer is not None:
            answer_value = answer_from_answer
    if cot_value is None:
        cot_value = cot_from_answer

    cot_steps = split_cot_steps(cot_value)
    answer = clean_answer(answer_value)
    if not question.strip() or not cot_steps or not answer:
        return None
    return Sample(
        question=question.strip(),
        cot_steps=cot_steps,
        answer=answer,
        source=source,
    )


def first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    value = first_value(record, keys)
    if value is None:
        return None
    return str(value)


def split_gsm8k_answer(answer_text: str) -> tuple[str | None, str | None]:
    parts = str(answer_text).split("####", maxsplit=1)
    if len(parts) == 1:
        return None, None
    cot = parts[0].strip()
    answer = clean_answer(parts[1])
    return cot or None, answer or None


def split_cot_steps(value: Any | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        steps = [str(step).strip() for step in value]
    else:
        text = str(value).replace("\r\n", "\n").strip()
        line_steps = [line.strip() for line in text.split("\n") if line.strip()]
        if len(line_steps) > 1:
            steps = line_steps
        else:
            steps = [
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+", text)
                if part.strip()
            ]
    return [
        step
        for step in steps
        if step and not step.lstrip().startswith("####")
    ]


def clean_answer(value: Any | None) -> str:
    if value is None:
        return ""
    answer = str(value).strip()
    answer = answer.replace("####", "").strip()
    answer = re.sub(r"(?i)^(?:the\s+)?answer\s+is\s*:?\s*", "", answer).strip()
    answer = re.sub(r"(?i)^final\s+answer\s*:?\s*", "", answer).strip()
    return answer


def require_enough_samples(samples: list[Sample], source: object) -> list[Sample]:
    if not samples:
        raise RuntimeError(f"no usable GSM8K-style samples found in {source}")
    return samples


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_metrics_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def build_tokenizer_and_model(
    args: argparse.Namespace,
    samples: list[Sample],
) -> tuple[Any, nn.Module]:
    if args.model_id == "tiny":
        tokenizer = MinimalTokenizer()
        tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
        warm_tokenizer_vocab(tokenizer, samples, args)
        model = TinyCausalLM(vocab_size=len(tokenizer))
        return tokenizer, model

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for --model-id. Install "
            "lsp_jepa/requirements-mvp.txt or run with --model-id tiny."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def warm_tokenizer_vocab(
    tokenizer: MinimalTokenizer,
    samples: list[Sample],
    args: argparse.Namespace,
) -> None:
    build_student_batch(
        tokenizer,
        samples,
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        num_latent_steps=args.num_latent_steps,
        device=torch.device("cpu"),
    )
    build_teacher_batch(tokenizer, samples, args, torch.device("cpu"))


def build_student_batch(
    tokenizer: Any,
    samples: list[Sample],
    *,
    latent_id: int,
    start_id: int,
    end_id: int,
    num_latent_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = []
    label_rows = []
    first_latent_positions = []
    for sample in samples:
        question_ids = list(tokenizer.encode(sample.question + "\n", add_special_tokens=True))
        answer_ids = list(tokenizer.encode("### " + sample.answer, add_special_tokens=False))
        answer_ids.append(int(tokenizer.eos_token_id))
        input_ids = (
            question_ids
            + [start_id]
            + [latent_id] * num_latent_steps
            + [end_id]
            + answer_ids
        )
        labels = [-100] * (len(question_ids) + num_latent_steps + 2) + answer_ids
        rows.append(input_ids)
        label_rows.append(labels)
        first_latent_positions.append(input_ids.index(latent_id))

    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    aligned_rows = []
    aligned_labels = []
    aligned_attention = []
    aligned_positions = []
    target_first_latent = max(first_latent_positions)
    for input_ids, labels, first_latent in zip(
        rows,
        label_rows,
        first_latent_positions,
        strict=True,
    ):
        left_pad = target_first_latent - first_latent
        aligned_rows.append([pad_token_id] * left_pad + input_ids)
        aligned_labels.append([-100] * left_pad + labels)
        aligned_attention.append([0] * left_pad + [1] * len(input_ids))
        aligned_positions.append([0] * left_pad + list(range(len(input_ids))))

    rows = aligned_rows
    label_rows = aligned_labels
    max_len = max(len(row) for row in rows)
    padded_ids = []
    padded_labels = []
    attention = []
    positions = []
    for input_ids, labels, attn, pos in zip(
        rows,
        label_rows,
        aligned_attention,
        aligned_positions,
        strict=True,
    ):
        pad = max_len - len(input_ids)
        padded_ids.append(input_ids + [pad_token_id] * pad)
        padded_labels.append(labels + [-100] * pad)
        attention.append(attn + [0] * pad)
        positions.append(pos + [0] * pad)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
        "position_ids": torch.tensor(positions, dtype=torch.long, device=device),
    }


def build_teacher_batch(
    tokenizer: Any,
    samples: list[Sample],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    batch = build_teacher_inputs(
        tokenizer,
        [sample.question for sample in samples],
        [sample.cot_steps for sample in samples],
        answers=[sample.answer for sample in samples],
        max_length=args.max_teacher_length,
        padding=True,
        truncation=True,
        exclude_answer_tokens=not args.include_answer_tokens,
        exclude_answer_prefix=not args.include_answer_prefix,
        filter_answer_steps=args.reasoning_step_filter == "exclude_answer_only_steps",
    )
    tensor_keys = {"input_ids", "attention_mask", "step_boundaries", "step_starts", "step_mask"}
    for key in tensor_keys:
        batch[key] = batch[key].to(device)
    batch["teacher_input_ids"] = batch["input_ids"]
    batch["final_valid_step_position"] = final_valid_step_position(
        batch["step_boundaries"],
        batch["step_mask"],
    )
    return batch


def final_valid_step_position(
    step_boundaries: torch.Tensor,
    step_mask: torch.Tensor,
) -> torch.Tensor:
    if step_boundaries.ndim != 2 or step_mask.ndim != 2:
        raise ValueError("step_boundaries and step_mask must have shape [batch, steps]")
    if step_boundaries.shape != step_mask.shape:
        raise ValueError("step_boundaries and step_mask shape mismatch")
    mask = step_mask.to(device=step_boundaries.device, dtype=torch.bool)
    counts = mask.to(dtype=torch.long).sum(dim=1)
    last_step = (counts - 1).clamp_min(0)
    final_position = step_boundaries.gather(1, last_step.view(-1, 1)).squeeze(1)
    return torch.where(counts.gt(0), final_position, torch.zeros_like(final_position))


def validate_no_answer_leakage(
    samples: Sequence[Sample],
    teacher_inputs: Mapping[str, Any],
    *,
    include_answer_tokens: bool,
    include_answer_prefix: bool,
) -> bool:
    if include_answer_tokens or include_answer_prefix:
        raise RuntimeError("answer leakage check failed: answer targets were explicitly included")
    texts = teacher_inputs.get("texts", [])
    filtered_steps = teacher_inputs.get("filtered_steps", [])
    for sample, text, steps in zip(samples, texts, filtered_steps, strict=True):
        if "####" in str(text):
            raise RuntimeError(f"answer leakage check failed for {sample.source}: found ####")
        for step in steps:
            if looks_like_answer_prefix(str(step)):
                raise RuntimeError(
                    f"answer leakage check failed for {sample.source}: {step!r}"
                )
    return True


def looks_like_answer_prefix(text: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:^|\b)(?:the\s+)?answer\s+is\b|"
            r"(?:^|\b)final\s+answer\b|####",
            text,
        )
    )


def select_last_valid_target(targets: TeacherTargetBatch) -> TeacherTargetBatch:
    states, mask = select_last_valid_state(targets.target_states, targets.target_mask)
    indices = None
    if targets.step_indices is not None:
        indices, _ = select_last_valid_state(
            targets.step_indices.unsqueeze(-1),
            targets.target_mask,
        )
        indices = indices.squeeze(-1)
    return TeacherTargetBatch(
        target_states=states,
        target_mask=mask,
        step_indices=indices,
        debug={**targets.debug, "selection": "last_valid_reasoning_step"},
    )


def select_last_valid_state(
    states: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 3:
        raise ValueError("states must have shape [batch, T, dim]")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, T]")
    if states.shape[:2] != mask.shape:
        raise ValueError("states and mask shape mismatch")
    mask = mask.to(device=states.device, dtype=torch.bool)
    counts = mask.to(dtype=torch.long).sum(dim=1)
    last_indices = (counts - 1).clamp_min(0)
    gather_index = last_indices.view(-1, 1, 1).expand(-1, 1, states.shape[-1])
    selected = states.gather(dim=1, index=gather_index)
    selected_mask = counts.gt(0).view(-1, 1)
    selected = torch.where(selected_mask.unsqueeze(-1), selected, torch.zeros_like(selected))
    return selected, selected_mask


def validate_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).detach().cpu().item()):
        raise RuntimeError(f"{name} is not finite")


def grad_l1(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().cpu().item())
    return total


def clone_params(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
    }


def param_delta_l1(before: dict[str, torch.Tensor], module: nn.Module) -> float:
    total = 0.0
    params = dict(module.named_parameters())
    for name, previous in before.items():
        current = params[name].detach().to(device=previous.device, dtype=previous.dtype)
        total += float((current - previous).abs().sum().cpu().item())
    return total


def to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def tensor_to_int_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().tolist()]


def append_metrics(path: Path, metrics: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
