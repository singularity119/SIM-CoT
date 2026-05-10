"""PR11 full-data sequence-level LSP-JEPA-Core debug training on Coconut.

This script is intentionally small: one student forward, one EMA-teacher target
forward, one scalar sequence-level alignment loss, optional host answer CE, one
backward/optimizer step, one EMA update, JSONL metrics, and a compact summary.
It supports complete GSM8K-style training splits through a DataLoader for
PR11-MVP runs without adding step-level objectives, adapter expansion,
distributed training, wandb, or nonlocal artifact writes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
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
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is expected in the training env
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[2]
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
from lsp_jepa.core.anti_collapse import compute_anti_collapse_loss  # noqa: E402
from lsp_jepa.core.ema_teacher import EMATeacher  # noqa: E402
from lsp_jepa.core.latent_interface import TeacherTargetBatch  # noqa: E402
from lsp_jepa.core.losses import compute_lsp_state_loss  # noqa: E402
from lsp_jepa.core.metrics import (  # noqa: E402
    effective_rank,
    pairwise_cosine,
    pairwise_l2,
    per_dim_variance,
)
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
    sample_id: str = ""


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


class SampleDataset(Dataset[tuple[int, Sample]]):
    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[int, Sample]:
        return index, self.samples[index]


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int


def main() -> None:
    args = parse_args()
    if args.objective not in {"sequence", "step_trajectory"}:
        raise ValueError("--objective must be sequence or step_trajectory")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    if args.num_latent_steps < 1:
        raise ValueError("--num-latent-steps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.min_samples < 1:
        raise ValueError("--min-samples must be at least 1")
    if args.effective_rank_every < 0:
        raise ValueError("--effective-rank-every must be non-negative")

    distributed = init_distributed_context(args.device)
    torch.manual_seed(args.seed + distributed.rank)
    device = resolve_device(args.device)
    if distributed.enabled:
        if device.type != "cuda":
            cleanup_distributed(distributed)
            raise RuntimeError("distributed training requires --device cuda or --device auto with CUDA")
        torch.cuda.set_device(distributed.local_rank)
        device = torch.device("cuda", distributed.local_rank)
    normalize_output_paths(args)
    configure_data_cache(args)
    samples = load_samples(args)
    if len(samples) < args.min_samples:
        raise RuntimeError(
            f"need at least min_samples={args.min_samples} usable samples, "
            f"found {len(samples)}"
        )
    if len(samples) < args.batch_size:
        raise RuntimeError(
            f"need at least batch_size={args.batch_size} samples, found {len(samples)}"
        )
    args.distributed = distributed.enabled
    args.distributed_rank = distributed.rank
    args.distributed_local_rank = distributed.local_rank
    args.distributed_world_size = distributed.world_size
    args.local_batch_size = args.batch_size
    args.global_batch_size = args.batch_size * distributed.world_size
    args.epoch_steps = steps_per_epoch(len(samples), args.global_batch_size, args.drop_last)
    if args.save_every_epoch:
        args.save_every = args.epoch_steps
    if args.eval_every_epoch:
        args.eval_every = args.epoch_steps
    train_loader = build_train_dataloader(samples, args)

    tokenizer, base_model = build_tokenizer_and_model(args, samples)
    base_model.to(device)
    epoch_eval_samples = load_epoch_eval_samples(args) if args.eval_every > 0 else []
    if isinstance(tokenizer, MinimalTokenizer) and epoch_eval_samples:
        warm_tokenizer_vocab(tokenizer, list(epoch_eval_samples), args)
        base_model.resize_token_embeddings(len(tokenizer))

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
    student_core = student
    student_core.train()
    if distributed.enabled:
        student = DistributedDataParallel(
            student_core,
            device_ids=[distributed.local_rank],
            output_device=distributed.local_rank,
        )

    ema_teacher = EMATeacher.from_student(
        student_core.base_causallm,
        decay=args.ema_decay,
        trainable_only=True,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        student_core.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.resume_from_checkpoint is not None:
        args.append_metrics = True
    metrics_path = resolve_metrics_path(args.metrics_path)
    summary_path = resolve_metrics_path(args.summary_path)
    config_snapshot_path = resolve_metrics_path(args.config_snapshot_path)
    checkpoint_dir = resolve_metrics_path(args.checkpoint_dir)
    eval_output_dir = resolve_metrics_path(args.eval_output_dir) if args.eval_output_dir else summary_path.parent / "evals"
    eval_metrics_path = (
        resolve_metrics_path(args.eval_metrics_path)
        if args.eval_metrics_path
        else eval_output_dir / "metrics.jsonl"
    )
    if is_main_process(distributed):
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.eval_every > 0:
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            if eval_metrics_path.exists() and not args.append_metrics:
                eval_metrics_path.unlink()
        if metrics_path.exists() and not args.append_metrics:
            metrics_path.unlink()
        write_config_snapshot(
            config_snapshot_path,
            build_config_snapshot(
                args,
                device=device,
                sample_count=len(samples),
                metrics_path=metrics_path,
                summary_path=summary_path,
                config_snapshot_path=config_snapshot_path,
            ),
        )
    distributed_barrier(distributed)

    start_step = 0
    if args.resume_from_checkpoint is not None:
        start_step = load_training_checkpoint(
            args.resume_from_checkpoint,
            student=student_core,
            ema_teacher=ema_teacher,
            optimizer=optimizer,
            device=device,
        )
        if start_step >= args.max_steps:
            if is_main_process(distributed):
                print(
                    json.dumps(
                        {
                            "resume_from_checkpoint": str(args.resume_from_checkpoint),
                            "checkpoint_step": start_step,
                            "max_steps": args.max_steps,
                            "status": "already_complete",
                        },
                        sort_keys=True,
                    )
                )
            distributed_barrier(distributed)
            cleanup_distributed(distributed)
            return
    batch_iter = iter_train_batches(
        train_loader,
        start_epoch=start_step // args.epoch_steps,
    )
    for _ in range(start_step % args.epoch_steps):
        next(batch_iter)

    adapter = CoconutLSPAdapter(latent_token_id=latent_id)
    metrics_history: list[dict[str, Any]] = []
    seen_sample_indices: set[int] = set()
    best_total_loss: float | None = None
    best_checkpoint_path: Path | None = None
    progress = build_tqdm_progress(args, initial=start_step) if is_main_process(distributed) else None

    for train_step in range(start_step + 1, args.max_steps + 1):
        batch_indices, batch_samples = next(batch_iter)
        seen_sample_indices.update(batch_indices)
        teacher_inputs = build_teacher_batch(tokenizer, batch_samples, args, device)
        answer_leakage_ok = validate_no_answer_leakage(
            batch_samples,
            teacher_inputs,
            include_answer_tokens=args.include_answer_tokens,
            include_answer_prefix=args.include_answer_prefix,
        )

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
        latent_counts = latent_counts_for_objective(teacher_targets, args)
        student_batch = build_student_batch(
            tokenizer,
            batch_samples,
            latent_id=latent_id,
            start_id=start_id,
            end_id=end_id,
            num_latent_steps=args.num_latent_steps,
            latent_counts=latent_counts,
            device=device,
        )

        host_output = adapter.forward_student(
            student,
            {"student_inputs": student_batch},
            output_latent_states=True,
        )
        latent_mask_nonempty = bool(host_output.latent_mask.any().detach().cpu().item())
        if not latent_mask_nonempty:
            raise RuntimeError("student latent mask is empty")
        aligned_student, aligned_teacher, alignment_mask = align_student_teacher_states(
            host_output.latent_states,
            host_output.latent_mask,
            teacher_targets.target_states,
            teacher_targets.target_mask,
            objective=args.objective,
            alignment=args.alignment_loss,
        )

        if args.objective == "step_trajectory":
            lsp_loss = compute_step_trajectory_state_loss(
                aligned_student,
                aligned_teacher,
                alignment_mask,
                alignment=args.alignment_loss,
            )
        else:
            lsp_loss = compute_lsp_state_loss(
                aligned_student,
                aligned_teacher,
                alignment_mask,
                alignment=args.alignment_loss,
            )
        host_answer_ce = host_output.host_losses.get("host_answer_ce")
        if host_answer_ce is None:
            raise RuntimeError("Coconut host output did not provide host_answer_ce")
        anti_collapse_loss = compute_anti_collapse_loss(
            aligned_student,
            alignment_mask,
            method=args.anti_collapse_type,
            weight=args.anti_collapse_weight,
        )
        total_loss = (
            args.lsp_weight * lsp_loss
            + args.host_answer_ce_weight * host_answer_ce
            + anti_collapse_loss
        )
        latent_variance = per_dim_variance(aligned_student, alignment_mask)
        raw_latent_variance = per_dim_variance(host_output.latent_states, host_output.latent_mask)
        pairwise = pairwise_cosine_summary(aligned_student, alignment_mask)
        pairwise_l2_mean = pairwise_l2(aligned_student, alignment_mask)
        (
            latent_effective_rank,
            latent_effective_rank_status,
            latent_effective_rank_error,
        ) = effective_rank_diagnostic(
            aligned_student,
            alignment_mask,
            compute=should_compute_effective_rank(args, train_step),
        )
        answer_ce_terms = answer_ce_terms_in_total(args, host_answer_ce)
        answer_ce_double_count_ok = len(answer_ce_terms) <= 1

        validate_finite("total_loss", total_loss)
        validate_finite("lsp_loss", lsp_loss)
        validate_finite("host_answer_ce", host_answer_ce)
        validate_finite("anti_collapse_loss", anti_collapse_loss)
        validate_finite("latent_variance_mean", latent_variance["mean"])
        validate_finite("pairwise_cosine_mean", pairwise["mean"])
        validate_finite("pairwise_l2_mean", pairwise_l2_mean)
        if not bool(alignment_mask.any().detach().cpu().item()):
            raise RuntimeError("no valid LSP alignment targets were found")
        if not answer_ce_double_count_ok:
            raise RuntimeError("answer CE would be counted more than once in total loss")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        student_base_grad_l1 = grad_l1(student_core)
        student_grad_l1 = student_base_grad_l1
        if student_grad_l1 <= 0.0 or not math.isfinite(student_grad_l1):
            raise RuntimeError("student parameters did not receive finite gradients")
        if student_base_grad_l1 <= 0.0 or not math.isfinite(student_base_grad_l1):
            raise RuntimeError("student backbone parameters did not receive finite gradients")
        if any(param.grad is not None for param in ema_teacher.parameters()):
            raise RuntimeError("EMA teacher parameters received gradients")

        teacher_before = clone_params(ema_teacher.teacher_model)
        optimizer.step()
        ema_teacher.update(student_core.base_causallm)
        teacher_delta_l1 = param_delta_l1(teacher_before, ema_teacher.teacher_model)
        if teacher_delta_l1 <= 0.0 or not math.isfinite(teacher_delta_l1):
            raise RuntimeError("EMA update did not change teacher parameters")
        checkpoint_path = None
        if is_main_process(distributed):
            checkpoint_path = maybe_save_checkpoint(
                args,
                step=train_step,
                checkpoint_dir=checkpoint_dir,
                student=student_core,
                ema_teacher=ema_teacher,
                optimizer=optimizer,
            )
        if checkpoint_path is not None:
            total_loss_value = to_float(total_loss)
            if args.keep_best_total_loss and (
                best_total_loss is None or total_loss_value < best_total_loss
            ):
                best_total_loss = total_loss_value
                best_checkpoint_path = checkpoint_path
                write_json(
                    checkpoint_dir / "best_total_loss.json",
                    {
                        "step": train_step,
                        "total_loss": best_total_loss,
                        "checkpoint_path": str(best_checkpoint_path),
                    },
                )
            prune_old_checkpoints(
                checkpoint_dir,
                keep_last=args.keep_last_checkpoints,
                keep_paths=[best_checkpoint_path] if best_checkpoint_path is not None else [],
            )
        epoch_eval = None
        if is_main_process(distributed):
            epoch_eval = maybe_run_epoch_eval(
                args,
                step=train_step,
                student=student_core,
                tokenizer=tokenizer,
                eval_samples=epoch_eval_samples,
                checkpoint_path=checkpoint_path,
                eval_output_dir=eval_output_dir,
                eval_metrics_path=eval_metrics_path,
                device=device,
            )
        distributed_barrier(distributed)

        if not is_main_process(distributed):
            continue

        metrics = {
            "step": train_step,
            "max_steps": args.max_steps,
            "experiment_mode": "core",
            "backbone": "coconut",
            "objective": "lsp_state",
            "training_objective": args.objective,
            "mapping": "one_to_one" if args.objective == "step_trajectory" else "sequence",
            "target_position": (
                "all_valid_reasoning_step_boundaries"
                if args.objective == "step_trajectory"
                else "final_valid_reasoning_step"
            ),
            "no_step_level_training": args.objective != "step_trajectory",
            "step_trajectory_training": args.objective == "step_trajectory",
            "no_adapter_expansion": True,
            "sample_count": len(samples),
            "expected_samples": args.expected_samples,
            "full_train_dataloader": True,
            "dataloader_drop_last": args.drop_last,
            "data_cache_dir": str(resolve_repo_path(Path(args.data_cache_dir))),
            "dataset_local_path": (
                str(resolve_repo_path(args.dataset_local_path))
                if args.dataset_local_path is not None
                else None
            ),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "checkpoint_keep_last": args.keep_last_checkpoints,
            "best_total_loss": best_total_loss,
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
            "epoch_steps": args.epoch_steps,
            "epoch_index": epoch_index(train_step, args.epoch_steps),
            "unique_samples_seen": len(seen_sample_indices),
            "batch_size": args.global_batch_size,
            "local_batch_size": args.local_batch_size,
            "distributed": distributed.enabled,
            "distributed_world_size": distributed.world_size,
            "distributed_rank": distributed.rank,
            "batch_indices": batch_indices,
            "data_source": args.dataset_source if args.data_path is None else str(args.data_path),
            "batch_sample_sources": [sample.source for sample in batch_samples],
            "batch_sample_ids": [sample.sample_id or sample.source for sample in batch_samples],
            "total_loss": to_float(total_loss),
            "lsp_loss": to_float(lsp_loss),
            "host_answer_ce": to_float(host_answer_ce),
            "host_answer_ce_weight": args.host_answer_ce_weight,
            "lsp_weight": args.lsp_weight,
            "anti_collapse_type": args.anti_collapse_type,
            "anti_collapse_weight": args.anti_collapse_weight,
            "anti_collapse_loss": to_float(anti_collapse_loss),
            "answer_ce_terms_in_total": answer_ce_terms,
            "answer_ce_double_count_ok": answer_ce_double_count_ok,
            "latent_variance": to_float(latent_variance["mean"]),
            "latent_variance_mean": to_float(latent_variance["mean"]),
            "latent_variance_min": to_float(latent_variance["min"]),
            "latent_variance_max": to_float(latent_variance["max"]),
            "raw_latent_variance_mean": to_float(raw_latent_variance["mean"]),
            "effective_rank": (
                to_float(latent_effective_rank)
                if latent_effective_rank is not None
                else None
            ),
            "effective_rank_status": latent_effective_rank_status,
            "effective_rank_error": latent_effective_rank_error,
            "effective_rank_every": args.effective_rank_every,
            "pairwise_cosine": to_float(pairwise["mean"]),
            "pairwise_cosine_mean": to_float(pairwise["mean"]),
            "pairwise_cosine_min": to_float(pairwise["min"]),
            "pairwise_cosine_max": to_float(pairwise["max"]),
            "pairwise_cosine_all_one": pairwise["all_one"],
            "pairwise_cosine_near_one": bool(to_float(pairwise["mean"]) >= args.pairwise_cosine_fail_threshold),
            "pairwise_l2_mean": to_float(pairwise_l2_mean),
            "student_grad_l1": student_grad_l1,
            "student_base_grad_l1": student_base_grad_l1,
            "teacher_grad_params": sum(
                param.grad is not None for param in ema_teacher.parameters()
            ),
            "teacher_delta_l1": teacher_delta_l1,
            "ema_drift_l1": teacher_delta_l1,
            "teacher_target_mask_nonempty": teacher_target_mask_nonempty,
            "latent_mask_nonempty": latent_mask_nonempty,
            "valid_teacher_targets": int(
                teacher_targets.target_mask.sum().detach().cpu().item()
            ),
            "valid_student_latents": int(
                host_output.latent_mask.sum().detach().cpu().item()
            ),
            "valid_alignment_targets": int(alignment_mask.sum().detach().cpu().item()),
            "valid_final_targets": int(alignment_mask.sum().detach().cpu().item()),
            "student_latent_steps_max": int(host_output.latent_mask.shape[1]),
            "teacher_steps_max": int(teacher_targets.target_mask.shape[1]),
            "student_latent_counts": tensor_to_int_list(host_output.latent_mask.sum(dim=1)),
            "teacher_target_counts": tensor_to_int_list(teacher_targets.target_mask.sum(dim=1)),
            "teacher_input_ids_shape": list(teacher_inputs["teacher_input_ids"].shape),
            "final_valid_step_position": tensor_to_int_list(
                teacher_inputs["final_valid_step_position"]
            ),
            "answer_leakage_ok": answer_leakage_ok,
            "teacher_exclude_answer_tokens": not args.include_answer_tokens,
            "teacher_exclude_answer_prefix": not args.include_answer_prefix,
        }
        if epoch_eval is not None:
            metrics["epoch_eval"] = {
                "metrics_path": epoch_eval["metrics_path"],
                "step_metrics_path": epoch_eval["step_metrics_path"],
                "accuracy": epoch_eval["accuracy"],
                "exact_match": epoch_eval["exact_match"],
                "invalid_answer_rate": epoch_eval["invalid_answer_rate"],
                "num_eval_samples": epoch_eval["num_eval_samples"],
            }
        append_metrics(metrics_path, metrics)
        metrics_history.append(metrics)
        update_tqdm_progress(progress, metrics)
        if should_log_step(args, train_step):
            tqdm_log_line = build_tqdm_log_line(progress, metrics)
            if tqdm_log_line is not None:
                progress_write(progress, tqdm_log_line)
            if should_log_json(args, train_step):
                progress_write(progress, json.dumps(metrics, sort_keys=True))

    if is_main_process(distributed):
        close_tqdm_progress(progress)
        summary = build_summary(
            metrics_history,
            args=args,
            sample_count=len(samples),
            unique_samples_seen=len(seen_sample_indices),
            metrics_path=metrics_path,
            summary_path=summary_path,
            config_snapshot_path=config_snapshot_path,
        )
        write_json(summary_path, summary)
        print(
            json.dumps(
                {
                    "metrics_path": str(metrics_path),
                    "summary_path": str(summary_path),
                    "config_snapshot_path": str(config_snapshot_path),
                    "completed_steps": args.max_steps,
                    "unique_samples_seen": len(seen_sample_indices),
                    "acceptance_passed": summary["acceptance"]["passed"],
                },
                sort_keys=True,
            )
        )
    distributed_barrier(distributed)
    cleanup_distributed(distributed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("lsp_jepa/configs/core/lsp_seq_full_debug.yaml"),
        help="YAML config path. CLI flags override config values.",
    )
    parser.add_argument("--model-id", default="tiny", help="'tiny' or a local/HF model id")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--objective", default="sequence")
    parser.add_argument("--lsp-weight", type=float, default=1.0)
    parser.add_argument("--host-answer-ce-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--target-layer", default="last_2")
    parser.add_argument("--num-latent-steps", type=int, default=4)
    parser.add_argument("--max-teacher-length", type=int, default=512)
    parser.add_argument("--alignment-loss", default="normalized_mse")
    parser.add_argument("--anti-collapse-type", default="variance")
    parser.add_argument("--anti-collapse-weight", type=float, default=0.01)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--config-snapshot-path",
        default=None,
    )
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Load student, EMA teacher, and optimizer state from a step_*.pt checkpoint.",
    )
    parser.add_argument("--autodl-root", default="/root/autodl-tmp")
    parser.add_argument("--data-cache-dir", default=None)
    parser.add_argument("--dataset-local-path", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument(
        "--dataset-source",
        default="gsm8k",
        choices=("gsm8k_smoke", "gsm8k", "gsm8k_aug", "synthetic"),
    )
    parser.add_argument("--dataset-id", default="openai/gsm8k")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--expected-samples", type=int, default=7473)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="0 means use the complete configured training split.",
    )
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--loss-stability-ratio", type=float, default=10.0)
    parser.add_argument("--collapse-eps", type=float, default=1e-10)
    parser.add_argument("--pairwise-cosine-fail-threshold", type=float, default=0.999)
    parser.add_argument(
        "--effective-rank-every",
        type=int,
        default=0,
        help=(
            "Compute effective-rank diagnostics every N steps; "
            "0 skips online computation to avoid eigensolver failures."
        ),
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override --save-every with the computed steps per full training epoch.",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=0,
        help="0 keeps all checkpoints; N keeps only the latest N step_*.pt files.",
    )
    parser.add_argument(
        "--keep-best-total-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the epoch checkpoint with the lowest observed total_loss.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run final-answer eval every N training steps; 0 disables training-time eval.",
    )
    parser.add_argument(
        "--eval-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override --eval-every with the computed steps per full training epoch.",
    )
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--eval-metrics-path", default=None)
    parser.add_argument("--eval-json", type=Path)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-dataset-id", default="openai/gsm8k")
    parser.add_argument("--eval-dataset-config", default="main")
    parser.add_argument("--eval-expected-samples", type=int, default=1319)
    parser.add_argument("--eval-limit-samples", type=int, default=20)
    parser.add_argument("--eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-save-examples", type=int, default=5)
    parser.add_argument("--tqdm-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--console-json-every",
        type=int,
        default=0,
        help="Print full metrics JSON to stdout every N steps; 0 keeps JSON only in metrics.jsonl.",
    )
    parser.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-answer-tokens", action="store_true")
    parser.add_argument("--include-answer-prefix", action="store_true")
    parser.add_argument(
        "--reasoning-step-filter",
        default="exclude_answer_only_steps",
        choices=("exclude_answer_only_steps", "none"),
    )
    args = parser.parse_args()
    provided_flags = provided_cli_flags(sys.argv[1:])
    if args.config is not None:
        config_path = resolve_repo_path(args.config)
        if config_path.exists():
            apply_config(args, load_yaml_config(config_path), provided_flags)
    return args


def provided_cli_flags(argv: Sequence[str]) -> set[str]:
    flags = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token.split("=", maxsplit=1)[0][2:]
        if name.startswith("no-"):
            name = name[3:]
        flags.add(name.replace("-", "_"))
    return flags


def load_yaml_config(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config; install lsp_jepa/requirements-mvp.txt"
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def apply_config(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    provided_flags: set[str],
) -> None:
    set_from_config(args, provided_flags, "model_id", config_get(config, "runtime.model_id"))
    set_from_config(args, provided_flags, "device", config_get(config, "runtime.device"))
    set_from_config(args, provided_flags, "seed", config_get(config, "runtime.seed"))
    set_from_config(args, provided_flags, "max_steps", config_get(config, "training.max_steps"))
    set_from_config(args, provided_flags, "batch_size", config_get(config, "training.batch_size"))
    set_from_config(args, provided_flags, "save_every", config_get(config, "training.save_every"))
    set_from_config(args, provided_flags, "save_every_epoch", config_get(config, "training.save_every_epoch"))
    set_from_config(args, provided_flags, "keep_last_checkpoints", config_get(config, "training.keep_last_checkpoints"))
    set_from_config(args, provided_flags, "keep_best_total_loss", config_get(config, "training.keep_best_total_loss"))
    set_from_config(args, provided_flags, "tqdm_progress", config_get(config, "training.tqdm_progress"))
    set_from_config(args, provided_flags, "console_json_every", config_get(config, "logging.console_json_every"))
    set_from_config(args, provided_flags, "effective_rank_every", config_get(config, "logging.effective_rank_every"))
    set_from_config(args, provided_flags, "effective_rank_every", config_get(config, "training.effective_rank_every"))
    set_from_config(args, provided_flags, "eval_every", config_get(config, "eval.every_steps"))
    set_from_config(args, provided_flags, "eval_every_epoch", config_get(config, "eval.every_epoch"))
    set_from_config(args, provided_flags, "eval_output_dir", config_get(config, "eval.output_dir"))
    set_from_config(args, provided_flags, "eval_metrics_path", config_get(config, "eval.metrics_path"))
    set_from_config(args, provided_flags, "eval_json", config_get(config, "eval.eval_json"), path=True)
    set_from_config(args, provided_flags, "eval_split", config_get(config, "eval.eval_split"))
    set_from_config(args, provided_flags, "eval_dataset_id", config_get(config, "eval.dataset_id"))
    set_from_config(args, provided_flags, "eval_dataset_config", config_get(config, "eval.dataset_config"))
    set_from_config(args, provided_flags, "eval_expected_samples", config_get(config, "eval.expected_samples"))
    set_from_config(args, provided_flags, "eval_limit_samples", config_get(config, "eval.limit_samples"))
    set_from_config(args, provided_flags, "eval_max_new_tokens", config_get(config, "eval.max_new_tokens"))
    set_from_config(args, provided_flags, "eval_save_examples", config_get(config, "eval.save_examples"))
    set_from_config(args, provided_flags, "log_every", config_get(config, "training.log_every"))
    set_from_config(args, provided_flags, "lr", config_get(config, "training.lr"))
    set_from_config(args, provided_flags, "weight_decay", config_get(config, "training.weight_decay"))
    set_from_config(args, provided_flags, "drop_last", config_get(config, "training.drop_last"))
    set_from_config(args, provided_flags, "shuffle", config_get(config, "training.shuffle"))
    set_from_config(args, provided_flags, "output_dir", config_get(config, "training.output_dir"))
    set_from_config(args, provided_flags, "metrics_path", config_get(config, "training.metrics_path"))
    set_from_config(args, provided_flags, "summary_path", config_get(config, "training.summary_path"))
    set_from_config(args, provided_flags, "checkpoint_dir", config_get(config, "training.checkpoint_dir"))
    set_from_config(args, provided_flags, "save_checkpoints", config_get(config, "training.save_checkpoints"))
    set_from_config(
        args,
        provided_flags,
        "config_snapshot_path",
        config_get(config, "training.config_snapshot_path"),
    )
    set_from_config(args, provided_flags, "data_path", config_get(config, "data.data_path"), path=True)
    set_from_config(args, provided_flags, "autodl_root", config_get(config, "data.autodl_root"))
    set_from_config(args, provided_flags, "data_cache_dir", config_get(config, "data.cache_dir"))
    set_from_config(
        args,
        provided_flags,
        "dataset_local_path",
        config_get(config, "data.local_dataset_path"),
        path=True,
    )
    set_from_config(args, provided_flags, "dataset_source", config_get(config, "data.dataset_source"))
    set_from_config(args, provided_flags, "dataset_id", config_get(config, "data.dataset_id"))
    set_from_config(args, provided_flags, "dataset_config", config_get(config, "data.dataset_config"))
    set_from_config(args, provided_flags, "dataset_split", config_get(config, "data.dataset_split"))
    set_from_config(args, provided_flags, "hf_endpoint", config_get(config, "data.hf_endpoint"))
    set_from_config(args, provided_flags, "expected_samples", config_get(config, "data.expected_samples"))
    set_from_config(args, provided_flags, "num_samples", config_get(config, "data.max_samples"))
    set_from_config(args, provided_flags, "min_samples", config_get(config, "data.min_samples"))
    set_from_config(args, provided_flags, "ema_decay", config_get(config, "teacher.ema_decay"))
    set_from_config(args, provided_flags, "target_layer", config_get(config, "teacher.target_layer"))
    set_from_config(
        args,
        provided_flags,
        "reasoning_step_filter",
        config_get(config, "teacher.reasoning_step_filter"),
    )
    exclude_answer_tokens = config_get(config, "teacher.exclude_answer_tokens")
    if "include_answer_tokens" not in provided_flags and exclude_answer_tokens is not None:
        args.include_answer_tokens = not bool(exclude_answer_tokens)
    exclude_answer_prefix = config_get(config, "teacher.exclude_answer_prefix")
    if "include_answer_prefix" not in provided_flags and exclude_answer_prefix is not None:
        args.include_answer_prefix = not bool(exclude_answer_prefix)
    set_from_config(args, provided_flags, "num_latent_steps", config_get(config, "student.num_latent_steps"))
    objective_type = config_get(config, "lsp_objective.type")
    mapping_strategy = config_get(config, "mapping.strategy")
    if "objective" not in provided_flags:
        if objective_type == "step_trajectory" or mapping_strategy == "one_to_one":
            args.objective = "step_trajectory"
        elif objective_type == "state":
            args.objective = "sequence"
    set_from_config(args, provided_flags, "alignment_loss", config_get(config, "loss.alignment"))
    set_from_config(args, provided_flags, "lsp_weight", config_get(config, "loss.align_weight"))
    anti_collapse = config_get(config, "loss.anti_collapse")
    if anti_collapse is None:
        anti_collapse = config_get(config, "loss.anti_collapse_type")
    set_from_config(args, provided_flags, "anti_collapse_type", anti_collapse)
    set_from_config(
        args,
        provided_flags,
        "anti_collapse_weight",
        config_get(config, "loss.anti_collapse_weight"),
    )
    set_from_config(
        args,
        provided_flags,
        "host_answer_ce_weight",
        config_get(config, "host_losses.host_answer_ce_weight"),
    )
    set_from_config(args, provided_flags, "max_teacher_length", config_get(config, "teacher.max_length"))
    set_from_config(args, provided_flags, "max_teacher_length", config_get(config, "data.max_teacher_length"))


def set_from_config(
    args: argparse.Namespace,
    provided_flags: set[str],
    attr: str,
    value: Any,
    *,
    path: bool = False,
) -> None:
    if value is None or attr in provided_flags:
        return
    if path:
        value = Path(value)
    setattr(args, attr, value)


def config_get(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def init_distributed_context(requested_device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return DistributedContext(False, rank=0, local_rank=0, world_size=1)
    if requested_device == "cpu":
        raise RuntimeError("distributed training requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedContext(True, rank=rank, local_rank=local_rank, world_size=world_size)


def is_main_process(distributed: DistributedContext) -> bool:
    return not distributed.enabled or distributed.rank == 0


def distributed_barrier(distributed: DistributedContext) -> None:
    if distributed.enabled:
        dist.barrier()


def cleanup_distributed(distributed: DistributedContext) -> None:
    if distributed.enabled and dist.is_initialized():
        dist.destroy_process_group()


def default_samples(limit: int = 2) -> list[Sample]:
    """Generate deterministic GSM8K-style samples for offline debug training."""

    samples = [
        Sample(
            question="What is two plus three?",
            cot_steps=[
                "Represent two as 2.",
                "Add three more to make five.",
                "#### 5",
            ],
            answer="5",
            source="synthetic:0",
        ),
        Sample(
            question="A box has four red pens and three blue pens. How many pens?",
            cot_steps=[
                "Start with four red pens.",
                "Add three blue pens for a total of seven pens.",
                "The answer is 7",
            ],
            answer="7",
            source="synthetic:1",
        ),
    ]
    for index in range(len(samples), max(limit, len(samples))):
        samples.append(synthetic_arithmetic_sample(index))
    return samples[:limit]


def synthetic_arithmetic_sample(index: int) -> Sample:
    pattern = index % 4
    a = 2 + (index * 3) % 37
    b = 3 + (index * 5) % 29
    c = 1 + (index * 7) % 17
    if pattern == 0:
        answer = a + b + c
        return Sample(
            question=(
                f"A shelf has {a} red books, {b} blue books, and {c} green books. "
                "How many books are on the shelf?"
            ),
            cot_steps=[
                f"Start with {a} red books.",
                f"Add {b} blue books to get {a + b} books.",
                f"Add {c} green books to get {answer} books.",
            ],
            answer=str(answer),
            source=f"synthetic:{index}",
        )
    if pattern == 1:
        answer = a * b
        return Sample(
            question=f"There are {a} boxes with {b} markers in each box. How many markers?",
            cot_steps=[
                f"Each box has {b} markers.",
                f"With {a} boxes, multiply {a} by {b}.",
                f"The product is {answer} markers.",
            ],
            answer=str(answer),
            source=f"synthetic:{index}",
        )
    if pattern == 2:
        start = a + b + c
        answer = start - b
        return Sample(
            question=(
                f"Mia collected {start} stickers and gave {b} stickers away. "
                "How many stickers remain?"
            ),
            cot_steps=[
                f"Mia starts with {start} stickers.",
                f"She gives away {b} stickers.",
                f"Subtracting leaves {answer} stickers.",
            ],
            answer=str(answer),
            source=f"synthetic:{index}",
        )
    unit = 2 + (index % 9)
    answer = a * unit + c
    return Sample(
        question=(
            f"Noah buys {a} packs with {unit} pencils each and then finds {c} more. "
            "How many pencils does Noah have?"
        ),
        cot_steps=[
            f"The packs contain {a} times {unit} pencils.",
            f"That gives {a * unit} pencils from packs.",
            f"Adding {c} more gives {answer} pencils.",
        ],
        answer=str(answer),
        source=f"synthetic:{index}",
    )


def load_samples(args: argparse.Namespace) -> list[Sample]:
    limit = sample_limit(args)
    if args.data_path is not None:
        return load_samples_from_path(args.data_path, limit=limit)
    if args.dataset_local_path is not None:
        local_path = resolve_repo_path(args.dataset_local_path)
        if local_path.exists():
            return load_samples_from_path(local_path, limit=limit)
    if args.dataset_source == "synthetic":
        return default_samples(limit or args.expected_samples)
    if args.dataset_source == "gsm8k_smoke":
        return load_samples_from_path(
            REPO_ROOT / "lsp_jepa" / "data" / "gsm8k_smoke.jsonl",
            limit=limit,
        )
    dataset_id = args.dataset_id if args.dataset_source == "gsm8k" else "zen-E/GSM8k-Aug"
    config = args.dataset_config if args.dataset_source == "gsm8k" else None
    samples = load_hf_dataset_samples(
        dataset_id=dataset_id,
        config=config,
        split=args.dataset_split,
        limit=limit,
        expected_samples=args.expected_samples,
        hf_endpoint=args.hf_endpoint,
    )
    if args.dataset_local_path is not None:
        save_samples_jsonl(samples, resolve_repo_path(args.dataset_local_path))
    return samples


def sample_limit(args: argparse.Namespace) -> int | None:
    return None if args.num_samples is None or args.num_samples <= 0 else int(args.num_samples)


def load_samples_from_path(path: Path, *, limit: int | None) -> list[Sample]:
    path = resolve_repo_path(path)
    records = read_records(path)
    samples = samples_from_records(records, source=str(path))
    if limit is not None:
        samples = samples[:limit]
    return require_enough_samples(samples, path)


def save_samples_jsonl(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            handle.write(
                json.dumps(
                    {
                        "id": sample.sample_id or f"sample_{index}",
                        "question": sample.question,
                        "steps": sample.cot_steps,
                        "answer": sample.answer,
                        "source": sample.source,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def load_hf_dataset_samples(
    *,
    dataset_id: str,
    config: str | None,
    split: str,
    limit: int | None,
    expected_samples: int,
    hf_endpoint: str,
) -> list[Sample]:
    try:
        from datasets import load_dataset
    except ImportError:
        rows = load_hf_rows_via_dataset_server(
            dataset_id=dataset_id,
            config=config,
            split=split,
            limit=limit,
            expected_samples=expected_samples,
        )
    else:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        if config is None:
            dataset = load_dataset(dataset_id, split=split)
        else:
            dataset = load_dataset(dataset_id, config, split=split)
        row_count = len(dataset) if limit is None else min(limit, len(dataset))
        rows = [dataset[idx] for idx in range(row_count)]
    samples = samples_from_records(rows, source=f"hf:{dataset_id}:{split}")
    if limit is not None:
        samples = samples[:limit]
    return require_enough_samples(samples, dataset_id)


def load_hf_rows_via_dataset_server(
    *,
    dataset_id: str,
    config: str | None,
    split: str,
    limit: int | None,
    expected_samples: int,
    page_size: int = 100,
) -> list[Mapping[str, Any]]:
    rows = []
    target = expected_samples if limit is None else limit
    offset = 0
    while offset < target:
        length = min(page_size, target - offset)
        query: dict[str, str] = {
            "dataset": dataset_id,
            "split": split,
            "offset": str(offset),
            "length": str(length),
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
        page = []
        for item in payload["rows"]:
            row = item.get("row", item)
            if isinstance(row, Mapping):
                page.append(row)
        if not page:
            break
        rows.extend(page)
        if len(page) < length:
            break
        offset += len(page)
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
        sample_id=str(record.get("id") or record.get("sample_id") or source),
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


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def normalize_output_paths(args: argparse.Namespace) -> None:
    autodl_root = Path(args.autodl_root)
    if args.data_cache_dir is None:
        args.data_cache_dir = str(autodl_root / "lsp_jepa" / "data")
    if args.dataset_local_path is None and args.dataset_source == "gsm8k" and args.data_path is None:
        args.dataset_local_path = Path(args.data_cache_dir) / "gsm8k_train.jsonl"
    if args.output_dir is None:
        args.output_dir = str(
            autodl_root
            / "lsp_jepa"
            / "runs"
            / "pr11_mvp"
            / "full_seq_debug"
            / f"max_steps_{args.max_steps}"
        )
    output_dir = resolve_repo_path(Path(args.output_dir))
    if args.metrics_path is None:
        args.metrics_path = str(output_dir / "metrics.jsonl")
    if args.summary_path is None:
        args.summary_path = str(output_dir / "summary.json")
    if args.config_snapshot_path is None:
        args.config_snapshot_path = str(output_dir / "config_snapshot.yaml")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(output_dir / "checkpoints")


def configure_data_cache(args: argparse.Namespace) -> None:
    cache_dir = resolve_repo_path(Path(args.data_cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_home = cache_dir / "hf_home"
    datasets_cache = cache_dir / "hf_datasets"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)


def build_train_dataloader(
    samples: Sequence[Sample],
    args: argparse.Namespace,
) -> DataLoader[tuple[list[int], list[Sample]]]:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataset = SampleDataset(samples)
    sampler = None
    if getattr(args, "distributed", False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=int(args.distributed_world_size),
            rank=int(args.distributed_rank),
            shuffle=args.shuffle,
            drop_last=args.drop_last and len(samples) >= args.global_batch_size,
            seed=args.seed,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle and sampler is None,
        sampler=sampler,
        drop_last=args.drop_last and len(samples) >= args.global_batch_size,
        collate_fn=collate_sample_items,
        generator=generator if sampler is None else None,
    )


def collate_sample_items(items: Sequence[tuple[int, Sample]]) -> tuple[list[int], list[Sample]]:
    indices = [int(index) for index, _ in items]
    samples = [sample for _, sample in items]
    return indices, samples


def iter_train_batches(
    dataloader: DataLoader[tuple[list[int], list[Sample]]],
    *,
    start_epoch: int = 0,
) -> Any:
    epoch = start_epoch
    while True:
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


def steps_per_epoch(sample_count: int, batch_size: int, drop_last: bool) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if drop_last:
        return max(1, sample_count // batch_size)
    return max(1, math.ceil(sample_count / batch_size))


def epoch_index(step: int, epoch_steps: int) -> int:
    if epoch_steps < 1:
        return 0
    return math.ceil(step / epoch_steps)


def latent_counts_for_objective(
    teacher_targets: TeacherTargetBatch,
    args: argparse.Namespace,
) -> list[int]:
    if args.objective == "step_trajectory":
        counts = teacher_targets.target_mask.to(dtype=torch.long).sum(dim=1)
        return [max(1, int(count.detach().cpu().item())) for count in counts]
    return [int(args.num_latent_steps)] * int(teacher_targets.target_mask.shape[0])


def align_student_teacher_states(
    student_states: torch.Tensor,
    student_mask: torch.Tensor,
    teacher_states: torch.Tensor,
    teacher_mask: torch.Tensor,
    *,
    objective: str,
    alignment: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del alignment
    teacher_states = teacher_states.to(device=student_states.device, dtype=student_states.dtype).detach()
    teacher_mask = teacher_mask.to(device=student_states.device, dtype=torch.bool)
    student_mask = student_mask.to(device=student_states.device, dtype=torch.bool)
    if objective == "sequence":
        h_final, h_final_mask = select_last_valid_state(student_states, student_mask)
        z_final, z_final_mask = select_last_valid_state(teacher_states, teacher_mask)
        return h_final, z_final, h_final_mask & z_final_mask
    if objective != "step_trajectory":
        raise ValueError(f"unsupported objective: {objective}")
    steps = min(student_states.shape[1], teacher_states.shape[1])
    aligned_student = student_states[:, :steps, :]
    aligned_teacher = teacher_states[:, :steps, :]
    aligned_mask = student_mask[:, :steps] & teacher_mask[:, :steps]
    return aligned_student, aligned_teacher, aligned_mask


def compute_step_trajectory_state_loss(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
    mask: torch.Tensor,
    *,
    alignment: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    if alignment not in {"normalized_mse", "masked_normalized_mse"}:
        return compute_lsp_state_loss(student_states, teacher_states, mask, alignment=alignment)
    if student_states.shape != teacher_states.shape:
        raise ValueError("student_states and teacher_states must have identical shape")
    mask = mask.to(device=student_states.device, dtype=torch.bool)
    student_norm = F.normalize(student_states, p=2, dim=-1, eps=eps)
    teacher_norm = F.normalize(teacher_states.detach(), p=2, dim=-1, eps=eps)
    step_loss = (student_norm - teacher_norm).square().mean(dim=-1)
    safe_step_loss = torch.where(mask, step_loss, torch.zeros_like(step_loss))
    valid_counts = mask.to(dtype=step_loss.dtype).sum(dim=1)
    per_sample = safe_step_loss.sum(dim=1) / valid_counts.clamp_min(1.0)
    valid_samples = valid_counts.gt(0)
    if not bool(valid_samples.any().detach().cpu().item()):
        return student_states.sum() * 0.0
    return per_sample[valid_samples].mean()


def build_tqdm_progress(args: argparse.Namespace, *, initial: int = 0) -> Any:
    if not args.tqdm_progress or tqdm is None:
        return None
    progress = tqdm(
        total=args.max_steps,
        initial=initial,
        desc=args.objective,
        dynamic_ncols=sys.stdout.isatty(),
        mininterval=1.0,
        file=sys.stdout,
        ascii=not sys.stdout.isatty(),
        disable=not sys.stdout.isatty(),
    )
    progress._lsp_jepa_log_start_time = time.monotonic()
    return progress


def update_tqdm_progress(progress: Any, metrics: Mapping[str, Any]) -> None:
    if progress is None:
        return
    progress.set_postfix(
        total_loss=f"{float(metrics['total_loss']):.6f}",
        lsp_loss=f"{float(metrics['lsp_loss']):.6f}",
        host_answer_ce=f"{float(metrics['host_answer_ce']):.6f}",
        latent_var=f"{float(metrics['latent_variance_mean']):.4g}",
        refresh=False,
    )
    progress.update(1)


def build_tqdm_log_line(progress: Any, metrics: Mapping[str, Any]) -> str | None:
    if progress is None or tqdm is None:
        return None
    start_time = getattr(progress, "_lsp_jepa_log_start_time", None)
    if start_time is not None:
        elapsed = max(1e-9, time.monotonic() - float(start_time))
    else:
        elapsed = float(getattr(progress, "format_dict", {}).get("elapsed") or 0.0)
    step = int(metrics["step"])
    max_steps = int(metrics["max_steps"])
    epoch_steps = max(1, int(metrics.get("epoch_steps") or max_steps))
    epoch_index = int(metrics.get("epoch_index") or ((step - 1) // epoch_steps + 1))
    total_epochs = max(1, math.ceil(max_steps / epoch_steps))
    epoch_step = ((step - 1) % epoch_steps) + 1
    if step >= max_steps and max_steps % epoch_steps:
        epoch_step = max_steps % epoch_steps
    steps_per_second = step / elapsed if elapsed > 0.0 else 0.0
    epoch_eta = "?"
    if steps_per_second > 0.0:
        epoch_remaining = max(0, epoch_steps - epoch_step)
        epoch_eta = tqdm.format_interval(epoch_remaining / steps_per_second)
    postfix = (
        f"total_loss={float(metrics['total_loss']):.6f}, "
        f"lsp_loss={float(metrics['lsp_loss']):.6f}, "
        f"host_answer_ce={float(metrics['host_answer_ce']):.6f}, "
        f"latent_var={float(metrics['latent_variance_mean']):.4g}, "
        f"epoch={epoch_index}/{total_epochs}, "
        f"epoch_step={epoch_step}/{epoch_steps}, "
        f"epoch_eta={epoch_eta}"
    )
    meter = tqdm.format_meter(
        n=step,
        total=max_steps,
        elapsed=elapsed,
        prefix=str(metrics.get("training_objective") or metrics.get("objective") or "train"),
        ascii=False,
        ncols=None,
        postfix=postfix,
    )
    return meter


def progress_write(progress: Any, text: str) -> None:
    if progress is not None and sys.stdout.isatty():
        progress.write(text)
    else:
        print(text, flush=True)


def close_tqdm_progress(progress: Any) -> None:
    if progress is not None:
        progress.close()


def should_log_step(args: argparse.Namespace, step: int) -> bool:
    return step == 1 or step == args.max_steps or (args.log_every > 0 and step % args.log_every == 0)


def should_log_json(args: argparse.Namespace, step: int) -> bool:
    return args.console_json_every > 0 and (
        step == 1
        or step == args.max_steps
        or step % args.console_json_every == 0
    )


def should_compute_effective_rank(args: argparse.Namespace, step: int) -> bool:
    return args.effective_rank_every > 0 and (
        step == 1
        or step == args.max_steps
        or step % args.effective_rank_every == 0
    )


def model_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    embeddings = model.get_input_embeddings()
    return int(embeddings.embedding_dim)


def load_training_checkpoint(
    path: Path,
    *,
    student: nn.Module,
    ema_teacher: EMATeacher,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint_path = resolve_repo_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping payload: {checkpoint_path}")
    step = int(checkpoint.get("step") or 0)
    if step < 1:
        raise ValueError(f"checkpoint has invalid step={step}: {checkpoint_path}")
    student_state = checkpoint.get("student_base_causallm")
    ema_state = checkpoint.get("ema_teacher")
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(student_state, Mapping):
        raise ValueError(f"checkpoint missing student_base_causallm state: {checkpoint_path}")
    if not isinstance(ema_state, Mapping):
        raise ValueError(f"checkpoint missing ema_teacher state: {checkpoint_path}")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError(f"checkpoint missing optimizer state: {checkpoint_path}")
    student.base_causallm.load_state_dict(student_state)
    ema_teacher.teacher_model.load_state_dict(ema_state)
    optimizer.load_state_dict(optimizer_state)
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    return step


def maybe_save_checkpoint(
    args: argparse.Namespace,
    *,
    step: int,
    checkpoint_dir: Path,
    student: nn.Module,
    ema_teacher: EMATeacher,
    optimizer: torch.optim.Optimizer,
) -> Path | None:
    if not args.save_checkpoints:
        return None
    if args.save_every <= 0:
        return None
    if step != args.max_steps and step % args.save_every != 0:
        return None
    path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "student_base_causallm": student.base_causallm.state_dict(),
            "ema_teacher": ema_teacher.teacher_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "max_steps": args.max_steps,
                "objective": args.objective,
                "ema_decay": args.ema_decay,
                "lsp_weight": args.lsp_weight,
                "host_answer_ce_weight": args.host_answer_ce_weight,
                "anti_collapse_type": args.anti_collapse_type,
                "anti_collapse_weight": args.anti_collapse_weight,
            },
        },
        path,
    )
    return path


def prune_old_checkpoints(
    checkpoint_dir: Path,
    *,
    keep_last: int,
    keep_paths: Sequence[Path] = (),
) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda path: (checkpoint_step_number(path), path.name),
    )
    keep = {path.resolve() for path in keep_paths if path is not None and path.exists()}
    keep.update(path.resolve() for path in checkpoints[-keep_last:])
    for old_path in checkpoints:
        if old_path.resolve() in keep:
            continue
        old_path.unlink(missing_ok=True)


def checkpoint_step_number(path: Path) -> int:
    match = re.match(r"step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def load_epoch_eval_samples(args: argparse.Namespace) -> list[Sample]:
    limit = None if args.eval_limit_samples is None or args.eval_limit_samples <= 0 else int(args.eval_limit_samples)
    if args.eval_json is not None:
        return load_samples_from_path(resolve_repo_path(args.eval_json), limit=limit)
    return load_hf_dataset_samples(
        dataset_id=args.eval_dataset_id,
        config=args.eval_dataset_config,
        split=args.eval_split,
        limit=limit,
        expected_samples=args.eval_expected_samples,
        hf_endpoint=args.hf_endpoint,
    )


def maybe_run_epoch_eval(
    args: argparse.Namespace,
    *,
    step: int,
    student: nn.Module,
    tokenizer: Any,
    eval_samples: Sequence[Sample],
    checkpoint_path: Path | None,
    eval_output_dir: Path,
    eval_metrics_path: Path,
    device: torch.device,
) -> dict[str, Any] | None:
    if args.eval_every <= 0:
        return None
    if step != args.max_steps and step % args.eval_every != 0:
        return None
    if not eval_samples:
        raise RuntimeError("eval_every is enabled but no eval samples were loaded")

    was_training = student.training
    student.eval()
    rows = []
    exact_matches = 0
    numeric_matches = 0
    invalid_answers = 0
    generated_lengths = []
    with torch.no_grad():
        for index, sample in enumerate(eval_samples):
            input_ids = build_final_answer_eval_input_ids(
                tokenizer,
                sample.question,
                num_latent_steps=args.num_latent_steps,
                device=device,
            )
            outputs = student.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, device=device),
                max_new_tokens=args.eval_max_new_tokens,
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
    if was_training:
        student.train()
    else:
        student.eval()

    total = len(eval_samples)
    step_metrics_path = eval_output_dir / f"step_{step:06d}.json"
    metrics = {
        "step": step,
        "epoch_index": epoch_index(step, args.epoch_steps),
        "epoch_steps": args.epoch_steps,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
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
        "limit_eval_samples": args.eval_limit_samples,
        "eval_json": str(resolve_repo_path(args.eval_json)) if args.eval_json else None,
        "eval_split": args.eval_split,
        "dataset_id": args.eval_dataset_id,
        "dataset_config": args.eval_dataset_config,
        "answer_normalization": "Coconut baseline: split on '#', remove commas, strip whitespace",
        "generation": {
            "generate_cot": False,
            "final_answer_only": True,
            "max_new_tokens": args.eval_max_new_tokens,
            "greedy": True,
        },
        "core_eval_contract": {
            "no_teacher_branch": True,
            "ema_updated": False,
            "step_level_eval": False,
            "adapter_expansion": False,
        },
        "examples": rows[: args.eval_save_examples],
        "metrics_path": str(eval_metrics_path),
        "step_metrics_path": str(step_metrics_path),
    }
    write_json(step_metrics_path, metrics)
    write_json(eval_output_dir / "latest.json", metrics)
    append_metrics(eval_metrics_path, metrics)
    print(json.dumps({"epoch_eval": compact_eval_metrics(metrics)}, sort_keys=True))
    return metrics


def compact_eval_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step": metrics["step"],
        "epoch_index": metrics["epoch_index"],
        "checkpoint_path": metrics["checkpoint_path"],
        "accuracy": metrics["accuracy"],
        "exact_match": metrics["exact_match"],
        "invalid_answer_rate": metrics["invalid_answer_rate"],
        "num_eval_samples": metrics["num_eval_samples"],
        "step_metrics_path": metrics["step_metrics_path"],
    }


def build_final_answer_eval_input_ids(
    tokenizer: Any,
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


def decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
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
    return float(numerator / denominator) if denominator else 0.0


def mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def batch_indices_for_step(
    *,
    sample_count: int,
    batch_size: int,
    step: int,
) -> list[int]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    start = ((step - 1) * batch_size) % sample_count
    return [(start + offset) % sample_count for offset in range(batch_size)]


def pairwise_cosine_summary(
    states: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor | bool]:
    matrix = pairwise_cosine(states, mask, reduction="none")
    mean = pairwise_cosine(states, mask)
    if matrix.shape[0] < 2:
        zero = states.sum() * 0.0
        return {"mean": mean, "min": zero, "max": zero, "all_one": False}
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=torch.bool)
    values = matrix[~eye]
    if values.numel() == 0:
        zero = states.sum() * 0.0
        return {"mean": mean, "min": zero, "max": zero, "all_one": False}
    return {
        "mean": mean,
        "min": values.min(),
        "max": values.max(),
        "all_one": bool(
            torch.allclose(
                values.detach(),
                torch.ones_like(values),
                rtol=1e-5,
                atol=1e-6,
            )
        ),
    }


def effective_rank_diagnostic(
    states: torch.Tensor,
    mask: torch.Tensor,
    *,
    compute: bool,
) -> tuple[torch.Tensor | None, str, str | None]:
    if not compute:
        return None, "skipped", None
    try:
        rank = effective_rank(states, mask)
    except RuntimeError as exc:
        message = str(exc)
        if "linalg" not in message.lower() and "eig" not in message.lower():
            raise
        return None, "failed", format_diagnostic_error(exc)
    if not bool(torch.isfinite(rank).detach().cpu().item()):
        return None, "nonfinite", "effective_rank is not finite"
    return rank, "ok", None


def format_diagnostic_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message}"[:500]


def configured_answer_ce_terms(args: argparse.Namespace) -> list[str]:
    terms = []
    if args.host_answer_ce_weight != 0.0:
        terms.append("host_answer_ce")
    return terms


def answer_ce_terms_in_total(
    args: argparse.Namespace,
    host_answer_ce: torch.Tensor | None,
) -> list[str]:
    if host_answer_ce is None:
        return []
    return configured_answer_ce_terms(args)


def build_config_snapshot(
    args: argparse.Namespace,
    *,
    device: torch.device,
    sample_count: int,
    metrics_path: Path,
    summary_path: Path,
    config_snapshot_path: Path,
) -> dict[str, Any]:
    data_source = args.dataset_source if args.data_path is None else str(args.data_path)
    answer_ce_terms = configured_answer_ce_terms(args)
    return {
        "experiment": {
            "mode": "core",
            "host": "simcot",
            "backbone": "coconut",
            "use_lsp_jepa": True,
            "run_kind": (
                "LSP-JEPA-Core StepTrajectory train"
                if args.objective == "step_trajectory"
                else "LSP-JEPA-Core SequenceFinal train"
            ),
        },
        "teacher": {
            "ema_decay": args.ema_decay,
            "update_trainable_only": True,
            "output_hidden_states": True,
            "target_layer": args.target_layer,
            "target_pooling": "step_last_token",
            "target_space": "raw_hidden",
            "exclude_answer_tokens": not args.include_answer_tokens,
            "exclude_answer_prefix": not args.include_answer_prefix,
            "reasoning_step_filter": args.reasoning_step_filter,
            "target_position": (
                "all_valid_reasoning_step_boundaries"
                if args.objective == "step_trajectory"
                else "final_valid_reasoning_step"
            ),
        },
        "student": {
            "latent_arch": "latent_tokens",
            "num_latent_steps": args.num_latent_steps,
            "latent_dim": None,
            "normalize_latents": True,
            "detach_between_steps": False,
        },
        "lsp_objective": {
            "type": "step_trajectory" if args.objective == "step_trajectory" else "state",
            "state_offset": 0,
            "transition_weight": 0.0,
        },
        "mapping": {
            "strategy": "one_to_one" if args.objective == "step_trajectory" else "sequence",
            "sparse_positions": ["first", "middle", "final"],
            "attention_coverage_weight": 0.0,
        },
        "loss": {
            "alignment": args.alignment_loss,
            "align_weight": args.lsp_weight,
            "anti_collapse": args.anti_collapse_type,
            "anti_collapse_weight": args.anti_collapse_weight,
            "answer_readout_weight": 0.0,
        },
        "host_losses": {
            "host_answer_ce_weight": args.host_answer_ce_weight,
            "codi_distill_weight": 0.0,
            "simcot_decoder_weight": 0.0,
            "intermediate_cot_ce_weight": 0.0,
            "answer_ce_terms_in_total": answer_ce_terms,
            "answer_ce_double_count_ok": len(answer_ce_terms) <= 1,
        },
        "optimization": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_steps": args.max_steps,
            "supported_max_steps": [100, 500],
            "batch_size": args.batch_size,
            "local_batch_size": args.local_batch_size,
            "global_batch_size": args.global_batch_size,
            "distributed": args.distributed,
            "distributed_world_size": args.distributed_world_size,
            "epoch_steps": args.epoch_steps,
            "save_every": args.save_every,
            "save_every_epoch": args.save_every_epoch,
            "keep_last_checkpoints": args.keep_last_checkpoints,
            "keep_best_total_loss": args.keep_best_total_loss,
            "log_every": args.log_every,
            "console_json_every": args.console_json_every,
            "effective_rank_every": args.effective_rank_every,
            "save_checkpoints": args.save_checkpoints,
            "single_backward_per_batch": True,
            "ema_update_after_optimizer_step": True,
            "no_step_level_training": args.objective != "step_trajectory",
            "step_trajectory_training": args.objective == "step_trajectory",
            "no_adapter_expansion": True,
        },
        "data": {
            "source": data_source,
            "dataset_id": args.dataset_id,
            "dataset_split": args.dataset_split,
            "dataset_config": args.dataset_config,
            "expected_samples": args.expected_samples,
            "num_samples_requested": args.num_samples,
            "num_samples_loaded": sample_count,
            "full_train_dataloader": True,
            "autodl_root": args.autodl_root,
            "data_cache_dir": str(resolve_repo_path(Path(args.data_cache_dir))),
            "dataset_local_path": (
                str(resolve_repo_path(args.dataset_local_path))
                if args.dataset_local_path is not None
                else None
            ),
            "drop_last": args.drop_last,
            "shuffle": args.shuffle,
            "min_samples": args.min_samples,
            "max_teacher_length": args.max_teacher_length,
        },
        "runtime": {
            "model_id": args.model_id,
            "device": str(device),
            "seed": args.seed,
        },
        "eval": {
            "every_steps": args.eval_every,
            "every_epoch": args.eval_every_epoch,
            "output_dir": args.eval_output_dir,
            "metrics_path": args.eval_metrics_path,
            "eval_json": str(args.eval_json) if args.eval_json else None,
            "eval_split": args.eval_split,
            "dataset_id": args.eval_dataset_id,
            "dataset_config": args.eval_dataset_config,
            "expected_samples": args.eval_expected_samples,
            "limit_samples": args.eval_limit_samples,
            "max_new_tokens": args.eval_max_new_tokens,
            "save_examples": args.eval_save_examples,
            "final_answer_only": True,
            "generate_cot": False,
        },
        "logging": {
            "log_latent_metrics": True,
            "effective_rank_online": args.effective_rank_every > 0,
            "effective_rank_every": args.effective_rank_every,
            "log_alignment_matrices": False,
            "log_target_leakage_checks": True,
            "log_host_losses": True,
        },
        "outputs": {
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "config_snapshot_path": str(config_snapshot_path),
            "checkpoint_dir": str(resolve_metrics_path(args.checkpoint_dir)),
        },
    }


def build_summary(
    metrics_history: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    sample_count: int,
    unique_samples_seen: int,
    metrics_path: Path,
    summary_path: Path,
    config_snapshot_path: Path,
) -> dict[str, Any]:
    if not metrics_history:
        raise RuntimeError("cannot build summary without metrics")
    first = metrics_history[0]
    last = metrics_history[-1]
    tail_count = max(1, len(metrics_history) // 4)
    tail = metrics_history[-tail_count:]
    first_lsp = float(first["lsp_loss"])
    last_lsp = float(last["lsp_loss"])
    tail_lsp_mean = sum(float(item["lsp_loss"]) for item in tail) / len(tail)
    lsp_loss_stable_or_decreasing = tail_lsp_mean <= first_lsp * args.loss_stability_ratio
    tail_pairwise_cosine_mean = sum(float(item["pairwise_cosine_mean"]) for item in tail) / len(tail)
    effective_rank_values = [
        value
        for value in (optional_float(item.get("effective_rank")) for item in metrics_history)
        if value is not None
    ]
    effective_rank_statuses = [
        str(item.get("effective_rank_status", "legacy")) for item in metrics_history
    ]
    total_loss_finite = all(math.isfinite(float(item["total_loss"])) for item in metrics_history)
    lsp_loss_finite = all(math.isfinite(float(item["lsp_loss"])) for item in metrics_history)
    host_answer_ce_finite = all(
        math.isfinite(float(item["host_answer_ce"])) for item in metrics_history
    )
    anti_collapse_finite = all(
        math.isfinite(float(item["anti_collapse_loss"])) for item in metrics_history
    )
    latent_variance_nonzero = all(
        float(item["latent_variance_mean"]) > args.collapse_eps
        for item in metrics_history
    )
    pairwise_cosine_not_all_one = not any(
        bool(item["pairwise_cosine_all_one"]) for item in metrics_history
    )
    pairwise_cosine_not_long_near_one = (
        pairwise_cosine_not_all_one
        and tail_pairwise_cosine_mean < args.pairwise_cosine_fail_threshold
    )
    answer_ce_double_count_ok = all(
        bool(item["answer_ce_double_count_ok"]) for item in metrics_history
    )
    answer_leakage_ok = all(bool(item["answer_leakage_ok"]) for item in metrics_history)
    teacher_no_grad = all(int(item["teacher_grad_params"]) == 0 for item in metrics_history)
    student_has_grad = all(
        float(item["student_grad_l1"]) > 0.0 and float(item["student_base_grad_l1"]) > 0.0
        for item in metrics_history
    )
    ema_drift_nonzero = all(float(item["ema_drift_l1"]) > 0.0 for item in metrics_history)
    full_dataset_loaded = args.expected_samples <= 0 or sample_count >= args.expected_samples
    supported_max_steps = args.max_steps in {100, 500}
    autodl_root = resolve_repo_path(Path(args.autodl_root))
    outputs_under_autodl = all(
        path_is_under(path, autodl_root)
        for path in (metrics_path, summary_path, config_snapshot_path, resolve_metrics_path(args.checkpoint_dir))
    )
    data_under_autodl = path_is_under(resolve_repo_path(Path(args.data_cache_dir)), autodl_root)
    acceptance = {
        "training_completed": len(metrics_history) == args.max_steps,
        "max_steps_100_or_500_supported": supported_max_steps,
        "full_train_dataloader_used": all(bool(item["full_train_dataloader"]) for item in metrics_history),
        "full_expected_sample_count_reached": full_dataset_loaded,
        "data_under_autodl_tmp": data_under_autodl,
        "outputs_under_autodl_tmp": outputs_under_autodl,
        "total_loss_finite": total_loss_finite,
        "lsp_loss_calculable": lsp_loss_finite,
        "host_answer_ce_calculable": host_answer_ce_finite,
        "anti_collapse_loss_calculable": anti_collapse_finite,
        "answer_ce_double_count_ok": answer_ce_double_count_ok,
        "student_has_grad": student_has_grad,
        "ema_drift_nonzero": ema_drift_nonzero,
        "latent_variance_nonzero": latent_variance_nonzero,
        "pairwise_cosine_not_long_near_one": pairwise_cosine_not_long_near_one,
        "teacher_no_grad": teacher_no_grad,
        "answer_leakage_ok": answer_leakage_ok,
        "training_mode_matches_objective": (
            all(bool(item.get("step_trajectory_training")) for item in metrics_history)
            if args.objective == "step_trajectory"
            else all(bool(item["no_step_level_training"]) for item in metrics_history)
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    return {
        "run": {
            "name": "pr11_mvp_full_seq_debug",
            "metrics_path": str(metrics_path),
            "summary_path": str(summary_path),
            "config_snapshot_path": str(config_snapshot_path),
            "steps_completed": len(metrics_history),
            "max_steps": args.max_steps,
            "sample_count": sample_count,
            "expected_samples": args.expected_samples,
            "unique_samples_seen": unique_samples_seen,
            "full_train_dataloader": True,
            "dataloader_drop_last": args.drop_last,
            "data_cache_dir": str(resolve_repo_path(Path(args.data_cache_dir))),
            "dataset_local_path": (
                str(resolve_repo_path(args.dataset_local_path))
                if args.dataset_local_path is not None
                else None
            ),
            "checkpoint_dir": str(resolve_metrics_path(args.checkpoint_dir)),
            "last_checkpoint_path": last.get("checkpoint_path"),
            "checkpoint_keep_last": args.keep_last_checkpoints,
            "best_total_loss": last.get("best_total_loss"),
            "best_checkpoint_path": last.get("best_checkpoint_path"),
            "epoch_steps": args.epoch_steps,
        },
        "loss_curve": {
            "first_total_loss": float(first["total_loss"]),
            "last_total_loss": float(last["total_loss"]),
            "first_lsp_loss": first_lsp,
            "last_lsp_loss": last_lsp,
            "tail_lsp_loss_mean": tail_lsp_mean,
            "lsp_loss_delta_last_minus_first": last_lsp - first_lsp,
            "loss_stability_ratio": args.loss_stability_ratio,
            "lsp_loss_stable_or_decreasing_diagnostic": lsp_loss_stable_or_decreasing,
        },
        "answer_ce": {
            "host_answer_ce_first": float(first["host_answer_ce"]),
            "host_answer_ce_last": float(last["host_answer_ce"]),
            "host_answer_ce_weight": args.host_answer_ce_weight,
            "terms_in_total": list(last["answer_ce_terms_in_total"]),
            "double_count_ok": answer_ce_double_count_ok,
        },
        "collapse": {
            "latent_variance_mean_first": float(first["latent_variance_mean"]),
            "latent_variance_mean_last": float(last["latent_variance_mean"]),
            "latent_variance_min_last": float(last["latent_variance_min"]),
            "effective_rank_last": optional_float(last.get("effective_rank")),
            "effective_rank_ok_count": len(effective_rank_values),
            "effective_rank_failed_count": sum(
                status in {"failed", "nonfinite"} for status in effective_rank_statuses
            ),
            "effective_rank_skipped_count": sum(
                status == "skipped" for status in effective_rank_statuses
            ),
            "effective_rank_every": args.effective_rank_every,
            "effective_rank_status_last": last.get("effective_rank_status", "legacy"),
            "effective_rank_error_last": last.get("effective_rank_error"),
            "pairwise_cosine_mean_last": float(last["pairwise_cosine_mean"]),
            "pairwise_cosine_tail_mean": tail_pairwise_cosine_mean,
            "pairwise_cosine_fail_threshold": args.pairwise_cosine_fail_threshold,
            "pairwise_cosine_min_last": float(last["pairwise_cosine_min"]),
            "pairwise_cosine_max_last": float(last["pairwise_cosine_max"]),
            "pairwise_cosine_all_one_last": bool(last["pairwise_cosine_all_one"]),
            "pairwise_l2_mean_last": float(last["pairwise_l2_mean"]),
        },
        "grad_contract": {
            "student_grad_l1_last": float(last["student_grad_l1"]),
            "student_base_grad_l1_last": float(last["student_base_grad_l1"]),
            "teacher_grad_params_last": int(last["teacher_grad_params"]),
            "ema_drift_l1_last": float(last["ema_drift_l1"]),
        },
        "leakage": {
            "teacher_exclude_answer_tokens": bool(last["teacher_exclude_answer_tokens"]),
            "teacher_exclude_answer_prefix": bool(last["teacher_exclude_answer_prefix"]),
            "answer_leakage_ok": answer_leakage_ok,
        },
        "epoch_eval": {
            "enabled": args.eval_every > 0,
            "every_steps": args.eval_every,
            "latest": last.get("epoch_eval"),
        },
        "acceptance": acceptance,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_config_snapshot(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(to_simple_yaml(config), encoding="utf-8")


def to_simple_yaml(value: Any, *, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if isinstance(item, Mapping) and item:
                lines.append(f"{prefix}{key}:")
                lines.append(to_simple_yaml(item, indent=indent + 2).rstrip())
            elif isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}{key}: []")
                elif all(is_yaml_scalar(entry) for entry in item):
                    rendered = ", ".join(yaml_scalar(entry) for entry in item)
                    lines.append(f"{prefix}{key}: [{rendered}]")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.append(to_simple_yaml(item, indent=indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]\n"
        if all(is_yaml_scalar(item) for item in value):
            return f"{prefix}[{', '.join(yaml_scalar(item) for item in value)}]\n"
        lines = []
        for item in value:
            if isinstance(item, Mapping):
                lines.append(f"{prefix}-")
                lines.append(to_simple_yaml(item, indent=indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{yaml_scalar(value)}\n"


def is_yaml_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))


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
    ensure_legacy_past_key_values(model)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def ensure_legacy_past_key_values(model: nn.Module) -> None:
    """Keep upstream Coconut compatible with newer transformers cache objects."""
    if getattr(model, "_lsp_jepa_legacy_cache_wrapped", False):
        return
    original_forward = model.forward

    def forward_with_legacy_cache(*args: Any, **kwargs: Any) -> Any:
        past_key_values = kwargs.get("past_key_values")
        if isinstance(past_key_values, (list, tuple)):
            try:
                from transformers.cache_utils import DynamicCache
            except ImportError:
                pass
            else:
                kwargs = dict(kwargs)
                kwargs["past_key_values"] = DynamicCache(
                    past_key_values,
                    config=getattr(model, "config", None),
                )
        outputs = original_forward(*args, **kwargs)
        past_key_values = getattr(outputs, "past_key_values", None)
        if hasattr(past_key_values, "to_legacy_cache"):
            outputs.past_key_values = past_key_values.to_legacy_cache()
        elif past_key_values is not None:
            legacy_cache = []
            for layer_cache in past_key_values:
                if not isinstance(layer_cache, tuple) or len(layer_cache) < 2:
                    legacy_cache = []
                    break
                legacy_cache.append((layer_cache[0], layer_cache[1]))
            if legacy_cache:
                outputs.past_key_values = tuple(legacy_cache)
        return outputs

    model.forward = forward_with_legacy_cache  # type: ignore[method-assign]
    model._lsp_jepa_legacy_cache_wrapped = True  # type: ignore[attr-defined]


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
    latent_counts: Sequence[int] | None = None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = []
    label_rows = []
    first_latent_positions = []
    if latent_counts is None:
        latent_counts = [num_latent_steps] * len(samples)
    if len(latent_counts) != len(samples):
        raise ValueError("latent_counts must match batch size")
    for sample, latent_count in zip(samples, latent_counts, strict=True):
        latent_count = max(1, int(latent_count))
        question_ids = list(tokenizer.encode(sample.question + "\n", add_special_tokens=True))
        answer_ids = list(tokenizer.encode("### " + sample.answer, add_special_tokens=False))
        answer_ids.append(int(tokenizer.eos_token_id))
        input_ids = (
            question_ids
            + [start_id]
            + [latent_id] * latent_count
            + [end_id]
            + answer_ids
        )
        labels = [-100] * (len(question_ids) + latent_count + 2) + answer_ids
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
