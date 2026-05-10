# PR13 MVP Full Sequence Train Runbook

This runbook launches full-data sequence-level LSP-JEPA-Core training from the
PR12 config without adding algorithms, adapters, step-level objectives,
attention, soft-DTW, looped rollout, DDP, or FSDP.

## Scope

- Config: `lsp_jepa/configs/core/lsp_seq_full_train.yaml`
- Launch script: `lsp_jepa/scripts/launch_full_seq_train.sh`
- Readiness report: `lsp_jepa/runs/pr10_mvp/full_data_readiness_report.json`
- Output root: `/root/autodl-tmp/lsp_jepa/runs/pr13_mvp_full_seq_launch`

The launch script gates the config before training:

- `experiment.mode: core`
- `lsp_objective.objective: lsp_state`
- `mapping.strategy: sequence`
- `host_losses.simcot_decoder_weight: 0.0`
- `host_losses.codi_distill_weight: 0.0`
- `host_losses.intermediate_cot_ce_weight: 0.0`
- `teacher.exclude_answer_tokens: true`
- `teacher.exclude_answer_prefix: true`
- `teacher.target_position: final_valid_reasoning_step`
- `student.detach_between_steps: false`
- `optimization_contract.ema_update_after_optimizer_step: true`

It also gates the full-data readiness report:

- `empty_teacher_target_rate == 0`
- `latent_empty_batch_rate == 0`
- `answer_leakage_count == 0`
- `skipped_sample_count == 0`
- all readiness acceptance fields are true

## Required Sequence

Run from the repository root on a single GPU or CPU process:

```bash
lsp_jepa/scripts/launch_full_seq_train.sh --dry-run
lsp_jepa/scripts/launch_full_seq_train.sh --max-steps 10
lsp_jepa/scripts/launch_full_seq_train.sh --max-steps 100
```

Each run writes `summary.json`, `metrics.jsonl`, `config.yaml`, checkpoints, and
logs under its output directory. The script appends a `pr13_mvp` block to the
summary and fails if any PR13 launch gate fails.

## PR13 Runtime Gates

The launch summary checks:

- no NaN or Inf metrics
- teacher parameters have no grad
- student backbone and predictor have grad
- EMA drift is nonzero after `optimizer.step`
- latent variance is nonzero
- pairwise cosine tail mean is below `0.999` and not all one
- answer CE appears through at most `host_answer_ce`
- teacher target and student latent masks are nonempty
- answer leakage checks pass

## Full 1000-Step Command

Only after dry-run, 10-step, and 100-step runs pass:

```bash
lsp_jepa/scripts/launch_full_seq_train.sh --max-steps 1000
```

Do not start this as an unattended background run until the operator confirms.
