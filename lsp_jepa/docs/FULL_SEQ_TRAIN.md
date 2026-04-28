# Full Sequence Train

PR12-MVP-FullSeq-TrainConfig promotes the PR11 debug sequence-level
LSP-JEPA-Core run into a formal full-dataset training configuration. It does
not add a new algorithm and does not change `train_lsp_jepa_core.py`.

The default run remains core-mode sequence-level LSP-State:

```text
h_final <-> z_final
```

Disabled by default:

- step-level LSP losses;
- CODI or SIM-CoT adapter objectives;
- SIM-CoT decoder CE, CODI distillation, and intermediate CoT CE;
- attention alignment and soft-DTW;
- looped latent rollout;
- answer-token or answer-prefix teacher targets.

## Files

```text
lsp_jepa/configs/core/lsp_seq_full_train.yaml
lsp_jepa/scripts/run_full_seq_train.sh
outputs/lsp_jepa/full_seq/config.yaml
outputs/lsp_jepa/full_seq/metrics.jsonl
outputs/lsp_jepa/full_seq/summary.json
outputs/lsp_jepa/full_seq/checkpoints/
outputs/lsp_jepa/full_seq/logs/
```

## Defaults

The formal config uses the PR10 readiness report recommendation:

- batch size: `16`;
- max teacher length: `512`;
- one epoch steps: `468`, covering the 7,473-sample GSM8K train split with
  `drop_last=false`;
- gradient accumulation steps: `1`.

The Coconut GSM baseline lr is `1e-4`, so PR12 uses `5e-5`.

Core hyperparameters:

- `ema_decay: 0.995`;
- `lsp_weight: 1.0`;
- `host_answer_ce_weight: 0.1`;
- `anti_collapse_weight: 0.01`;
- `target_layer: last_2`;
- `loss: normalized_mse`;
- `gradient_clip_norm: 1.0`;
- `warmup_ratio: 0.03`.

`gradient_clip_norm`, `warmup_ratio`, and `gradient_accumulation_steps` are
declared in the formal config for reproducibility. PR12 intentionally does not
modify the PR11 trainer to add scheduler, clipping, or accumulation behavior.

## Commands

Dry run:

```bash
lsp_jepa/scripts/run_full_seq_train.sh --dry-run
```

Ten-step acceptance run:

```bash
lsp_jepa/scripts/run_full_seq_train.sh --max-steps 10
```

One-hundred-step acceptance run:

```bash
lsp_jepa/scripts/run_full_seq_train.sh --max-steps 100
```

One full epoch:

```bash
lsp_jepa/scripts/run_full_seq_train.sh --one-epoch
```

Equivalent default when `--max-steps` is omitted:

```bash
lsp_jepa/scripts/run_full_seq_train.sh
```

Resume into the same output directory:

```bash
lsp_jepa/scripts/run_full_seq_train.sh --max-steps 100 --resume
```

The PR11 trainer writes checkpoints but does not expose stateful checkpoint
loading. The PR12 wrapper preserves existing metrics on `--resume`, verifies
that a checkpoint exists, appends new metrics, and records
`pr12_mvp.resume_requested` in `summary.json`.

## Summary Fields

After each wrapper run, `summary.json` includes `pr12_mvp`:

- `final_loss`;
- `final_lsp_loss`;
- `effective_sample_count`;
- `skip_sample_count`;
- `target_empty_sample_count`;
- `has_nan`;
- `acceptance_passed`;
- `resume_requested`;
- output paths for config, metrics, summary, checkpoints, and logs.

The underlying PR11 summary is retained, including loss curve, collapse
diagnostics, gradient/EMA checks, and answer-leakage checks.
