# PR11 MVP Full Sequence Debug Train

Run sequence-level LSP-JEPA-Core debug training on the complete GSM8K train
split. This path trains only LSP-State with final latent-to-final teacher target
alignment:

```text
h_final <-> z_final
```

It does not add step-level losses, adapter expansion, SIM-CoT decoder CE, CODI
distillation, or intermediate CoT CE.

Data and generated artifacts default to the AutoDL data disk:

```text
/root/autodl-tmp/lsp_jepa/data/gsm8k_train.jsonl
/root/autodl-tmp/lsp_jepa/runs/pr11_mvp/full_seq_debug/
```

If the local dataset JSONL is missing, the script loads the configured full
GSM8K train split and writes a reusable JSONL copy under `/root/autodl-tmp`.

## Default 100-step run

```bash
python lsp_jepa/scripts/train_lsp_jepa_core.py \
  --config lsp_jepa/configs/core/lsp_seq_full_debug.yaml \
  --max-steps 100
```

Outputs:

```text
/root/autodl-tmp/lsp_jepa/runs/pr11_mvp/full_seq_debug/max_steps_100/metrics.jsonl
/root/autodl-tmp/lsp_jepa/runs/pr11_mvp/full_seq_debug/max_steps_100/summary.json
/root/autodl-tmp/lsp_jepa/runs/pr11_mvp/full_seq_debug/max_steps_100/config_snapshot.yaml
/root/autodl-tmp/lsp_jepa/runs/pr11_mvp/full_seq_debug/max_steps_100/checkpoints/
```

## 500-step debug run

Use a separate output directory so the 100-step acceptance artifacts are kept:

```bash
python lsp_jepa/scripts/train_lsp_jepa_core.py \
  --config lsp_jepa/configs/core/lsp_seq_full_debug.yaml \
  --max-steps 500 \
  --output-dir /root/autodl-tmp/lsp_jepa/runs/pr11_mvp/full_seq_debug/max_steps_500
```

## Acceptance Checks

The final `summary.json` records the PR11 checks:

- complete configured `max_steps`;
- complete training split loaded through a DataLoader;
- dataset cache and all outputs are under `/root/autodl-tmp`;
- finite `total_loss`, `lsp_loss`, `host_answer_ce`, and anti-collapse loss;
- EMA teacher has zero grad parameters;
- student backbone and predictor receive gradients;
- EMA drift is nonzero after optimizer updates;
- projected latent variance is nonzero;
- tail pairwise cosine is not near 1.0;
- teacher targets exclude answer tokens and answer prefixes.

The per-step `metrics.jsonl` also records batch sample IDs, final valid teacher
positions, target masks, gradient magnitudes, EMA drift, collapse diagnostics,
and answer-leakage flags.
