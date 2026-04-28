# PR14 MVP Minimal Eval

`lsp_jepa/scripts/eval_lsp_jepa_core.py` evaluates an LSP-JEPA-Core sequence-level
checkpoint by generating only the final answer. It does not generate CoT, does
not score step-level behavior, does not construct a teacher branch, and does not
update EMA.

Default run:

```bash
python lsp_jepa/scripts/eval_lsp_jepa_core.py \
  --config lsp_jepa/configs/core/eval_lsp_seq.yaml
```

The default config points to the PR13 100-step checkpoint:

```text
/root/autodl-tmp/lsp_jepa/runs/pr13_mvp_full_seq_launch/max_steps_100/checkpoints/step_000100.pt
```

It also uses the locally cached GSM8K JSONL for a network-free MVP check. To use
an HF split instead, omit `data.eval_json` in the config or override it:

```bash
python lsp_jepa/scripts/eval_lsp_jepa_core.py \
  --checkpoint /path/to/step_000100.pt \
  --eval-split test \
  --limit-eval-samples 20
```

To evaluate a specific JSON/JSONL file:

```bash
python lsp_jepa/scripts/eval_lsp_jepa_core.py \
  --checkpoint /path/to/step_000100.pt \
  --eval-json /path/to/eval.jsonl \
  --limit-eval-samples 20
```

The script writes `metrics.json` containing:

- `accuracy`: numeric final-answer match using the extracted last number.
- `exact_match`: exact match after Coconut baseline-style answer normalization.
- `generated_length`: mean/min/max generated token count.
- `invalid_answer_rate`: fraction with no extractable numeric answer.
- `checkpoint_path`, `config_path`, and `git_commit_hash`.

Answer normalization follows the Coconut baseline evaluation behavior: split on
`#`, remove commas, then strip whitespace. The PR14 script also records whether
checkpoint state loaded strictly enough to run; missing or unexpected keys are
reported under `checkpoint_load_status`.
