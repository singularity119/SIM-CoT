# Full Data Readiness Scan

`lsp_jepa/scripts/scan_lsp_full_data.py` performs the PR10-MVP full-dataset readiness check for LSP-JEPA-Core. It is a data and target-construction scan only. It does not instantiate CODI/SIM-CoT adapters, run training, run `backward()`, or call `optimizer.step()`.

## Default command

```bash
HF_ENDPOINT=https://hf-mirror.com python lsp_jepa/scripts/scan_lsp_full_data.py \
  --dataset-source gsm8k \
  --tokenizer-id openai-community/gpt2 \
  --batch-size 16 \
  --num-latent-steps 4 \
  --configured-max-seq-len 512 \
  --output-path lsp_jepa/runs/pr10_mvp/full_data_readiness_report.json
```

For fully offline runs, pass a local JSON/JSONL/Parquet file with `--data-path /path/to/gsm8k_train.jsonl`. The expected GSM8K train count is 7,473 samples.

## Report Contents

The JSON report includes sample totals, valid questions, valid CoT step distributions, answer-only final-step filtering rate, empty teacher-target rate and reasons, empty latent batch rate, teacher/student token-length distributions, final valid reasoning-step position distributions, answer leakage diagnostics, first 100 skipped sample ids, OOM risk, and recommended `max_seq_len` and batch size.

Teacher input is built as `Question + filtered CoT` through `build_teacher_inputs(..., exclude_answer_tokens=True, exclude_answer_prefix=True, filter_answer_steps=True)`. The final reasoning step is retained unless it is answer-only or answer-formatting such as `#### 42` or `The answer is 42`.

Student input length is measured as question tokens plus `<|start-latent|>`, `num_latent_steps` copies of `<|latent|>`, and `<|end-latent|>`. Ground-truth CoT tokens are never included in the student input length.

`answer_leakage_count` is a blocking leakage metric. It counts retained answer markers, answer prefixes, or answer-only tokens in filtered teacher steps. `answer_value_occurs_in_reasoning_count` is informational only, because GSM8K reasoning often contains the final numeric value in a valid final reasoning sentence.

## Acceptance Checks

For a clean full-data run:

- `acceptance.script_completed` is `true`.
- `acceptance.no_optimizer_step` is `true`.
- `acceptance.answer_leakage_default_zero` is `true`.
- `teacher_target.empty_teacher_target_rate` should be `0.0`; otherwise inspect `teacher_target.empty_teacher_target_reason_counts` and `skips.skipped_samples_first100`.
- `latent.latent_empty_batch_rate` should be `0.0` for normal GSM8K data with `num_latent_steps > 0`.

## Dependencies

The scanner supports the existing whitespace tokenizer without `transformers`. Loading from Hugging Face or `hf-mirror.com` uses `datasets`; optional non-whitespace tokenizers use `transformers`. These packages are listed in `lsp_jepa/requirements-mvp.txt`.
