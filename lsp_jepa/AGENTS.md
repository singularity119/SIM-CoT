# AGENTS.md — 基于 SIM-CoT Host 的 LSP-JEPA v1.2

## 项目使命

本仓库使用 SIM-CoT 代码库作为工程 host，同时将 LSP-JEPA 实现为一个独立研究方法。

LSP-JEPA 将 ground-truth chain-of-thought traces 视为语义 latent trajectories，而不是需要模仿的文本。EMA teacher 将 `Question + CoT_<=i` 编码为 stop-gradient contextual hidden-state targets。只看到 `Question` 的 student rollout 出 latent reasoning states，并将其对齐到这些 targets。

SIM-CoT 可以用于 datasets、tokenization、prompts、baseline implementations、training/evaluation harnesses 和 latent-state extraction。LSP-JEPA-Core 不能依赖 SIM-CoT auxiliary textual decoding、CODI single-token distillation 或 intermediate CoT token CE。

默认方法：

```text
Teacher target:   z_i = target_head_ema(EMA_Model(Q + CoT_<=i))[step_i_boundary]  # no grad
Student rollout:  h_i = student(Q).latent_states[i]
Main loss:        LSP-State = Align(h_i, stop_grad(z_i))
Total loss:       lambda_lsp * LSP-State
                + beta * AntiCollapse(student_latents)
                + optional eta * AnswerReadoutCE
                + optional host_backbone_losses only in adapter experiments
EMA update:       theta_ema <- decay * theta_ema + (1 - decay) * theta_student
```

把 SIM-CoT 当作 harness，而不是方法本身。把 LSP-JEPA 当作方法，而不是 SIM-CoT decoder variant。

## 实现策略

使用 hybrid layout：

1. 保留 upstream SIM-CoT/Coconut/CODI code 作为 host 和 baseline layer。
2. 新增 `lsp_jepa/`，作为具有最小 host assumptions 的 method-specific package。
3. 通过轻量 adapters 将 host models 接入 LSP-JEPA。
4. 当 `use_lsp_jepa=false` 时，保留 baseline behavior。
5. 通过 experiment mode 区分科学主张：`core`、`adapter`、`baseline` 或 `ablation`。

推荐 package layout：

```text
lsp_jepa/
  core/
    ema_teacher.py              # EMA model manager
    target_builder.py           # Q+CoT prefix construction 与 step-boundary target extraction
    latent_interface.py         # latent states、masks、answer logits、host losses 的 dataclasses/protocols
    losses.py                   # normalized MSE、cosine、SmoothL1、InfoNCE
    mapping.py                  # sequence、one_to_one、sparse、uniform、attention、soft_dtw
    anti_collapse.py            # SIGReg、VICReg、variance/covariance fallbacks
    metrics.py                  # collapse 与 trajectory diagnostics
    trainer_mixin.py            # 计算 LSP losses 并记录 metrics；不包含 host-specific assumptions

  adapters/
    base.py                     # HostAdapter protocol
    simcot_host.py              # SIM-CoT-hosted training loop 的 adapter
    lsp_core_adapter.py          # 基于 SIM-CoT host utilities 的 pure LSP student
    coconut_lsp_adapter.py      # 抽取 Coconut latent states 和 answer logits
    codi_lsp_adapter.py         # 抽取 CODI latent states 和 host losses
    simcot_lsp_adapter.py       # 抽取 SIM-CoT latent states 和 optional decoder loss

  configs/
    core/
      gsm8k_lsp_state_seq.yaml
      gsm8k_lsp_state_step.yaml
      gsm8k_lsp_no_answer_ce.yaml
      gsm8k_lsp_small_answer_ce.yaml
      gsm8k_lsp_sigreg.yaml
    adapters/
      coconut_plus_lsp.yaml
      codi_plus_lsp.yaml
      simcot_plus_lsp.yaml
    ablations/
      target_layer.yaml
      latent_steps.yaml
      mapping_strategy.yaml
      anti_collapse.yaml
      leakage_targets.yaml
      lsp_transition_aux.yaml

  tests/
    test_baseline_flags_off.py
    test_host_adapter_contract.py
    test_ema_no_grad.py
    test_ema_update.py
    test_target_builder_no_answer_leakage.py
    test_step_boundary_determinism.py
    test_alignment_masks.py
    test_single_backward_contract.py
    test_sigreg_shapes.py
    test_core_mode_zero_host_losses.py
```

## 核心公式

训练样本：

```
(Q, CoT steps C1...CK, Answer A)
```

EMA teacher trajectory：

```
z_i = EMA_Model(Q + C_<=i) 在第 i 个 CoT step boundary 的 contextual hidden state
```

Student latent rollout：

```
h_i = Student(Q) 完成第 i 次 latent reasoning update 后的 latent state
```

默认主损失：

$$
\mathcal{L}_{LSP}=\sum_i d\left(h_i,\mathrm{sg}(z_i)\right)
$$

完整训练目标：

$$
\mathcal{L}_{total}=\lambda_{lsp}\mathcal{L}{LSP}+\lambda{ans}\mathcal{L}{ans}^{CE}+\beta\mathcal{L}{anti-collapse}+\mathcal{L}_{backbone}
$$

其中：

```
h_i:
  student latent rollout 在第 i 次 latent reasoning update 后得到的 latent state

z_i:
  EMA teacher 看到 Q + C_<=i 后，在第 i 个 CoT step boundary 的 contextual hidden state

sg:
  stop-gradient

L_ans:
  optional final answer CE

L_backbone:
  optional Coconut / CoDI / SIM-CoT 原始损失，仅 adapter 实验使用
```

默认 Core 方法不使用 student-side predictor / projection head，也不使用
teacher-side target projection head。主对齐是直接的：

```
h_i ↔ stop_grad(z_i)
```

如果后续为了 ablation 重新加入 projection head，必须显式命名为
`projection_head_ablation`，不能作为默认主方法。

EMA 更新：

$$
\bar{\theta}\leftarrow\alpha \bar{\theta}+(1-\alpha)\theta
$$


## 实验模式

每个 run 都必须声明 mode。

```yaml
experiment:
  mode: core        # core | adapter | baseline | ablation
  host: simcot      # simcot | standalone
  backbone: lsp     # lsp | coconut | codi | simcot
  use_lsp_jepa: true
```

### `core`

Core mode 是主科学方法。它可以复用 SIM-CoT data loaders、tokenizers、prompts、training loops 和 evaluation scripts。它不能使用 SIM-CoT decoder CE、CODI distillation、intermediate CoT token CE，student input 也不能包含 ground-truth CoT tokens。

只有 teacher targets 不包含 answer tokens 的 core-mode runs，才能用于支持 LSP-JEPA-Core claim。

### `adapter`

Adapter mode 将 LSP-JEPA 作为额外 latent-space loss 加到已有 implicit-CoT backbone 上。

示例：

```text
Coconut + LSP:  L = L_answer_CE + lambda_lsp * L_LSP + beta * L_reg
CODI + LSP:     L = L_answer_CE + lambda_codi * L_CODI + lambda_lsp * L_LSP + beta * L_reg
SIM-CoT + LSP:  L = L_answer_CE + lambda_simcot * L_decoder_CE + lambda_lsp * L_LSP + beta * L_reg
```

Adapter runs 适合支持 plug-and-play claims。它们不能作为 core method 的唯一证据。

### `baseline`

Baseline mode 运行未修改的 Direct Answer、Explicit CoT、Coconut、CODI 或 SIM-CoT。

### `ablation`

Ablation mode 用于 target layers、leakage targets、transition losses、mapping strategies、anti-collapse variants、latent step sweeps 和 optimizer/gradient-contract stress tests。

## HostAdapter Contract

LSP-JEPA core 只能调用 adapter interfaces，不能调用 host internals。

```python
@dataclass
class HostModelOutput:
    latent_states: torch.Tensor        # [batch, T, dim]
    latent_mask: torch.Tensor          # [batch, T]
    answer_logits: torch.Tensor | None
    answer_labels: torch.Tensor | None
    host_losses: dict[str, torch.Tensor]
    debug: dict[str, Any]
```

每个 adapter 都必须实现：

```python
class HostAdapter(Protocol):
    def prepare_batch(self, raw_batch) -> dict: ...
    def forward_student(self, model, batch, *, output_latent_states: bool) -> HostModelOutput: ...
    def build_teacher_input(self, batch) -> dict: ...
    def extract_answer_loss(self, host_output: HostModelOutput, batch) -> torch.Tensor | None: ...
```

## 不可违反的约束

1. LSP-JEPA-Core 不能实现成 CODI objective。
2. LSP-JEPA-Core 不能依赖 SIM-CoT auxiliary textual decoder CE。
3. LSP-JEPA-Core 不能使用 intermediate CoT token CE。
4. Core mode 下，student 不能接收 ground-truth CoT tokens。
5. EMA teacher 永远不能接收 gradients。
6. EMA teacher 不能被加入 optimizer 或 scheduler parameter groups。
7. EMA updates 只能发生在 `optimizer.step()` 之后、下一次 forward pass 之前。
8. Teacher trajectory targets 默认必须排除 ground-truth answer tokens 和 answer prefixes。
9. Teacher targets 必须使用 contextual hidden states，而不是 raw token embeddings，除非是 explicit ablations。
10. 默认 objective 是 LSP-State：latent step `h_i` 对齐 `z_i = EMA(Q + CoT_<=i)`，不是 `z_{i+1}`。
11. Padding、missing steps、invalid steps 和 sampled-out steps 对 losses 和 metrics 的贡献必须为零。
12. 当 `use_lsp_jepa=false` 时，SIM-CoT/Coconut/CODI behavior 必须保留。
13. 不要使用 absolute paths、fixed GPU IDs 或 hard-coded dataset/model names。
14. 每个 result table 都必须说明是否使用 answer CE、SIM-CoT decoder CE、CODI distillation、intermediate CoT token CE，以及是否使用 answer-token-containing targets。

## 必需 Config Fields

```yaml
experiment:
  mode: core              # core | adapter | baseline | ablation
  host: simcot            # simcot | standalone
  backbone: lsp           # lsp | coconut | codi | simcot
  use_lsp_jepa: true

teacher:
  ema_decay: 0.995
  update_trainable_only: true
  output_hidden_states: true
  target_layer: last_2    # embedding only allowed in explicit ablations
  target_pooling: step_last_token   # step_last_token | step_mean | prefix_last_token
  target_space: raw_hidden          # default Core target space
  exclude_answer_tokens: true
  exclude_answer_prefix: true
  reasoning_step_filter: exclude_answer_only_steps

student:
  latent_arch: latent_tokens        # latent_tokens | recurrent | looped
  num_latent_steps: 4
  latent_dim: null                  # null means model hidden size
  normalize_latents: true
  detach_between_steps: false       # true only for explicit ablation

lsp_objective:
  type: state                       # state | transition | state_plus_transition | skip
  state_offset: 0                   # default h_i <-> z_i
  transition_weight: 0.0

mapping:
  strategy: sequence                # sequence | one_to_one | sparse | uniform | attention | soft_dtw
  sparse_positions: [first, middle, final]
  attention_coverage_weight: 0.0

loss:
  alignment: normalized_mse         # mse | normalized_mse | cosine | smooth_l1 | infonce
  align_weight: 1.0
  anti_collapse: sigreg             # none | sigreg | vicreg
  anti_collapse_weight: 0.05
  answer_readout_weight: 0.0        # pure latent by default

host_losses:
  host_answer_ce_weight: 0.0        # may be nonzero only when intentionally configured
  codi_distill_weight: 0.0          # must be zero for core
  simcot_decoder_weight: 0.0        # must be zero for core
  intermediate_cot_ce_weight: 0.0   # must be zero for core

logging:
  log_latent_metrics: true
  log_alignment_matrices: false
  log_target_leakage_checks: true
  log_host_losses: true
```

## Teacher Target Construction

Targets 来自 ground-truth CoT prefixes：

```text
prefix_0 = Question
prefix_1 = Question + CoT step 1
prefix_2 = Question + CoT step 1 + CoT step 2
...
prefix_K = Question + CoT step 1 + ... + CoT step K
```

对 decoder-only models，优先在 `Question + full CoT` 上做一次 teacher forward，然后在 step boundaries 处 gather hidden states。Causal masking 使 boundary state `i` 等价于编码 `Question + CoT_<=i`。

默认 target selection：

- 使用 CoT prefix states，不使用 answer states。
- 使用 step boundary 处的 contextual hidden states，或 step span 的 mean-pooled hidden states。
- 默认排除 answer tokens 和 answer prefixes。
- 过滤 answer-only 或 answer-formatting steps，例如 `The answer is:` 或 `####`。
- 不要机械地丢弃 final CoT step。只有当它是 answer-only 或 answer-formatting 时才丢弃。
- 当 step boundaries 缺失时，不要静默 fallback 到 answer-prefix targets；应把样本标记为 invalid，或使用配置的 sequence-level fallback。

Raw input embeddings 只允许作为 negative-control ablations。它们不是 LSP-JEPA-Core 的有效默认 targets。

## Student Latent Rollout

Core mode 下 student 只看到问题。它不能看到 ground-truth CoT tokens。

支持的 rollout modes：

1. `latent_tokens`：在问题后追加 learned latent slots，并在这些 slots 上收集 hidden states。
2. `recurrent`：编码问题，并用共享模块迭代更新 latent states。
3. `looped`：重复应用同一个 transformer block 或 block group，产生 latent trajectory。

Looped/recurrent 路径是后续阶段扩展。最小 core experiments 从 latent tokens 开始。

## LSP Objectives

一致使用以下名称：

```text
LSP-State:
  h_i <-> z_i = EMA(Q + CoT_<=i)
  Default main objective.

LSP-Transition:
  q_next(h_i) <-> z_{i+1}
  Optional auxiliary objective.

LSP-State+Transition:
  Use both state and transition losses.
  Extension experiment.

LSP-Skip:
  h_i <-> z_{i+1}
  Ablation only; never default.
```

## Alignment Losses and Mapping

所有 alignment losses 都必须支持 masks：

```python
loss = align(pred_states, target_states, pred_mask, target_mask, mapping_strategy)
```

默认 alignment：

```python
pred = normalize(student_states)
target = normalize(stop_grad(teacher_target_states))
loss = masked_mse(pred, target, mask)
```

支持的 mapping strategies：

- `sequence`：final predicted latent 对齐 final valid teacher target。
- `one_to_one`：latent step `i` 对齐 teacher step `i`。
- `sparse`：对齐 first/middle/final 或配置指定的 sparse steps。
- `uniform`：均匀采样 teacher steps，使其匹配 latent step count。
- `attention`：student states attend to teacher states；增加可选 coverage regularizer。
- `soft_dtw`：可微的 monotonic trajectory alignment。

## Anti-Collapse Losses and Metrics

至少实现 projected JEPA latent vectors 上的 SIGReg-style regularization。默认不要强迫 raw LLM hidden states 匹配 isotropic Gaussian。

同时支持 VICReg 或 variance/covariance fallbacks。

记录以下 diagnostics：

- per-dimension variance mean and minimum；
- effective rank；
- singular value spectrum summary；
- batch-wise pairwise cosine similarity；
- pairwise L2 distance；
- student-teacher cosine similarity；
- target coverage under attention alignment；
- valid target fraction and invalid trajectory counts；
- EMA-student parameter drift；
- alignment loss vs answer accuracy。

如果 alignment loss 下降，但 accuracy 保持不变且 effective rank 坍缩，则将该 run 标记为 failed representation-learning run。

## Optimization Contract

Step-level supervision 不等于 step-level optimizer updates。

先计算每个 step 的 LSP loss，再把所有 LSP losses、optional answer CE、anti-collapse loss 和 host losses 聚合为一个 total loss，然后每个 batch 只调用一次 backward pass 和一次 optimizer step。

默认：

```text
teacher target: detach
student latents: no detach
latent rollout loop: no optimizer.step()
loss.backward()
optimizer.step()
optimizer.zero_grad()
ema_teacher.update(student)
```

Per-step optimizer updates、detached student rollout 或 truncated BPTT 只允许作为 explicit ablations。

## 可复用的 SIM-CoT Host Components

允许复用：

- dataset loading 和 GSM8K-style CoT/answer preprocessing；
- tokenizer 和 prompt formatting utilities；
- Direct Answer、Explicit CoT、Coconut、CODI 和 SIM-CoT 的 baseline implementations；
- training/evaluation scripts 和 result aggregation utilities；
- Coconut/CODI/SIM-CoT 中的 latent-state extraction points；
- 如果 SIM-CoT latent instability diagnostics 被做成 host-agnostic，则可以复用。

不要作为 core-method dependencies 复用：

- SIM-CoT auxiliary decoder CE；
- SIM-CoT textual-step reconstruction 作为 required training target；
- CODI single designated-token distillation 作为 required objective；
- intermediate CoT token CE；
- answer-prefix hidden target；
- 任何 CODI-specific data-cleaning workaround，除非它被复现为命名 baseline 或 ablation。

## Merge 前必需测试

- `use_lsp_jepa=false` 保持 baseline host behavior。
- Host adapters 返回 shape 和 masks 正确的 `HostModelOutput`。
- EMA teacher 不接收 gradients，且不在 optimizer parameter groups 中。
- EMA update 数值上匹配 `decay * ema + (1 - decay) * student`。
- Teacher target builder 默认排除 answer tokens 和 answer prefixes。
- Final reasoning steps 会被保留，除非它们被分类为 answer-only 或 answer-formatting。
- 对固定 tokenized input，step boundary extraction 是 deterministic 的。
- Alignment losses 会把 masked positions 清零。
- 默认 LSP objective 对齐 `h_i` 到 `z_i`，而不是 `z_{i+1}`。
- Core mode 中 SIM-CoT decoder CE、CODI distillation 和 intermediate CoT CE 均为零。
- 默认 training loop 每个 batch 只执行一次 optimizer step，且 latent rollout 内部不执行 optimizer step。
- Student latents 默认不会在 steps 之间 detach。
- SIGReg/VICReg losses 在 normal batches 上是 finite 的，并且对 small batches 安全。

## 必需实验

Core experiments：

1. Direct Answer baseline。
2. Explicit CoT baseline。
3. Coconut baseline。
4. CODI baseline。
5. SIM-CoT baseline。
6. LSP-State sequence-level alignment。
7. LSP-State sparse step-level alignment。
8. LSP-State full step-level alignment。
9. LSP-State + SIGReg/VICReg。
10. LSP-State with no answer CE。
11. LSP-State with small answer CE。
12. Latent step count sweep。
13. `T_latent != K_CoT` with uniform、attention 或 soft-DTW mapping。

Adapter experiments：

1. Coconut + LSP-State。
2. CODI + LSP-State。
3. SIM-CoT + LSP-State。
4. Coconut + SIM-CoT + LSP-State。
5. CODI + SIM-CoT + LSP-State。

Ablations：

- EMA decay：`0.99`、`0.995`、`0.999`。
- target layer：`embedding`、`early`、`middle`、`upper_middle`、`last_2`、`last`、`multi_layer_mean`。
- target pooling：`step_last_token`、`step_mean`、`prefix_last_token`。
- alignment loss：`mse`、`normalized_mse`、`cosine`、`smooth_l1`、`infonce`。
- anti-collapse：`none`、`sigreg`、`vicreg`、variance-only、covariance-only。
- latent rollout architecture：`latent_tokens`、`recurrent`、`looped`。
- number of latent steps：`1`、`2`、`4`、`8`、`16`。
- leakage：answer tokens 默认 excluded；只在命名 leakage ablations 中 include。
- objective：`state`、`transition`、`state_plus_transition`、`skip`。

## Research Reporting Discipline

每个 run 和 result table 都必须报告：

- experiment mode：core、adapter、baseline 或 ablation；
- host 和 backbone；
- 是否使用 answer CE 及其 weight；
- 是否使用 SIM-CoT decoder CE 及其 weight；
- 是否使用 CODI distillation 及其 weight；
- 是否使用 intermediate CoT token CE；
- teacher targets 是否包含 answer tokens 或 answer prefixes；
- EMA decay 和 update scope；
- target layer、target pooling 和 target space；
- latent step count 和 mapping strategy；
- anti-collapse regularizer 和 weight；
- collapse metrics 和 invalid trajectory counts。

只有当 SIM-CoT decoder CE、CODI distillation、intermediate CoT token CE 都为零，且 teacher targets 不包含 answer tokens 时，对应 run 才能用于支持 LSP-JEPA-Core claim。

## Coding Style

- 优先使用小型、可测试模块，而不是大规模重写 trainer。
- 每个新行为都放在显式 config flags 后面。
- 对 structured configs 和 outputs 使用 typed dataclasses。
- 除非绝对必要，不要修改 baseline public APIs。
- 记录 invalid samples 和 failed target extraction；不要静默丢弃。
- Comments 聚焦于不直观的 research logic，而不是通用 Python 行为。
- 不要隐藏 failed examples、collapse runs 或 invalid trajectory counts。
