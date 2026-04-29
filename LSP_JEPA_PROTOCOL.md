不再把它描述成 “CoDI + EMA” 或 “SIM-CoT 的变体”，而是定义为一个独立的 **EMA-guided JEPA-style latent state prediction framework**；同时允许它作为 adapter 插到 Coconut、CoDI、SIM-CoT 里。

---

# 1. LSP-JEPA v1.0 的最终研究目标

## 1.1 一句话定义

> **LSP-JEPA 用 EMA teacher 将 ground-truth CoT prefixes 编码为 causal latent reasoning states，并训练 student 在只看到 Question 的情况下，通过 latent rollout 达到相同的语义状态轨迹。**
> 

核心不是让 latent state 解码成 CoT 文本，而是让 latent state 接近 “模型看到对应 CoT 步骤之后” 的内部状态。这个定义和你们原始文档中的核心主张一致：CoT 的价值不在表面 token，而在它诱导出的语义状态转移；真正值得学习的是执行完第 (i) 步推理之后模型内部所处的 causal hidden state。

---

## 1.2 核心公式

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

---

# 2. 最终敲定的关键语义：默认是 `h_i ↔ z_i`

我们最终冻结的默认对齐不是：

```
h_i ↔ z_{i+1}
```

而是：

```
h_i ↔ z_i
```

也就是：

```
Student:
Q → latent update 1 → h1

EMA Teacher:
Q + CoT1 → z1

Loss:
Align(h1, z1)
```

这是最合理的时间定义。因为 $h_i$ 不是“当前状态”，而是 student 从 $h_{i-1}$出发完成第i次 latent update 后得到的预测状态。因此它已经是一个 predicted post-step state。对应 teacher 侧，它应该对齐 “EMA teacher 已经看到 $Q + CoT_{\le i}$ 之后” 的 post-step state。

这仍然是 JEPA-style latent state prediction。它不是预测原始 token 或 CoT 文本，而是在 embedding / hidden-state space 中建模 reasoning state。对我们来说，student latent rollout 的输出 $h_i$ 本身就是从 question state 出发预测出的第 $i$ 个 reasoning state，因此默认不再额外接 student-side predictor head。

---

# 3. v1.0 中的四种目标命名

为了后续代码和论文不混乱，建议固定这四个名字。

| 名称 | 形式 | 语义 | 地位 |
| --- | --- | --- | --- |
| **LSP-State** | $h_i \leftrightarrow z_i$ | 第 (i) 个 student latent state 对齐 teacher 完成第 (i) 步 CoT 后的状态 | **默认主方法** |
| **LSP-Transition** | $q_{next}(h_i) \leftrightarrow z_{i+1}$ | 从当前 latent state 额外预测下一 reasoning state | 可选辅助损失 |
| **LSP-State+Transition** | 同时使用上面两者 | 同时约束当前状态和下一状态预测能力 | 扩展实验 |
| **LSP-Skip** | $h_i \leftrightarrow z_{i+1}$ | 第 (i) 个 latent state 直接跳到下一步 teacher state | 只做 ablation |

默认论文方法应叫：

```
LSP-State
```

而不是 LSP-Transition 或 LSP-Skip。

---

# 4. Teacher target 最终定义

## 4.1 不取 raw embedding layer

我们不把 teacher target 定义为 embedding 层的 token embedding。原因是 raw token embedding 主要是 token identity + position，不包含完整上下文推理状态。它会把目标重新拉回 “预测下一步 CoT 文本 embedding”，与我们“不模仿 CoT 表面形式”的主张冲突。

默认 target 应是：

```
EMA_Model(Q + CoT_<=i)
在第 i 步 CoT boundary 位置的 contextual hidden state
```

也可以做 ablation：

```
target_layer:
  embedding
  early
  middle
  upper_middle
  last_2
  last
  multi_layer_mean
```

但 `embedding` 只作为 negative control / ablation，不作为主方法。

---

## 4.2 推荐 target 层

默认建议不要一开始就用最后一层。最后一层可能过于贴近 LM head 和 next-token / answer prediction，shortcut 风险更高。

推荐默认优先级：

```
last_2 或 upper_middle
→ middle
→ last
→ multi_layer_mean
```

最终通过 ablation 决定。

---

## 4.3 target position

默认：

```
step_last_token
```

即每个 CoT step 最后一个非特殊 token 的 hidden state。

可选：

```
step_mean
prefix_last_token
separator_token
multi_layer_mean
```

主方法默认：

```
z_i = hidden_state_at_end_of_CoT_step_i
```

这和项目文档里 “decoder-only 模型在 CoT step i 位置的 causal hidden state 表示模型看到 Q + step1 + ... + step_i 后，对后续推理方向的状态表示” 是一致的。

---

# 5. Final CoT step 与 answer leakage 的最终处理

这里要比之前的 AGENTS.md 更精确。

旧草稿里写的是：

```yaml
include_last_cot_step: false
```

这个太粗暴。v1.0 应改成：

```yaml
reasoning_step_filter: exclude_answer_only_steps
exclude_answer_tokens: true
exclude_answer_prefix: true
```

原因是：我们默认要做 (h_i \leftrightarrow z_i) 的 step-level state alignment。如果一律排除最后一个 CoT step，很多样本会丢失真正的最终 reasoning state。

但如果最后一个 step 只是：

```
Therefore, the answer is 42.
The answer is 42.
#### 42
```

它就不是 reasoning step，而是 answer-revealing step，应当过滤。

所以 v1.0 的规则是：

```
保留真正的 final reasoning step；
过滤 answer-only / answer-formatting / answer-prefix step；
answer tokens 默认永远不进入 teacher target。
```

包含 answer prefix、answer token、answer-only final step 的实验必须命名为：

```
leakage_ablation
shortcut_ablation
```

---

# 6. 训练反向传播逻辑最终敲定

默认训练方式：

> **每个 latent step 都计算 LSP loss，但把所有 step loss、anti-collapse loss、final answer CE 合成一个 total loss，然后一次 backward、一次 optimizer step。**
> 

不要每一步 rollout 都 optimizer step。

标准伪代码：

```python
with torch.no_grad():
    target_states = ema_teacher_encode(
        question_plus_cot,
        output_hidden_states=True,
    )
    z = gather_step_boundary_states(target_states)

student_outputs = student(
    question,
    latent_steps=T,
    output_latent_states=True,
)

h = student_outputs.latent_states

lsp_loss = align_trajectory(
    student_states=h,
    teacher_states=z.detach(),
    masks=masks,
    mapping="one_to_one",
)

answer_loss = ce(student_outputs.answer_logits, answer) if use_answer_ce else 0
reg_loss = anti_collapse(h) if use_reg else 0

loss = (
    lambda_lsp * lsp_loss
    + lambda_ans * answer_loss
    + beta * reg_loss
    + backbone_losses
)

loss.backward()
optimizer.step()
optimizer.zero_grad()
ema_teacher.update(student)
```

重要约束：

```
teacher target detach
student latent 不 detach
默认不做 per-step optimizer update
默认不做 detach_between_steps
EMA update 在 optimizer.step() 之后
```

LeWorldModel 的训练也是多步 embedding prediction loss 聚合后优化，而不是每个时间步单独更新；它用 next-embedding prediction loss 加 step-wise SIGReg regularizer 形成完整 objective。 SIM-CoT 的辅助监督也是训练时对每个 implicit latent step 产生 step-level supervision，但辅助 decoder 仅训练时存在、推理时移除；论文摘要中明确说它用 auxiliary decoder 对齐每个 implicit token 与对应 explicit reasoning step，并在推理时移除 decoder。([arXiv](https://arxiv.org/abs/2509.20317?utm_source=chatgpt.com))

---

# 7. LSP-JEPA-Core 与 LSP-JEPA-Adapter 的最终区分

## 7.1 LSP-JEPA-Core

这是论文主方法。

```
Input:
  Question only

Teacher:
  EMA(Q + CoT prefixes) → z_1...z_K

Student:
  Q → h_1...h_T

Loss:
  LSP-State alignment
  + optional answer CE
  + anti-collapse regularizer
```

不允许：

```
intermediate CoT token CE
SIM-CoT auxiliary decoder CE
CODI distillation loss
ground-truth CoT tokens into student
```

只有这个版本能支撑主 claim：

> latent reasoning 不模仿 CoT 文本，而是在 latent space 中对齐 CoT 诱导的语义状态轨迹。
> 

---

## 7.2 LSP-JEPA-Adapter

这是工程扩展线。

LSP 可以作为额外 loss 插到：

```
Coconut + LSP-JEPA
CoDI + LSP-JEPA
SIM-CoT + LSP-JEPA
```

SIM-CoT 本身已经是 plug-and-play step-level supervision module，并且官方仓库给出了 Coconut + SIM-CoT 的训练和评估入口。([GitHub](https://github.com/InternLM/SIM-CoT?utm_source=chatgpt.com)) 它的监督目标是 auxiliary decoder 生成 textual reasoning step；我们的 LSP adapter 则是 hidden-vector alignment。

Adapter 总损失可以写成：

```
Coconut + LSP:
  L = L_answer_CE
    + lambda_lsp * L_LSP
    + beta * L_reg

CoDI + LSP:
  L = L_answer_CE
    + lambda_codi * L_CODI
    + lambda_lsp * L_LSP
    + beta * L_reg

SIM-CoT + LSP:
  L = L_answer_CE
    + lambda_simcot * L_decoder_step_CE
    + lambda_lsp * L_LSP
    + beta * L_reg
```

这组实验的意义不是定义主方法，而是证明：

```
EMA latent state supervision 是一个可插拔模块。
```

---

# 8. v1.0 项目架构

建议仓库结构如下。

```
LSP_JEPA/
  src/
    models/
      lsp_jepa_model.py
      ema_teacher.py
      answer_readout.py
      adapters/
        coconut_lsp_adapter.py
        codi_lsp_adapter.py
        simcot_lsp_adapter.py

    data/
      cot_trajectory_dataset.py
      prefix_builder.py
      step_segmenter.py
      answer_normalizer.py
      leakage_filter.py

    targets/
      trajectory_targets.py
      target_mapping.py

    losses/
      alignment.py
      sigreg.py
      vicreg.py
      coverage.py
      temporal_geometry.py

    trainers/
      lsp_jepa_trainer.py
      ema_update.py
      logging_hooks.py

    metrics/
      collapse.py
      rollout.py
      trajectory.py
      leakage.py
      step_bucket_eval.py

  configs/
    core/
      gsm8k_lsp_state_seq.yaml
      gsm8k_lsp_state_step.yaml
      gsm8k_lsp_no_answer_ce.yaml
      gsm8k_lsp_small_answer_ce.yaml
      gsm8k_lsp_sigreg.yaml

    adapters/
      coconut_lsp_state.yaml
      codi_lsp_state.yaml
      simcot_lsp_state.yaml

    ablations/
      ema_decay.yaml
      target_layer.yaml
      target_pooling.yaml
      latent_steps.yaml
      mapping_strategy.yaml
      anti_collapse.yaml
      transition_auxiliary.yaml
      leakage_targets.yaml

  tests/
    test_ema_no_grad.py
    test_ema_update.py
    test_prefix_builder.py
    test_step_boundaries.py
    test_trajectory_targets.py
    test_no_answer_leakage_by_default.py
    test_alignment_masks.py
    test_single_backward_contract.py
    test_sigreg_shapes.py
    test_baseline_flags_off.py
```

关键原则：

```
不要直接大改 CODI internals。
不要让 SIM-CoT decoder 变成主方法依赖。
LSP-JEPA 单独成包。
Adapter 只负责从 host backbone 取 latent_states 和 backbone_losses。
```

---

# 9. v1.0 实验逻辑

## 9.1 阶段 0：复现 baseline

先跑：

```
Direct Answer
Explicit CoT
Coconut
CODI
SIM-CoT
```

这些是必要对照。SIM-CoT 论文报告其能提升 Coconut、CODI 等 implicit CoT baseline，并用 auxiliary decoder 提供 step-level supervision。([arXiv](https://arxiv.org/abs/2509.20317?utm_source=chatgpt.com))

---

## 9.2 阶段 1：LSP-State Sequence-level

最小验证：

```
Student final latent h_T
↔
Teacher final valid reasoning state z_K
```

目标：

```
验证 EMA teacher 的 CoT-induced hidden state 是否能作为 latent reasoning target。
```

成功信号：

```
accuracy > Direct Answer
alignment loss 下降
collapse metrics 正常
answer tokens excluded 时仍有效
```

---

## 9.3 阶段 2：Sparse Step-level

先对齐：

```
first / middle / final valid reasoning step
```

或：

```
odd steps only
```

目标：

```
验证局部 step supervision 是否比 sequence-only 更有效。
```

---

## 9.4 阶段 3：Full Step-level LSP-State

默认：

```
h_i ↔ z_i
```

当 (T = K)：

```
h1 ↔ z1
h2 ↔ z2
...
hK ↔ zK
```

目标：

```
验证逐步 latent state alignment 是否增强多步推理。
```

这正是原始 Looped Transformer 想法中的中间隐状态监督：looped/recurrent hidden states $h_1,\ldots,h_T$ 可以被看成连续空间中的推理链，每个隐状态功能上等价于一个 CoT 中间步骤，只是发生在 hidden space 中。

---

## 9.5 阶段 4：latent steps 与 CoT steps 解耦

实验：

```
T_latent ∈ {1, 2, 4, 8, 16}
K_CoT = variable
mapping ∈ {uniform, attention, soft_dtw}
```

目标：

```
验证 latent reasoning 不是机械模仿 CoT 步数。
```

这是最强 claim：

```
T_latent < K_CoT 时仍能接近或超过 T=K 的性能。
```

---

## 9.6 阶段 5：Anti-collapse 系统实验

必须跑：

```
none
SIGReg
VICReg
variance-only
covariance-only
```

LeWorldModel 的关键启发是：单纯 embedding prediction 容易 collapse，因此它用 SIGReg 约束 latent embeddings 接近 isotropic Gaussian；它的完整 objective 是 prediction loss + SIGReg。 我们不一定照搬到 raw LLM hidden state 上，但应把 SIGReg/VICReg 用在 projected JEPA latent space 上。

监控：

```
per-dim variance
effective rank
singular values
pairwise cosine
pairwise L2
student-teacher cosine
target coverage
EMA-student parameter drift
alignment loss vs accuracy
```

---

## 9.7 阶段 6：Adapter 实验

跑：

```
Coconut
Coconut + LSP-State

CODI
CODI + LSP-State

SIM-CoT
SIM-CoT + LSP-State

Coconut + SIM-CoT
Coconut + SIM-CoT + LSP-State

CODI + SIM-CoT
CODI + SIM-CoT + LSP-State
```

目的：

```
验证 LSP-JEPA 是 plug-and-play latent supervision。
```

但论文主结果要把 Core 和 Adapter 分开汇报。

---

## 9.8 阶段 7：Looped / recurrent latent rollout

后期再加：

```
latent_tokens → recurrent → looped
```

Looped 版本：

```
Q → h0
h0 → shared block → h1
h1 → shared block → h2
...
hT → answer
```

测试：

```
T_train fixed
T_test sweep: 1,2,4,8,16,32
```

观察：

```
accuracy vs latent steps
overthinking
hard/easy bucket
latent trajectory straightness
collapse
```

---

# 10. 必做 ablation 清单

## 10.1 Teacher target

```
target_layer:
  embedding
  early
  middle
  upper_middle
  last_2
  last
  multi_layer_mean

target_pooling:
  step_last_token
  step_mean
  prefix_last_token

target_space:
  raw_hidden
```

`embedding` 只做 ablation。
`projected_hidden` 只允许在显式 `projection_head_ablation` 中使用，不能作为 Core 默认方法。

---

## 10.2 Objective

```
LSP-State
LSP-Transition
LSP-State+Transition
LSP-Skip
```

默认主方法：

```
LSP-State
```

---

## 10.3 Mapping

```
sequence
one_to_one
sparse
uniform
attention
soft_dtw
```

---

## 10.4 Loss

```
mse
normalized_mse
cosine
smooth_l1
infonce
```

默认建议：

```
normalized_mse 直接对齐 h_i 与 stop_grad(z_i)
```

---

## 10.5 EMA

```
ema_decay:
  0.99
  0.995
  0.999

update:
  full_model
  trainable_only
```

---

## 10.6 Answer CE

```
answer_ce_weight:
  0
  0.05
  0.1
  0.3
  1.0
```

必须区分：

```
pure latent
small answer CE
normal answer CE
```

---

## 10.7 Leakage

```
answer tokens excluded
answer prefix included
answer tokens included
final answer-like step included
wrong CoT
shuffled CoT
```

如果 include answer token 后效果暴涨，但 exclude 后下降，说明 shortcut 风险很高。
