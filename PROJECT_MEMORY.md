# LSP-JEPA 项目记忆与换机交接

最后整理：2026-09-02（Asia/Shanghai）。本文件面向“换电脑使用 Codex、实验继续留在原服务器”的场景。

## 1. 先读：这份交接确认了什么

- **本次已核实：** 本地 Git、已跟踪的代码/配置/报告，以及旧电脑保留的研究记录。整理前分支为 `lsp-jepa-v1-mvp`，HEAD 为 `f46810ffe258f63206f984ae1cea1eeb048d3cec`，工作区干净；GitHub 同名分支在推送前核对时也指向该提交。
- **本次未核实：** 远端任务是否仍运行、当前 GPU/进程、服务器最新代码、run 目录和 checkpoint 是否仍存在；没有 SSH、训练、评估或重启实验。
- **当前真正的下一步：** 在新电脑配置访问权限后，只读核对原服务器，再由用户确定接续哪个实验。本文不是新实验启动指令。
- **证据口径：** `[CODE]` 为本次代码/配置检查；`[HIST]` 为历史记录，本次未重查原始远端产物；`[INFER]` 为机制推断；`[TODO]` 为未完成核实或尚未验证的建议。

代码事实的适用基线是上述提交；后续远端或本地代码不同，必须重新核对。不要把“配置存在”“监控脚本存在”或“历史完成过”解释为今天仍有某任务在运行。

## 2. 仓库、目录和资料入口

仓库：`singularity119/SIM-CoT`。`origin` 是用户仓库；`upstream` 是 `InternLM/SIM-CoT`。接续分支是 `lsp-jepa-v1-mvp`，不是默认分支的同义词。

| 入口 | 用途与限制 |
| --- | --- |
| [AGENTS.md](AGENTS.md) / [lsp_jepa/AGENTS.md](lsp_jepa/AGENTS.md) | 工作和研究约束；包含设计要求，不代表每项功能已实现 |
| [LSP_JEPA_PROTOCOL.md](LSP_JEPA_PROTOCOL.md) | 方法协议与研究边界；实现情况仍要检查代码 |
| [README.md](README.md) | 上游 SIM-CoT 论文与用法，不是 LSP-JEPA 的实验结果报告 |
| [训练器](lsp_jepa/scripts/train_lsp_jepa_core.py) | 数据、Coconut 初始化、训练、EMA、评估、checkpoint 的主要执行路径 |
| [Coconut host](Coconut/coconut.py) / [adapter](lsp_jepa/adapters/coconut_lsp_adapter.py) | latent rollout 与状态抽取 |
| [target builder](lsp_jepa/core/target_builder.py) / [EMA](lsp_jepa/core/ema_teacher.py) | CoT 边界目标与 teacher 更新 |
| [losses](lsp_jepa/core/losses.py) / [metrics](lsp_jepa/core/metrics.py) / [anti-collapse](lsp_jepa/core/anti_collapse.py) | 损失、统计口径与正则化 |
| [序列训练说明](lsp_jepa/docs/FULL_SEQ_TRAIN_RUNBOOK.md) / [评估说明](lsp_jepa/docs/EVAL_LSP_CORE.md) | 早期 PR13/PR14 操作记录；不是当前 C2 run 的即用命令 |
| [历史短跑报告](lsp_jepa/docs/reports/LSP_SEQ_FULL_1K_REPORT.md) | 标题含 1K，但正文明确该 1K run 没有执行，详见第 5 节 |

旧电脑的项目工作目录比 Git 仓库大一层，Git 根是其下的 `SIM-CoT/`。仓库外另有 `lsp_jepa/`、启动/监控脚本副本、`ideaspark_run/`、`paper/` 和 `outputs/`。这些不是本仓库的一部分；不要把旧电脑父目录中的同名文件误认成当前受版本管理的代码。

本次只新增本文及修改 `AGENTS.md` 入口，没有迁入仓库外材料。已有受版本管理的启动/监控入口在 [lsp_jepa/launch_scripts](lsp_jepa/launch_scripts)。完整 HIFP 草案在旧电脑仓库外，本文只保留摘要和定位线索。

## 3. 长期研究目标与方法边界

研究问题：能否利用显式 CoT 的过程信息，训练只看到问题的模型在连续 hidden states 中完成有效推理，而不要求 student 逐 token 复述中间 CoT？SIM-CoT 是工程 host；LSP-JEPA 是这里研究的方法。

基本数据流：

```text
teacher: Q + CoT prefix -> stop-gradient contextual target z_i
student: Q -> Coconut latent rollout -> contextual state h_i
training: aligned-state loss + optional final-answer CE + anti-collapse
update: one backward/optimizer step per batch, then EMA teacher update
evaluation: student final-answer generation, no teacher branch or EMA update
```

现有协议的默认 Core 是 predictor-free、projection-free 的直接对齐。Core 不依赖 SIM-CoT decoder CE、CODI distillation 或 intermediate CoT token CE；student 不得接收 ground-truth CoT，teacher targets 排除 answer-only/answer-formatting 内容。保留正常的最后推理步，不能机械丢弃所有 final CoT step。

必须分别说明“研究方向”“当前实现”“某一次 run 的证据”。辅助损失下降不等于推理改善，单次 early-best 不等于稳定收益，某实现的负结果不等于整个方向被否定。

历史讨论曾建议测试 predictor、frozen teacher、normalized/cosine loss，但这些是候选对照，不是已经采纳的新默认方法。变更方法契约需用户明确批准；不能把建议悄悄写成当前实现。

## 4. C2 历史主线：配置与代码语义

这里的“C2”指命名配置 [lsp_step_trajectory_from_coconut_full_latent_c2_groupend_aw005_ce10_warmup30.yaml](lsp_jepa/configs/core/lsp_step_trajectory_from_coconut_full_latent_c2_groupend_aw005_ce10_warmup30.yaml)，不是所有 launcher 的默认模式。

### 4.1 配置快照 `[CODE]`

| 项目 | C2 配置中的值 |
| --- | --- |
| mode / host / backbone | `core / simcot / coconut` |
| 初始化 | 从已训练 Coconut 的 `best_full_latent_eval_accuracy.pt` 初始化，然后建立 EMA teacher |
| 模型 | 配置指向 `openai-community/gpt2` 的本地 snapshot；不要把 checkpoint 来源当作新架构 |
| 数据 | GSM8K-Aug 训练，期望 `385531` 样本；eval 配置期望 `1319` 样本 |
| student | `10` 个 `fixed_full` latents，steps 之间不 detach |
| teacher | EMA `0.995`，`last_2`，reasoning step boundary，`raw_hidden` |
| mapping | `grouped_step_end`，每组 `2` latents，最多 `5` 个 teacher groups，`uniform_first_last` |
| losses | raw `mse`；LSP 目标权重 `0.05`；answer CE 权重 `1.0`；variance 正则权重 `0.01` |
| LSP warmup | `max_steps` 的 `30%`；这是 LSP 权重渐增，不应混称 learning-rate warmup |
| 训练字段 | `max_steps=64260`，`batch_size=24`，`lr=5e-5`，`seed=0` |

这些是配置记录，不是本次确认的运行实参。它包含旧服务器的绝对路径、历史 run 名和 `3213` 的保存/诊断间隔；需要将 config snapshot、CLI overrides、实际样本数、world size 和日志中的 epoch/step 一起核对，不能由这些字段直接宣布“实际训练了多少 epoch”。

### 4.2 容易被历史描述误导的实现细节 `[CODE]`

1. **`normalize_latents: true` 不代表 C2 做了 normalized MSE。** 实际分派由 `loss.alignment` 决定；C2 为 `mse`。检查训练器的 `apply_config()`、`compute_step_trajectory_state_loss()` 与 `core/losses.py`。构建 snapshot 时的描述字段不能代替执行分支。
2. **`last_2` 是最后两组 hidden outputs 的均值，不是只取倒数第二层。** 见 target builder 的 `_select_contextual_hidden_states()`。此前模型核查将主线识别为 GPT-2 Small（12 blocks、hidden 768、12 heads）；新服务器仍应从实际加载模型配置确认，避免把旧规格套到后来换用的模型。
3. **student latent 是 latent-token 位置经过 transformer 后的 contextual state。** 不是把上一位置 hidden 回填到 input embedding 时的值。见 Coconut host、adapter 及 [post-latent 回归测试](lsp_jepa/tests/test_coconut_post_latent_states.py)。
4. **grouped mapping 不总是监督最后一个 latent。** group ends 为 1-based `[2,4,6,8,10]`。若有效 teacher steps 为 `K<5`，terminal target 对齐第 `2K` 个 latent；尾部仍受 answer CE/正则影响，但没有对应的 LSP pair。若 `K>=5`，才保留/选取 5 个 teacher targets 并到第 10 latent。见 `align_student_teacher_states()` 和 `select_teacher_step_indices()`。
5. **LSP/CE 比值比较的是加权项。** `lsp_to_ce_ratio = weighted_lsp_loss / weighted_host_answer_ce`，不含 anti-collapse；不是 raw loss 比值，也不是 LSP 占 total 的比例。见训练器主循环和 `safe_float_ratio()`。
6. **不要从全局 collapse 指标推断每个 step 都健康。** 当前 pairwise 统计 flatten 有效 batch×step vectors，混合跨样本与跨步骤差异；还要区分 aligned-state 与 all-latent 的统计范围。见 `core/metrics.py` 与训练器日志构造。

### 4.3 启动、评估及文档的已知陷阱 `[CODE]`

- [默认 StepTrajectory launcher](lsp_jepa/scripts/launch_step_trajectory_train.sh) 使用 `lsp_step_trajectory_full_train.yaml`，gate 要求 `one_to_one`，会拒绝 C2 的 `grouped_step_end`。不能拿它的命令当作 C2 复现实参，也不能为运行成功直接绕过 gate。
- [full35 启动脚本](lsp_jepa/launch_scripts/launch_gsm8k_aug_full35.sh) 没有显式传 C2 config，继承上述默认配置；它是另一条历史启动路径。
- 该 launcher 的 `--dry-run` 用法说明是“1-step smoke check with 32 samples”，不是纯打印命令。首次只读接续不要执行它。
- C2 配置用 test 数据在训练前/每 epoch 评估，并据相同 accuracy 选择 best checkpoint；不能把 best-test-selection 报成无偏的最终测试。后续严谨比较应先声明 validation 选择与独立 final test 协议。
- [早期独立 eval 配置](lsp_jepa/configs/core/eval_lsp_seq.yaml) 的本地文件名为 `gsm8k_train.jsonl`，虽然标签写 `eval_split: test`。必须检查实际读取的数据文件，不能只信 split 标签。
- 根 `AGENTS.md` 末尾包含通用 smoke 命令 `scripts/train.py`，当前 checkout 没有该入口。有效测试分布在 `tests/` 与 `lsp_jepa/tests/`；环境准备好后应按改动选择实际存在的测试。本次交接不重写历史规范或修改训练逻辑。

## 5. 重要历史结果与失败经验

### 5.1 早期 sequence closure `[HIST]`

仓库内的 [LSP_SEQ_FULL_1K_REPORT.md](lsp_jepa/docs/reports/LSP_SEQ_FULL_1K_REPORT.md) 实际总结 PR13 的 **100-step** gate 和 PR14 的 **20-sample** eval；正文明确请求的 1K run 未执行。

该报告记录短跑损失有限、未触发其 representation-collapse 标记，但 answer eval accuracy/exact-match 为 `0`、invalid-answer rate 为 `1`。这只支持“当时工程闭环部分成立、答案生成评估失败”，不支持方法有效。报告里的 predictor/projected-space 是早期实现描述，不能用来证明当前 C2 有 predictor。

PR9 synthetic debug 与 PR10 readiness 的小型产物已经在 Git 中，可分别查阅 [PR9 summary](lsp_jepa/runs/pr9_mvp/summary.json) 和 [PR10 readiness](lsp_jepa/runs/pr10_mvp/full_data_readiness_report.json)。它们不是 C2 的 10-latent GSM8K-Aug 结果；readiness 的 base GSM8K 样本统计也不能外推为 Aug 分布。

### 5.2 C2 完整训练的旧审计记录 `[HIST — 待远端复核]`

来源是此前项目审计记忆，以及旧电脑保留的 `ideaspark_run/lsp-jepa-identifiable-trajectory-alignment/current_evidence.md` 中的历史证据部分；该记录追溯到 2026-06-17 的审计讨论。原始远端 metrics/eval 本次未读取，旧审计全文也不包含在本次提交内。

历史 run 的目录由 C2 配置的 `training.output_dir` 给出，末级名称为：

```text
steptraj_from_coconut_full_latent_c2_groupend_aw005_ce10_warmup30_20260516_035534
```

| 旧审计报告项 | 约值；不是本次重测 |
| --- | --- |
| 初始化 baseline accuracy | 35.82% |
| early-best accuracy | 37.66% |
| post-warmup-best accuracy | 37.13% |
| final accuracy | 35.90% |
| post-warmup `lsp_to_ce_ratio` 中位数 | 4.35 |

保留的窄结论：该 predictor-free、raw-MSE、没有独立语义锚定的 EMA 实现未显示稳定、可明确归因的收益。不能把 early-best 当作最终结论，也不能据此否定所有 process-level latent supervision。

复核时至少读取 run 的 config snapshot、summary、训练 metrics、eval metrics、checkpoint metadata，并确认完整/抽样 eval、数据身份和 checkpoint 选择规则。上表不应进入新的正式结果表，直到原始证据复核完成。

### 5.3 需要带走的教训

- 参数名、YAML 注释、旧 runbook 和旧报告都可能滞后；配置读取、损失分派、实际 CLI 与 checkpoint metadata 才能说明某次运行做了什么。
- loss 权重数值小不等于该项梯度或加权损失小；同时报告 raw loss、weighted loss、CE 和 ratio。
- `[INFER]` raw-MSE 的尺度敏感、EMA target 漂移、短 trajectory 尾部缺监督都值得排查，但尚不能写成已确认的性能下降原因。
- teacher 没有独立语义锚定不等于已经证明 teacher 能力退化；需要 teacher-side eval/probe。
- 对照建议包括同起点的 `align_weight=0`、冻结 teacher、normalized/cosine alignment、显式 predictor ablation，以及离线 `(h_i,z_i)` 可预测性/R² 检查。它们不是已完成结果，也不代表允许现在启动。

## 6. 后续构想：HIFP `[TODO — 草案，不是已实现方法]`

旧电脑仓库外保留 `ideaspark_run/lsp-jepa-identifiable-trajectory-alignment/phase4/idea.std.zh.md`，题为“面向潜在推理的历史可识别功能进展”，方法名 History-Identifiable Functional Progress（HIFP）。

其思路是从复制 hidden coordinates 改为监督阶段引起的答案功能变化：冻结 teacher，在固定 `K=8` 候选答案上读取推理阶段加入前后的 log-probability 差；用双措辞视图一致性和样本级五折 cross-fitting 的历史可预测性筛选/加权；student 用原答案读取路径对齐这些 stage effects，不新增可训练辅助 head。

核心证伪比较是 ordered 与样本内循环错序目标的独立训练，保持目标集合、mask、起点和训练设置一致。若错序同样有效，就不能归因为正确的过程顺序监督。

这是设计记录，不是当前训练器已具备的功能。候选构造、视图有效性、validation 选择、预算匹配及对照定义仍有未决项；新颖性和效果也不能仅凭该草案确认。本次只带走上述摘要，不迁入完整文献检索、公式和草案资产；如要切换到这条研究线，先取回完整材料并重新确认范围。

## 7. 新电脑接续步骤

### 7.1 本地代码与访问

在选好的父目录中克隆指定分支，然后在 Codex 中打开克隆得到的 `SIM-CoT` 根目录：

```bash
git clone --branch lsp-jepa-v1-mvp https://github.com/singularity119/SIM-CoT.git
cd SIM-CoT
git status --short --branch
git log -5 --oneline
```

GitHub 私有仓库权限、原服务器 SSH/VPN、需要的技能/插件在新电脑单独配置。不要把 SSH 私钥、密码、token、整份个人 Codex 历史或机器专用配置提交进仓库。没有凭据时由用户提供安全的访问方式，不从旧日志寻找密码。

只换控制端，不在新电脑上重装原服务器训练环境，不自动向远端覆盖 checkout。依赖入口是 [requirements.txt](requirements.txt) 与 [lsp_jepa/requirements-mvp.txt](lsp_jepa/requirements-mvp.txt)，它们不是精确锁定的环境快照；真正复现应记录服务器 Python/PyTorch/Transformers/CUDA 版本。已有规范要求使用 `uv`、不要全局安装；本文没有执行安装。

### 7.2 首次只读核对清单

1. 确认用户要接续的服务器、代码目录、run；不能从历史 AutoDL 路径推断当前连接目标。
2. 检查远端 branch/commit、工作区差异、当前任务/进程及日志更新时间；分类为 queued、running、completed、failed、cancelled 或 unknown。
3. 核对实际启动命令、配置快照、数据/模型 revision 和初始化 checkpoint；保留远端未提交修改，不自动 pull、reset 或同步覆盖。
4. 核对 `summary.json`、`metrics.jsonl`、`evals/metrics.jsonl` 和 checkpoint metadata；进程不存在不自动等于训练成功。
5. 检查是否已有 checkpoint 监控器，避免启动重复写入者。先向用户报告核对结果，再按明确授权训练、恢复或改变监控。

### 7.3 Checkpoint 与监控陷阱

- Coconut baseline 的 `best_full_latent_eval_accuracy.pt` 与普通 best checkpoint 不同；需要确认保存时 latent curriculum 已达到目标阶段。实现入口是 [Coconut/run.py](Coconut/run.py)。
- LSP trainer 的 `best_eval_accuracy.pt`、`best_post_warmup_eval_accuracy.pt`、`best_total_loss.pt` 和 `latest.pt` 选择含义不同。不要把“最近”当作“最好”，也不要把最小训练 loss 当作最高答案 accuracy。
- [monitor_best_eval_checkpoint.py](lsp_jepa/launch_scripts/monitor_best_eval_checkpoint.py) 会复制 checkpoint、写状态；[watch_post_warmup_checkpoint.py](lsp_jepa/scripts/watch_post_warmup_checkpoint.py) 会更新 alias 和 metadata。即使传 `--once` 也不是纯只读检查。
- 数据与大 checkpoint 留在原服务器，Git 保存代码、配置和证据索引即可。`.gitignore` 中已有 run/output 忽略项，但部分早期小型 run 产物已被跟踪；忽略规则不代表这些旧文件不存在。
- 如确实需要 resume，应检查 trainer 的恢复路径及 checkpoint 保存的 student、EMA、optimizer、step、RNG/数据顺序状态；不要把 `initialization.coconut_checkpoint` 的权重初始化等同于完整断点恢复。

推荐给新设备 Codex 的第一条请求：

> 请先阅读 AGENTS.md 和 PROJECT_MEMORY.md。只读核对当前代码版本与我指定的原服务器任务、日志、配置和 checkpoint，区分已确认事实、历史记录与未知项，报告文档差异和下一步；先不要启动、恢复、停止训练或修改监控器。

## 8. 更新规则

- 最新状态写核实时间、branch/commit、run ID、状态及证据路径；没有重新检查的旧状态继续标为历史。
- 关键决策记下“选择、原因、替代方案、证据和适用范围”，保留负结果和废弃方案的原因，不堆积聊天全文。
- 代码链接使用仓库相对路径；服务器绝对路径作为历史定位信息，不当作跨机器可直接执行的命令。
- 事实冲突时，以实际实现/运行产物确定发生了什么，但不能用事实冲突自动获得修改方法或启动实验的权限；向用户说明差异。
- 本次交接仅迁移研究上下文，不代表全部旧电脑资料已备份，也不代表服务器实验已完成健康检查。
