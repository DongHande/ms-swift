# 20261118 loss scale 迁移记录

## 状态与基线

截至 2026-11-19，本候选版本已完成真实 Qwen3.5 template/tokenizer 三方对照、六组角色权重检查、84 组注入往返及选定回归测试。**不是已完成 35B 模型训练、GPU/Megatron/多节点验收的发布版本。**20261118 的静态检查与环境阻塞记录保留为历史，当前结果见下一节。

- 工作仓库：`/data_train/vectorchen/github/ms-swift-codebuddy`。
- 新分支：`codebuddy/loss-scale-20261118`。
- 官方基线：`1997fa6f4192d41769ea17b43eb811f943ed2aed`，2026-11-18。
- 源分支：`origin/feat/agent-role-loss-scale-v20260908`。
- 源 tip：`3b3f8d30314bec40f6dd808e2400188749d25881`。
- 共同祖先：`75baafb27f3f2b9676f0fa1b6bfe4dc5715698a4`，2026-09-07。
- 源增量：5 个提交、22 个文件、2501 行新增、26 行删除。
- 原 fork `main` 不作为迁移源，保持原引用。官方独立仓库不修改。
- 没有创建提交或推送；原增量的三方应用已暂存部分文件。交付补丁使用整个工作树相对官方基线的差异。

5 个源提交依次为：`c02603690`（核心角色权重）、`210fc4fd0`（缓存构建）、`331546805`（tool 别名）、`ce58b24ea`（cuDNN 环境传递）、`3b3f8d303`（生产配置）。

## 20261119 真实验证结果

节点：`gpu-node-112`；共享 Python：`/data_train/vectorchen/env/GLM/baseline/bin/python`。按使用者要求安装 `qwen_vl_utils==0.0.14`、`decord==0.6.0`、`pytest==9.2.0`、`kernels==0.18.3`，并新增其缺失的 `av`、`pluggy`、`iniconfig`；已有包版本未改变。安装前后包清单和日志均已保存。离线环境仍会提示未缓存 bitsandbytes CPU 量化 kernel，这不是包缺失，也不影响本次非量化 tokenizer/CPU 回归；没有执行量化验收。

模型目录：`/data_train/train/qwen/models/Qwen3.5-35B-A3B`。使用真实官方 processor、tokenizer、Qwen3.5 模板与 agent 格式化入口；不加载 35B 权重、不占用 GPU。验证器拒绝包含自定义 Python 或 `auto_map` 的模型目录，不通过伪造模型元数据或 monkeypatch 绕过官方流程。

原始样例：`/data_train/vectorchen/datasets/agent_rollout_example/agent_role_loss_scale/agent_trajectory_standard_example.json`，SHA256 `7853795b2381508a58729fce89bc039d2e55207d875094b8de30fb3355f37455`，未修改。

| 项目 | 最终结果 |
|---|---|
| 独立官方 default+ignore_empty_think vs fork default+ignore_empty_think | 14/14 通过 |
| 独立官方 default+ignore_empty_think vs fork 等价 role 配置 | 14/14 通过 |
| 标准轨迹 | 三者均 782 个 token、228 个监督 token；input_ids、labels、有效权重、解码文本完全一致 |
| 六组权重 | equivalent、historical、fractional、amplified、answer_only、all_roles 全部通过 |
| 独立区域 oracle | 六组均覆盖 782/782 token 且每 token 恰好一次；标签和预期角色/语义/固定 mask 权重一致 |
| 配置注入往返 | 六组权重 × 14 种轨迹，84/84 原始与注入后编码一致 |
| 最终选定 pytest 回归 | 93 passed、132 subtests passed、3 skipped、0 failure/error |

14 类输入包括标准与显式 preserve_thinking、tool 别名、工具后单/多 user、空/重复正文、正常/空/仅关闭 think、直接工具调用、standalone tool、loss=False 历史、OpenAI 并行工具 ID 对齐。3 个 skip 仅因没有提供 Hunyuan tokenizer，不涉及 Qwen/role 核心测试。

修复项：

1. `role.py` 接受官方允许的开头 `</think>` 前缀，固定屏蔽前缀后保留答案权重；原有完整 think 语义权重保持。
2. `template/base.py` 将 agent 新增的工具调用前分隔符归回前一 assistant 片段，解决 assistant 与 tool_call 系数不同时的错误权重；渲染字节不变，非 role 保留官方实现。
3. 注入器先复用官方 OpenAI 消息规范化，再按真实 role 赋权；派生 tool_call 不再错误继承 assistant 的配置权重，并按 ID 对齐并行返回。已包含显式权重时仍拒绝注入，除非 overwrite；规范化后的新 JSONL 是预期产物，原文件不改变。
4. padding/packing 单测原夹具缺少真实模型接口，改为真实小型 Qwen3.5 文本模型，未跳过或伪造测试通过。
5. 验证器首轮错误地向官方传入 fork 独有的 role_loss_config 参数，已修复；首轮失败日志保留，不作为通过证据。

等价范围必须明确：无显式覆盖的等价配置与上述官方 SFT 输入逐 token 一致；role 原始输入与角色权重物化后的输入也一致。**不承诺官方在任意显式 message.loss_scale 下都与 role 相同**：官方显式权重覆盖 ignore_empty_think，而 role 固定屏蔽空 think，这是保留的独有语义。baseline 验证没有把“向官方注入 role 权重”当成 oracle，也没有删除失败样例来宣布等价。

复跑入口：`scripts/utils/validate_role_equivalence.py`，显式传入 official、fork、model、input、output 路径。三方在独立 Python 进程导入各自仓库，并检查实际 swift.__file__。失败返回非零，报告逐项差异。采用原历史区域验证思路，但实际使用当前 tokenizer、当前配置和全部 token 的独立区域规则，不调用被测 role 函数产生期望值。

最终产物：`/data_train/vectorchen/github/ms-swift-validation-20261119/run3/report.json`、`official.json`、`fork_default.json`、`role.json`、`cases.json`、六组配置及注入 JSONL；回归报告 `regression3.xml`；运行前代码快照 `run3-source-snapshot.json`。结束时核对 14 个被测/验证代码文件和原样例 SHA256 均未改变。

新版完整候选补丁归档为 `migration-validated-20261119.patch`，校验值、实际源码哈希与结果汇总为 `validation-final.json`。20261118 原始补丁和记录继续保留追溯。分支仍为 `codebuddy/loss-scale-20261118`，未提交、未推送。

仍待独立验收：真实 35B forward/backward、Trainer 一步更新、实际缓存 export/reload、截断、多模态数据、GPU SP/CP/Megatron 与多节点训练。当前结论是本次 template/tokenizer 与选定 CPU 回归通过，不是全仓库测试或所有训练后端通过。

## 能力的细节与意义

这不是混合精度训练的 GradScaler，也不是样本级整体 loss 缩放；它控制**会话轨迹各部分的逐 token 监督权重**。

1. 五类消息：`system`、`user`、`assistant`、`tool_call`、`tool_response`；兼容官方 `tool` 别名。
2. 显式 `message.loss_scale` 优先于角色配置；缺失时读取 `role_weights`。
3. assistant 文本进一步拆成 `think_tag`、`think_content`、`answer`，语义系数乘以消息基础权重。工具调用使用其来源角色权重，不误当作普通答案。
4. `role+hermes` 将响应片段权重与 Hermes modifier 相乘；`role+ignore_empty_think` 接受参数但由 role 自身固定屏蔽空 think，避免再次拆分后的闭合标签误屏蔽。其他 modifier 仍明确拒绝，不宣称任意组合均可用。
5. 权重必须是非负有限数值。权重为零时 label 不监督；正的连续权重不能降为二值 labels。`role` 固定使用非二值链路，拒绝 `is_binary_loss_scale=true`。
6. 模板固定字符、padding、输入媒体占位展开保留框架 mask；历史 EOS 按 0908 的正权重响应规则赋 1，最终 suffix 默认赋 1。**消息 loss=False 不意味着最终 suffix 也被屏蔽**。
7. 新增 token-backed 响应的显式 token mask 与角色权重相乘，不能从 token IDs 还原 think/answer 语义；历史 EOS 对此路径以末尾 token 权重为准。普通文本历史 EOS 保留 0908 的“存在正权重片段”规则，它并不等同于官方动态 EOS 的所有边界行为。
8. 权重分段发生在分词前；改变段边界可能改变 tokenizer 的 tokenization。不能假定仅权重变化而 input_ids 永远不变。

普通 SFT 下仍复用官方损失归一化：

\[
L = \frac{\sum_t w_t\,\mathrm{CE}_t}{N_{\mathrm{有效标签}}}
\]

权重在 loss 端与 shifted target 对齐。分母不是权重总和，也不是逐样本平均。因此提高某类 token 的权重会提高其梯度贡献及整体梯度量级；它可用于分别控制用户指令、工具使用、工具结果、思考与答案的学习侧重。零权重改变有效标签数，也会影响归一化。

缓存构建只物化消息基础权重和 lengths，训练时仍需同一 role 配置来计算 assistant 语义权重及逐 token 权重；缓存不能代替 role 策略或配置文件。

## 原 22 文件逐项处理

“迁移”表示源增量已纳入候选工作树，不代表相应运行测试通过。

| 原文件 | 处理策略 |
|---|---|
| `swift/loss_scale/role.py` | 完整保留角色配置、注入、语义拆分和 modifier；补充无效消息权重拒绝、token-backed mask、混合查询及 standalone 来源适配 |
| `swift/loss_scale/mapping.py` | 完整迁移 role 工厂与组合限制；保留官方其余策略注册 |
| `swift/arguments/base_args/template_args.py` | 迁移配置参数、路径处理、二值模式拒绝和 kwargs 透传；其余官方参数保留 |
| `swift/template/register.py` | 迁移 role_loss_config 透传，复用新版注册接口 |
| `swift/template/utils.py` | 迁移 SYSTEM、QUERY、RESPONSE_SUFFIX 类型，保留新版 helper |
| `swift/template/template_inputs.py` | 迁移系统权重、来源角色保存；保留新版 OpenAI 工具结果按 ID 对齐与偏好归一化 |
| `swift/template/base.py` | 人工解决两块代码冲突：保留官方工具后追加 user 的渲染、逐轮闭合及字面占位符；附加原角色来源、正文边界和历史 EOS；不覆盖新版 encoding/collator/packing |
| `scripts/utils/inject_role_loss_scale.py` | 完整迁移独立 JSONL 注入入口，不另建算法 |
| `scripts/utils/build_cached_dataset/distributed_cached_dataset.py` | 完整迁移分片、任务领取、heartbeat、export、共享复制、合并和指纹；增强源码/数据配置指纹及 SSH 安全默认值 |
| `scripts/utils/build_cached_dataset/run_distributed_build.sh` | 完整迁移；补齐其引用的缺失 example TOML |
| `scripts/utils/build_cached_dataset/role_loss_config_sft0_equiv.json` | 原样迁移官方 output-only 等价角色配置 |
| `scripts/utils/build_cached_dataset/cached_dataset_config_d64_sft0_default.toml` | 保留历史生产参数；仓库路径更新为 fork，标注必须重新核对模型、节点、环境和输出；不默认启动 |
| `tests/loss_scale/test_role_loss_scale.py` | 保留全部原有回归，新增三个关闭 think / OpenAI 注入与覆盖检查；已在 20261119 最终回归通过 |
| `docs/source/Customization/Agent-role-loss-scale.md` | 完整迁移能力说明；旧测试结果改标历史记录并链接本记录 |
| `docs/source/Customization/Architecture.md` | 保留官方内容并迁移 role 入口说明 |
| `docs/source/Customization/Custom-dataset.md` | 人工合并：保留新版 0.5/2.0 非二值说明及 role 数据扩展 |
| `docs/source/Instruction/Agent-support.md` | 保留官方内容并迁移 role 链接 |
| `docs/source/Instruction/Command-line-parameters.md` | 迁移 role 参数说明，保留官方新参数 |
| `docs/source_en/Customization/Architecture.md` | 迁移英文 role 入口说明 |
| `docs/source_en/Customization/Custom-dataset.md` | 同中文冲突解决策略，保留非二值权重的新说明 |
| `docs/source_en/Instruction/Agent-support.md` | 迁移英文 role 链接 |
| `docs/source_en/Instruction/Command-line-parameters.md` | 迁移英文 role 参数说明 |

没有因文本冲突整文件选择 ours/theirs，也没有遗漏原 22 文件。

## 新版适配与新增项

- `swift/agent_template/base.py`：将已有 `prefix + user_content` 拆成两段，渲染字节应保持不变；这是为了逐片段辨别权重，不另建 agent 渲染器。
- role 混合查询使用唯一字符串子类追踪用户正文来源，支持空串及重复正文；不通过字符串搜索猜测来源，渲染器丢失边界时明确报错。
- 连续 assistant 同步合并 content/loss/loss_scale/source_role；连续 user 和 standalone tool 另存来源片段，仍维持官方交替 query/assistant 表示。
- 直接传入 TemplateInputs 也初始化来源角色；缺失来源回退 assistant。
- 非 role 模式使用官方原有占位符替换顺序及上下文片段，不应用新的角色边界；历史结束符保持 OTHER，再交给官方 dynamic EOS。并行工具权重一致性约束只对 role 生效。
- 缓存 fingerprint 增加 seed、shuffle、split 比例，覆盖 agent/template/loss_scale/dataset Python 实现和依赖声明文件。旧指纹会被拒绝复用。这不是缓存真实性认证，也没有记录所有已安装依赖版本。
- SSH 默认开启主机密钥验证，主机列表拒绝参数/路径/命令字符；使用前需配置 trusted known_hosts。没有运行任何远端任务或访问内部节点。
- 新增通用 `cached_dataset_config.example.toml`，使默认启动器不再引用缺失文件；默认从当前仓库定位 ms-swift，不固化旧仓库路径。
- 新增 `tests/loss_scale/test_role_migration.py`：混合工具/user、别名、空/重复正文、连续用户、standalone、公开 TemplateInputs、原生非 role mask、token mask、loss/梯度、padding/packing 回归。
- 新增 `tests/general/test_role_cached_builder.py`：配置、指纹、缓存拒绝、主机校验、cuDNN 环境传递；最后增加了源码覆盖用例，尚未重跑。

## 明确保留官方而未搬旧实现的部分

`swift/trainers`、`swift/megatron`、`swift/dataset/packing.py`、官方模型注册与模型实现均不复制 0908 旧文件；Trainer 的 loss shift/有效 token 归一化、序列并行、Megatron、padding-free、多模态展开继续使用官方基线。此处“复用”不是已经完成各模型/后端实测。

## 20261118 历史验证证据与未完成项

本节保留 20261118 当时状态；环境、原有回归、OpenAI 注入等已在上面的 20261119 节更新。不得把本节的未完成项与当前结果混为一谈。

当时已执行：

- 原始增量与工作树文件集合覆盖检查：22/22。
- 全部受影响 Python 的 AST 解析、JSON/TOML 解析、Shell 语法、git diff --check、无冲突标记/无 unmerged 索引检查。
- 针对编辑代码与测试的 IDE 诊断，无新增报告。
- 审查前的缓存构建器 14 个用例运行通过，有 `cache-tests.xml`。之后新增源码指纹覆盖及其他静态修复，**这份结果不是最终候选版本的完整验收**。

未通过/未完成：

- 系统 Python 缺 torch，loss_scale 整个目录在收集阶段报 4 项依赖错误；不是测试通过或主动 skip。
- 现有 GLM baseline 导入 torch I/O 超时；隔离 CPU 环境安装遇到 `.Trash-1231` 权限失败。没有修改原训练依赖；使用者已选择不继续排查环境。
- 原有 role 单测、新增融合单测、官方工具追加 user/缓存/损失回归均需要在可用训练环境重新执行。
- 新增逐 token loss/梯度参考用例不等同于完整 Trainer forward/backward；还需模型训练 smoke test。
- Qwen3.5/Hermes 是原 role 工具响应 `<tool_response>` 解析的重点路径；其他模型原生工具格式不保证具有同一标记。不能宣称所有模型格式已经支持、实测或自动泛化。
- OpenAI assistant.tool_calls 会由官方规范化为 tool_call；如果原 assistant 已有显式 loss_scale，官方会传给派生调用。注入原始 OpenAI 数据时不能假定 tool_call 自动改用另一配置权重；需要规范化后的数据与无注入路径的对照验收。
- 文本历史 EOS 的 0908 正片段规则保留；token-backed 尾 mask 分支新增。两者差异需用实际 tokenizer 验证，不能用“官方无回退”涵盖 role 模式自身训练目标。
- 尚无左右截断/多模态模型专属展开、SP/CP、GPU/Megatron、真实缓存 export 往返和多节点构建的本次运行结果。
- 训练入口仍不自动校验 builder 外部 manifest，需训练启动侧核对源码、role 配置及构建参数；不要加载不可信 pickle/indexed cache。

准入标准：在可用环境跑完上面的必需用例和实际使用模型/后端 smoke test、记录失败修复及最终补丁后，才可将候选版本标为可训练发布。不能仅因三方应用无冲突或静态检查通过而发布。

## 补丁与交付

外部归档目录：`/data_train/vectorchen/github/ms-swift-migration-20261118`。

- `0908-complete.patch`：共同祖先到源 tip 的完整原始补丁。
- `0001-*.patch` 至 `0005-*.patch`：源分支五个提交的补丁序列。
- `0908-files.txt`：完整源文件清单。
- `migration-final.patch`：当前完整工作树相对官方基线的可重放补丁，包括新增测试及本记录。
- `migration-files.json`：每个源文件的原版/新版内容哈希和是否原样的机器可读清单。
- `artifact-sha256.txt`：原始与最终补丁校验值。
- `cache-tests.xml`、`role-tests.xml`、`environment-install.log`：此前实际结果和环境失败证据，不能解释为最终验收通过。

最终补丁只在清洁的指定官方基线上应用，原始 0908 补丁用于追溯，不应在最终补丁之后重复应用。补丁可重放性使用独立临时 Git 索引检查，不创建提交、不改正式分支 HEAD。分支未提交状态下的持久代码交付以补丁和工作树为准。
