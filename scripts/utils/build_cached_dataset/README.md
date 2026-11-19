# Agent Role Loss Scale 迁移、配置与验证说明

## 1. 结论与阅读范围

本文是截至 **2026-11-19** 的中文单页审阅记录，汇总本次角色权重迁移、近期模板适配、缺陷修复、配置选择和实际验证证据。放在缓存构建配置旁，方便从配置入口追溯到训练模板和验收报告；不是只说明缓存脚本，也不是整个 ms-swift 的使用手册。

**已经完成：**使用 Qwen3.5、DeepSeek-V4.1、GLM-5.3-Flash 各自真实本地 processor/tokenizer、官方原生 template/agent template，对文本 Agent 轨迹完成独立官方、迁移默认策略、迁移等价 `role` 配置的严格 SFT 编码对照；六组权重的全部支持样例完成独立 token 区域检查与注入往返。核心回归为 **126 passed、132 subtests passed、3 skipped、0 failure/error**。

**没有完成：**三个大模型权重的 forward/backward、完整 Trainer 更新、实际缓存 export/reload、分布式缓存构建、多模态、GPU/SP/CP/Megatron 和多节点训练验收。不能把 tokenizer/template 编码通过写成“大模型训练通过”或“所有模型、所有后端通过”。

本文以外部归档的实际报告和源码快照为准。此前文档中的 Qwen-only 结果、环境阻塞和待验收说明是阶段记录，不应与最终跨模型结果混用。本文初次新增时没有重新运行模型验证、训练或缓存构建。

**后续范围收缩：**按使用者要求，`swift/agent_template/deepseek_v3_1.py` 已恢复官方 HEAD 原始实现，并移除仅针对该适配新增的三个正文来源参数用例；GLM 修改保留。下文 31 文件补丁与 126 passed 是恢复之前的真实历史证据，不改写原报告，也不冒充恢复后的整套复验结果。三个已测模型的 agent 调用链不经过 V31 override。恢复后完整运行 `tests/loss_scale/test_role_loss_scale.py`，**27 passed、退出码 0**，约 118 秒；证据保存在 `/data_train/vectorchen/github/ms-swift-validation-20261119/cross-model/v31-revert/role.log`、`role.xml`、`role.exit`。本轮未重新运行三模型完整对照。

### 1.1 基线与交付范围

| 项目 | 记录 |
|---|---|
| 候选仓库 | `/data_train/vectorchen/github/ms-swift-codebuddy` |
| 独立官方仓库 | `/data_train/vectorchen/github/ms-swift`，验证期间未修改 |
| 官方基线 | `1997fa6f4192d41769ea17b43eb811f943ed2aed` |
| 迁移源 | `origin/feat/agent-role-loss-scale-v20260908`，源 tip 为 `3b3f8d30314bec40f6dd808e2400188749d25881` |
| 共同祖先 | `75baafb27f3f2b9676f0fa1b6bfe4dc5715698a4` |
| 原始增量 | 5 个提交、22 个文件；后续在当前官方基线上适配，而非整文件覆盖 |
| 最终已核对补丁 | 31 个候选文件；另有未完成运行验收的训练集成测试，不在该补丁内 |
| 当前提交状态 | 迁移主体已暂存，后续部分修复未暂存；本次工作没有提交或推送，也没有更改用户暂存区 |

原始 5 个提交分别涉及角色权重、缓存构建、`tool` 别名、cuDNN 环境传递和生产配置。源增量、后续修复、验证工具和文档都属于本次交付，但其运行验收范围不同，见第 4～7 节。

## 2. 配置选择与职责边界

### 2.1 严格 SFT 等价配置

首选本目录已有的 [role_loss_config_sft0_equiv.json](role_loss_config_sft0_equiv.json)，不是另建一份名字相近的配置。

| 配置类别 | 字段 | 值 |
|---|---|---:|
| `role_weights` | `system` | 0.0 |
| `role_weights` | `user` | 0.0 |
| `role_weights` | `assistant` | 1.0 |
| `role_weights` | `tool_call` | 1.0 |
| `role_weights` | `tool_response` | 0.0 |
| `assistant_weights` | `think_tag`、`think_content`、`answer` | 均为 1.0 |

其内容与验证器生成的 `config-equivalent.json` 一致：只监督 assistant 正文及工具调用，query 侧不监督；空 thinking 前缀、历史 EOS 和最终 suffix 仍由模板与 `role` 规则处理。使用 `--loss_scale role` 或 `role+ignore_empty_think` 均由 `role` 自身固定屏蔽空 think；最终严格对照采用的是 **官方 `default+ignore_empty_think` 对迁移 `role+ignore_empty_think`**。

注意：

- 两组权重字段必须完整，拒绝未知字段、负数、非有限数值和错误类型。
- `role` 使用连续逐 token 权重，不能配置 `is_binary_loss_scale=true` 来丢弃权重数组。
- 显式 `message.loss_scale` 优先于角色默认值；已物化的消息不会因为修改 `role_weights` 自动改变。assistant 子区域系数训练时仍从配置读取。
- 等价对照没有任意显式消息权重覆盖。官方显式权重可以覆盖 `ignore_empty_think`，而 `role` 始终屏蔽空 think；这是保留的语义差异，**不承诺任意显式覆盖条件下严格等价**。
- 配置名称中的“等价”只指本次支持输入与已核对设置，不是对所有模型、RLHF 或 Jinja 后端的保证。

### 2.2 缓存构建配置

| 文件 | 用途 | 本次验收状态 |
|---|---|---|
| [cached_dataset_config.example.toml](cached_dataset_config.example.toml) | 新建环境的通用起点；占位模型、节点、环境、数据与输出路径；默认 4 分片、32768 长度、`delete` 截断、Qwen 模板 | 配置解析、当前仓库定位、role 配置解析及相关单测通过；未运行真实构建 |
| [cached_dataset_config_d64_sft0_default.toml](cached_dataset_config_d64_sft0_default.toml) | 0908 历史 Qwen 生产配置；64 分片、65536 长度、`right` 截断及旧环境/数据/节点路径 | 已迁移并调整仓库路径；不是本次三个模型已经验收的生产配置 |
| [role_loss_config_sft0_equiv.json](role_loss_config_sft0_equiv.json) | 构建物化与训练展开共用的 output-only 角色系数 | 与三模型严格等价配置内容一致 |
| [run_distributed_build.sh](run_distributed_build.sh) | 薄启动入口，读取 `PYTHON_BIN`、`CONFIG`，默认引用通用示例 TOML | 默认引用的缺失配置已补齐；没有执行多节点启动 |

新环境应复制通用示例并逐项填写，**不要直接启动历史 64 分片配置**。其中历史环境不是本次验证使用的共享 GLM baseline。构建器支持环境变量覆盖；审阅时需同时核对 TOML 与环境，尤其 `MODEL`、`MS_SWIFT_REPO`、`ROLE_LOSS_CONFIG`、`TEMPLATE`、`AGENT_TEMPLATE`、`LOSS_SCALE`、输出目录和节点列表。

`role_loss_config` 的相对路径以 TOML 所在目录解析；通用示例未指定 `runtime.ms_swift_repo` 时，从脚本位置定位当前仓库。三个真实模型的对应模板见第 5 节，不能只改模型目录而继续使用 Qwen 模板。本文不新增未经缓存运行验证的 DeepSeek/GLM 生产 TOML。

### 2.3 注入、缓存与训练不是同一层

1. `swift/loss_scale/role.py` 是角色权重与 assistant 语义拆分的唯一实现；模板保留来源、固定 mask 和结束符边界。
2. `scripts/utils/inject_role_loss_scale.py` 只调用公共注入函数，将消息基础权重写入新 JSONL，不重复实现 loss 算法。
3. 缓存构建器负责分片、任务领取、heartbeat、权重物化、调用当前仓库 `swift export`、复制与合并。
4. 普通 `swift export --to_cached_dataset` 保存的是**消息行及 `lengths`**，训练读取后重新 encode，不是已经完成的逐 token IDs/labels/权重缓存。完整 encoded Arrow 缓存是另一种路径。
5. 普通缓存中的 `lengths` 和消息权重不能替代训练时相同的 template/agent template、源码和 assistant 配置；构建 manifest 也不是训练入口自动完成的真实性认证。

## 3. DeepSeek V3.1 与 GLM4 修改是否必要

**结论：`glm4.py` 对已测 GLM-5.3-Flash 的 role 工具后追加 user 场景必要，保留；`deepseek_v3_1.py` 不属于三个已测模型的必需修改，已按使用者要求恢复官方 HEAD 原始形态。**不修改模型权重或工具协议。

| 模型 | 实际 agent 继承/实现 | 与这两处文件的关系 |
|---|---|---|
| Qwen3.5-35B-A3B | `Qwen3_5AgentTemplate` → `Qwen3CoderAgentTemplate` → `HermesAgentTemplate` | 不依赖这两处；Hermes 已保留 followup 正文片段 |
| DeepSeek-V4.1-Flash-1017 | `DeepSeekV41AgentTemplate(BaseAgentTemplate)` | 不调用 V31 agent override；需要的是共享 base 来源适配及 DeepSeek 原生返回、EOS 等修复 |
| GLM-5.3-Flash | `GLM5_1AgentTemplate(GLM4_5AgentTemplate)`，共享实现位于 `glm4.py` | 调用该文件的 `_format_tool_user_followup()`，文件名不能代表只服务 GLM4 |

保留的 GLM `_format_tool_user_followup()` 将 `prefix + user_content` 改成独立的 prefix 和原正文对象两个片段，最终连接文本不变。`Template._role_followup_prompt()` 用正文对象身份追踪 user 来源，而不是搜索正文字符串；拼接会丢失对象身份，尤其无法准确处理空串、重复正文、正文包含字面 `{{QUERY}}` 的情况。

需要区分两条 DeepSeek 继承链：普通会话模板 `DeepseekV41Template` 确实继承 `DeepseekV3_1Template`，但恢复的文件是 **agent** 的 `deepseek_v3_1.py`，V41 agent 并不继承它。因此 V41 通过不能作为 V31 真实模型通过的证据。

当前处理与历史证据：

- **保留 `glm4.py` 修改**：由 GLM-5.3-Flash 原生模板的 28 类样例、六组权重与注入验证覆盖；同时保留 GLM4.5/4.7/5.1 来源片段单测。
- **恢复 `deepseek_v3_1.py`**：撤销 prefix/正文分段适配，返回值与官方 HEAD 一致。此前 V31 空、重复、普通正文来源单测通过仅属于撤销前的记录，不代表当前支持；没有 V31 真实模型三方验收。
- 从新增 `test_native_followup_preserves_user_provenance` 参数矩阵移除 V31 的三个用例及专属期望分支，保留 GLM 的九个用例；没有删除官方原有 DeepSeek 工具协议测试或跳过已测三模型用例。
- V31 的 role 工具后追加 user 正文来源适配现在不在交付范围内。GLM4.5 补换行、GLM5 不补换行仍保持官方原有行为。

## 4. 全部代码变更与修复原因

### 4.1 核心实现与接线

本表只列相对官方 HEAD **当前实际新增或修改的代码文件**，包含运行实现、CLI/构建脚本、验证工具与测试；同时核对已暂存、未暂存和未跟踪代码，共 18 个文件。已恢复且无差异的文件不列入本表，恢复过程只在历史说明中保留。配置与文档另外列在第 4.3～4.4 节。

以下代码路径均相对候选仓库根目录，证据路径相对 `/data_train/vectorchen/github/ms-swift-validation-20261119/`。每个代码文件均说明修改原因及实际验证证据；“未验收”不表示通过。表中“回归”指第 6 节选定测试，不是全仓库测试。

共同证据入口：三模型编码及区域/注入检查见 `cross-model/run3-{qwen,deepseek,glm}/report.json`；恢复后完整角色单测见 `cross-model/v31-revert/role.xml`；配置接线、公开输入、来源、token mask、loss/梯度、padding/packing 及 builder 历史回归见 `cross-model/regression5.xml`。这些证据的版本边界遵循第 1 节说明。

| 文件 | 本次修改与原因 | 验证证据/边界 |
|---|---|---|
| `swift/loss_scale/role.py` | 迁移五类角色权重、assistant 的 think 标签/正文/答案系数、JSONL 注入和 modifier；保留显式权重优先、非负有限值校验、token-backed mask；补齐原生工具返回和关闭 think 前缀，避免角色系数失效 | 三模型编码、六组独立区域与注入检查；配置、语义、mask 单测。token-backed 全量模型矩阵未验收 |
| `swift/loss_scale/mapping.py` | 注册 `role` 工厂并传入配置；空 think 由 role 自行屏蔽，避免串联 modifier 后误屏蔽非空 think 的闭合标签；限制不支持的组合 | 配置/组合单测；三模型使用 `role+ignore_empty_think` |
| `swift/arguments/base_args/template_args.py` | 新增 `role_loss_config` 参数、路径处理、连续权重约束与模板 kwargs 透传 | 参数/模板构造回归；不等于完整训练 CLI 验收 |
| `swift/template/register.py` | 在当前官方注册接口透传 role 配置，不替换官方模型注册 | 模板构造及真实模型编码 |
| `swift/template/utils.py` | 增加 `SYSTEM`、`QUERY`、`RESPONSE_SUFFIX` 上下文类型，区分正文与固定包装/历史结束符 | 来源、mask 与 EOS 回归 |
| `swift/template/template_inputs.py` | 保存 system 权重及原角色来源；兼容 `tool` 别名，继续复用官方 OpenAI 工具 ID 对齐和规范化 | 别名、OpenAI 并行返回及公开输入回归；不覆盖全部非法 ID/偏好场景 |
| `swift/template/base.py` | 定向合并新版工具后 user、逐轮闭合、字面占位符逻辑；保留连续 assistant/tool_call 及混合 query 的来源，配置路径进入 loss-scale 缓存键；明确固定 mask、历史 EOS；截断同步裁剪 IDs/labels/权重 | 三模型与来源、EOS、padding/packing 回归；实际左右截断集成仍未通过运行验收 |
| `swift/agent_template/base.py` | followup prefix 与原始 user 正文分段，渲染连接值不变，供 role 来源映射使用 | DeepSeek-V4.1 真实编码与工具后 user 回归；非 role 官方对照 |
| `swift/agent_template/glm4.py` | 对共享 GLM followup override 做相同正文来源适配 | GLM-5.3-Flash 真实编码；必要性见第 3 节 |
| `scripts/utils/inject_role_loss_scale.py` | 迁移公共角色权重注入函数的 CLI 入口，统一输入/配置/输出/overwrite 参数，避免脚本与训练端重复定义算法 | 公共注入函数由三模型六组注入往返覆盖，见 `cross-model/run3-*/report.json`；对应注入/overwrite 单测恢复后仍通过，见 `cross-model/v31-revert/role.xml`。未独立验收每个 CLI 分支 |
| `scripts/utils/build_cached_dataset/distributed_cached_dataset.py` | 迁移分片、任务领取、heartbeat、export、复制与合并；增强源码/配置/seed/shuffle/split 指纹，拒绝错误缓存复用，核对实际导入仓库；补齐 cuDNN 环境传递、SSH 主机密钥检查及主机输入校验 | `tests/general/test_role_cached_builder.py` 15 项通过，见 `cross-model/regression5.xml`，覆盖配置、指纹、缓存拒绝、主机和环境。真实 export、任务恢复、多节点复制/合并未验收 |
| `scripts/utils/build_cached_dataset/run_distributed_build.sh` | 保留配置驱动的薄启动器，补齐默认引用的示例 TOML，避免默认入口指向缺失配置或旧生产环境 | 迁移阶段 Shell 语法静态检查；默认配置解析/仓库定位见 `test_example_resolves_current_repository` 和 `cross-model/regression5.xml`。未运行多节点启动 |
| `scripts/utils/validate_role_equivalence.py` | 新增并扩展独立官方/default/role 四字段对照；支持真实模型模板参数、28 case、六组配置、注入往返、worker 超时和清单核对后的结果收集，单列官方一致拒绝 | `cross-model/run3-qwen/report.json`、`run3-deepseek/report.json`、`run3-glm/report.json`：28/28、27/27、28/28 严格等价；九个 worker 与三个 collector 退出 0，六组全 case 区域/注入通过。是恢复 V31 前的完整运行证据 |
| `scripts/utils/role_region_oracle.py` | 新增独立原生区域期望，避免沿用 Qwen ChatML 假设或以被测 role 输出作为预期；识别 GLM 嵌入标记、并行调用批次、DeepSeek 前缀/EOS，并核对 BPE 边界和全 token 分区 | 上述三份报告的 `weighted` 与 `injection_roundtrips.regions`：六组标准轨迹均 6/6，全支持 case 分别 168/168、162/162、168/168，标签与权重检查通过；不覆盖所有模型/格式 |
| `tests/loss_scale/test_role_loss_scale.py` | 保留原角色回归，新增配置/语义/注入、原生返回正权重、空历史 EOS、GLM 空/重复正文来源及 thinking 分隔符回归；恢复 V31 后移除仅要求已撤销适配的三个参数用例和专属分支 | 当前文件完整运行 27/27 通过，无 skip/failure/error，退出 0；证据 `cross-model/v31-revert/role.log`、`role.xml`、`role.exit`。历史30项记录不混作当前数量 |
| `tests/loss_scale/test_role_migration.py` | 新增角色配置接线、混合 query、来源、token mask、loss/梯度及 padding/packing 数组对齐回归；采用真实随机初始化 tiny Qwen 夹具，避免缺失官方模型接口 | 24 项通过，见 `cross-model/regression5.xml`；loss/梯度参考与数组对齐不是完整大模型 Trainer optimizer step 验收。本文件在 V31 恢复后未改动，本轮未单独重跑 |
| `tests/general/test_role_cached_builder.py` | 新增默认配置定位、训练语义/role 配置/源码指纹、旧缓存拒绝、主机参数注入拒绝及 cuDNN 环境传递回归，明确缓存一致性与安全边界 | 15 项通过，见 `cross-model/regression5.xml`；不等于分布式缓存任务实际运行通过。本文件在 V31 恢复后未改动，本轮未单独重跑 |
| `tests/loss_scale/test_role_training_integration.py` | 新增左右截断、encoded Arrow 缓存重载、完整 tiny Qwen Trainer/SGD 一步和原生 CLI 普通缓存往返测试，用于补足编码/公式单测之外的集成验收 | 仅 AST、空白检查与 IDE 诊断通过；启动阶段 I/O 超时，600 秒退出 124，无 collection/JUnit/断言结果。证据 `continuation/validation-continuation.json`、`integration2.log/exit`；未运行通过，不在历史31文件补丁中 |

没有复制旧版 `swift/trainers`、`swift/megatron`、`swift/dataset/packing.py`，没有修改模型注册或模型权重实现。loss shift、有效标签分母、packing/collator 和各训练后端继续复用官方代码；“复用”不代表这些后端全部运行通过。

### 4.2 已定位的核心缺陷与修复

| 缺陷/风险 | 修复方式与原因 | 最终覆盖 |
|---|---|---|
| OpenAI assistant 中派生 tool_call 在注入时误继承 assistant 角色默认权重；并行返回可能错配 | 先复用官方消息规范化与 ID 对齐，再按规范化后的真实角色赋权；保留已有权重默认拒绝、显式 overwrite 契约 | OpenAI 并行返回、直接调用、六组注入往返及注入单测 |
| 工具后新增 user 被合并进工具包装，空/重复正文难以定位 | 使用 `RoleContent` 保存原正文身份；agent prefix/正文分段，模板仅给真正 user 正文用户权重，不给固定控制 token 用户权重 | `followup_3`～`followup_6`、并行返回后单/双 user；正 user/tool_response 权重检查 |
| Qwen 历史 thinking 删除中的 `strip()` 吞掉已归属 assistant 的工具前换行 | 记录 `_role_tool_call_prefix` 并在历史 thinking 剥离后恢复；既保留字节，又区分 assistant 与 tool_call 权重 | thinking 开关/历史保留组合；字符串与合并片段两种单测 |
| DeepSeek 原生返回不是 `<tool_response>`，非零返回权重可能静默失效 | 增加 `<tool_result>` 与 V31 原生 output 块识别；不改工具格式化协议 | V41 六组区域检查；三种返回块正权重单测；V31 仅语法级覆盖 |
| DeepSeek 工具 query 首段合并 EOS 与 user header，历史 EOS 监督丢失 | 在 role 路径分离原生 EOS，按前一个 response 的有效监督状态标注；不让正权重 query 触发错误的动态推断 | V41 严格标签/权重对照及 mask/history case |
| 空或 `loss=False` assistant 吸收独立工具前缀，改变 mask 或分词片段 | 仅在有可用 assistant 正文且未显式屏蔽时归回前缀；空/屏蔽轮保留工具自身的前缀及分段 | `empty_think_before_tool`、`masked_tool_history`、`call_content_16`～`call_content_20` |
| 直接 tool_call 的前导 `</think>` 没有屏蔽 | 同官方 `ignore_empty_think` 识别前导关闭标签及其空白；保留后续调用权重 | 三模型直接调用；独立关闭 think 前缀单测 |
| 空历史 assistant 没有正文 token，却被认为可恢复 EOS | 判断实际非空且有正权重的片段，不仅看配置系数为正 | `empty_history_no_prefix`、空历史 EOS 单测 |
| 分段改变可能造成 tokenizer IDs 与官方不同 | 分别比较 IDs、解码、labels 与有效权重；不把“权重预期正确”当作 tokenization 等价 | 三模型严格四字段对照 |

普通 SFT 加权 loss 仍复用官方归一化：

\[
L = \frac{\sum_t w_t\,\mathrm{CE}_t}{N_{\mathrm{有效标签}}}
\]

分母不是权重总和；权重需与 shifted target 对齐。loss/梯度公式及 padding/packing 数组对齐已有回归通过，但这不是完整 Trainer optimizer step 的运行证明。

### 4.3 注入与缓存构建相关文件

| 文件 | 变更与原因 | 已验证/未验证 |
|---|---|---|
| `scripts/utils/inject_role_loss_scale.py` | 迁移公共注入函数的 CLI 薄入口，输入/配置/输出/overwrite 参数齐全 | 公共注入逻辑与真实模板往返通过；不另宣称每个 CLI 分支都验收 |
| `scripts/utils/build_cached_dataset/distributed_cached_dataset.py` | 迁移分片、任务领取、heartbeat、export、共享复制、合并和指纹；增强源码、配置、seed/shuffle/split 指纹，拒绝旧缓存复用；核对实际导入仓库，传递 cuDNN 环境；SSH 默认校验主机密钥，拒绝主机参数/路径/命令注入 | 最终 builder 15 项配置/指纹/主机/环境单测通过；没有真实 export、任务恢复、多节点复制/合并运行结果 |
| `scripts/utils/build_cached_dataset/run_distributed_build.sh` | 保留轻量启动器，默认示例配置不再缺失 | Shell 静态检查；默认配置单测 |
| `scripts/utils/build_cached_dataset/role_loss_config_sft0_equiv.json` | 迁移 output-only 等价系数，构建与训练共用 | 内容与最终 equivalent 配置一致 |
| `scripts/utils/build_cached_dataset/cached_dataset_config_d64_sft0_default.toml` | 保留历史生产参数、更新 fork 路径并提示重新核对环境 | 配置迁移/静态检查，不作为本次生产构建通过证据 |
| `scripts/utils/build_cached_dataset/cached_dataset_config.example.toml` | 新增通用配置，补齐启动器原缺失引用，避免默用旧仓库/生产路径 | 解析、仓库定位、role 配置单测 |

源码/数据/配置指纹用于一致性与复用判断，不是恶意篡改防护认证，也没有记录所有已安装依赖版本。启动真实节点之前需审核节点列表、可信 `known_hosts`、资源参数和输出目录；本文没有触发任何远端构建。

### 4.4 验证工具、测试与文档

| 文件 | 变更与证据 |
|---|---|
| `scripts/utils/validate_role_equivalence.py` | 新增并扩展独立三方验证器；按真实模板参数加载 processor/tokenizer；派生 28 case、六组配置，记录导入路径、逐项差异和报告；加入限时 worker、清单核对后的收集模式，一致拒绝单列 |
| `scripts/utils/role_region_oracle.py` | 新增多模型独立区域期望，不调用被测 role 函数产生预期；识别原生头、think、工具批次/返回、固定包装和 EOS；全 token 分区检查 |
| `tests/loss_scale/test_role_loss_scale.py` | 保留原角色回归，新增关闭 think、OpenAI 注入与 overwrite、原生返回正权重、空历史 EOS、GLM 空/重复正文来源、历史 thinking 分隔符等测试；历史版本 30 项通过，恢复 V31 后移除三个范围外用例，当前完整角色单测 27 项通过，证据见第 1 节 |
| `tests/loss_scale/test_role_migration.py` | 新增配置接线、混合 query、来源、token mask、loss/梯度与 padding/packing 回归；夹具改为真实随机初始化 tiny Qwen，避免缺失模型接口；最终 24 项通过 |
| `tests/general/test_role_cached_builder.py` | 新增配置、指纹、源码覆盖、旧缓存拒绝、主机校验和 cuDNN 传递回归；最终 15 项通过 |
| `tests/loss_scale/test_role_training_integration.py` | 另行新增左右截断、encoded Arrow 缓存重载、tiny Qwen 完整 Trainer/SGD 一步、原生 CLI 普通缓存往返测试；仅静态检查通过，运行初始化超时；**不在已验证的 31 文件补丁及最终核心回归内** |
| `docs/source/Customization/Agent-role-loss-scale.md` | 迁移中文能力/配置/使用说明，修正示例配置路径，保留历史范围提示 |
| `docs/source/Customization/Loss-scale-migration-20261118.md` | 新增迁移逐项记录，补入 Qwen 阶段证据并标记 20261118 环境阻塞；最终跨模型结果在本文与外部报告中说明 |
| `docs/source/Customization/Architecture.md`、`docs/source/Customization/Custom-dataset.md` | 合并 role 入口/数据扩展，保留官方连续权重说明 |
| `docs/source/Instruction/Agent-support.md`、`docs/source/Instruction/Command-line-parameters.md` | 合并 role 文档入口与新增参数说明 |
| `docs/source_en/Customization/Architecture.md`、`docs/source_en/Customization/Custom-dataset.md` | 对应英文入口与数据说明同步迁移 |
| `docs/source_en/Instruction/Agent-support.md`、`docs/source_en/Instruction/Command-line-parameters.md` | 对应英文入口与参数说明同步迁移 |
| `scripts/utils/build_cached_dataset/README.md` | 本文：新增中文单页配置/变更/修复/验证审阅记录；是在模型验证结束之后新增，不属于当时已重放的补丁 |

历史补丁包含核心实现 10 个、注入/构建 6 个、验证工具及已选新增回归 5 个、文档 10 个，共计 31 个文件。恢复 V31 后，核心实现变更为 9 个，该文件不再属于当前增量；测试矩阵同步调整。另列训练集成测试及本文，避免将未验证代码遗漏或混入通过数。第 6 节的其他官方测试文件被运行，但未被本次修改。

## 5. 严格三方与独立区域验证

### 5.1 模型和真实入口

| 真实模型目录 | `template_type` | `agent_template` | 标准轨迹 token / 监督 token |
|---|---|---|---:|
| `/data_train/train/qwen/models/Qwen3.5-35B-A3B` | `qwen3_5` | `qwen3_5` | 782 / 228 |
| `/data_train/train/deepseek/models/DeepSeek-V4.1-Flash-1017` | `deepseek_v41` | `deepseek_v41` | 789 / 271 |
| `/data_train/train/glm/models/GLM-5.3-Flash` | `glm5_next` | `glm5_1` | 566 / 171 |

GLM-5.3-Flash 使用的是官方 `glm5_next` 注册及 `glm5_1` agent，不能按模型名称猜成 `glm5_3`。DeepSeek 的 `get_processor` 可能直接返回 tokenizer；验证器已兼容这类真实接口，不伪造 Qwen processor。

三方分别在独立 Python 进程导入独立官方仓库与候选仓库，并核对实际 `swift.__file__`。比较 `input_ids`、`labels`、有效权重和完整解码文本；`label == -100` 的有效权重记为零，官方缺少连续数组时监督 token 权重按 1 处理。候选默认策略用于证明没有改变官方行为，不能代替候选 role 与官方的第二份对照。

### 5.2 最终结果

| 模型 | 官方 vs 候选默认 | 官方 vs 等价 role | 六组标准轨迹区域/注入 | 六组 × 全部支持 case 区域/注入 |
|---|---:|---:|---:|---:|
| Qwen3.5 | 28/28 | 28/28 | 6/6 | 168/168 |
| DeepSeek-V4.1 | 27/27 | 27/27 | 6/6 | 162/162 |
| GLM-5.3-Flash | 28/28 | 28/28 | 6/6 | 168/168 |

DeepSeek 的 `standalone_tool` 被独立官方明确拒绝：`NotImplementedError`，原生 agent 不支持没有前置调用的 standalone 工具返回。三方拒绝结果一致，六组角色配置也一致拒绝，报告单列 **7/7 expected_rejections**；这是一个不支持的输入加六组重复验证，不是七个额外成功编码样例。它不计入 27 或 162 的分母。

六组为 `equivalent`、`historical`、`fractional`、`amplified`、`answer_only`、`all_roles`。除 equivalent 外，其余组用于功能验证，**不要求改变训练目标后仍与官方默认权重相同**。正的 user/tool_response、不同的 assistant/tool_call 系数可发现默认零权重掩盖的错误。

独立 oracle 从原生渲染和原始消息分组生成区域预期，再比较真实 token 权重；不是以 role 输出反推期望，也不只证明注入前后自洽。校正了 GLM 嵌入 system 标记、tool_call 的 observation 终止符、并行批次内部换行和 DeepSeek EOS/直接调用前缀。允许相邻同权区域合并以容纳真实 BPE 边界；不同权重边界必须精确匹配。全部 token 明确分区，固定包装/空 think 为零，标签与有效权重共同检查。

### 5.3 原例与新增盲点

原始来源：`/data_train/vectorchen/datasets/agent_rollout_example/agent_role_loss_scale/agent_trajectory_standard_example.json`。

SHA256：`7853795b2381508a58729fce89bc039d2e55207d875094b8de30fb3355f37455`。原文件未修改，派生样例保存在每个模型的 `cases.json` 及注入 JSONL 中。

此前 14 类已包括标准/保留 thinking、`tool` 别名、工具后含字面 `{{QUERY}}` 的 user、多 user、空/重复正文、三种 think 边界、直接调用、standalone、屏蔽历史和 OpenAI 并行调用。本轮新增 14 类，最终 28 类：

| 新增 case | 盲点与原因 |
|---|---|
| `parallel_followups_1`、`parallel_followups_2` | 完整并行返回按 ID 对齐后再加单/双 user，避免只测试工具对齐或 user 合并各自正确 |
| `call_content_16`～`call_content_20` | 调用前 assistant 分别为缺失内容的 OpenAI 调用、空串、答案、只有非空 think、think 加答案，覆盖 prefix/分词/监督来源 |
| `thinking_False_False`、`thinking_False_True`、`thinking_True_False`、`thinking_True_True` | thinking 开关与历史保留组合；发现 Qwen 历史剥离吞分隔符 |
| `empty_history_no_prefix` | 禁止自动非 thinking 前缀时的空历史轮，检查没有正文却监督 EOS，以及空片段与工具前缀归属 |
| `masked_tool_history` | 原轨迹工具前 assistant 的 `loss=False`，检查工具本身仍可监督且 tokenization 不改变 |
| `empty_think_before_tool` | 空 think 紧接工具调用，检查标签 mask 与工具自身换行的边界 |

### 5.4 定向审查的官方近期更新

审查区间为共同祖先到上述官方 HEAD，只针对相关 template、agent、thinking、历史监督及工具规范化链路，不是审查全仓库所有提交。

| 官方提交 | 更新点 | 对应覆盖 |
|---|---|---|
| `69872d106` | 不同 agent 格式支持工具后 user | 原 followup 与新增并行 followup、空/重复正文来源 |
| `bdb874169` | 追加 user 时逐轮闭合 | 原生渲染字节、结束符、连续 user 边界 |
| `9d56934dc` | DeepSeek-V4.1 prompt 与 DSML 工具调用对齐 | V41 标准/直接调用、thinking 组合、原生返回与 EOS |
| `d85802da3` | OpenAI 工具结果先按 ID 对齐再规范化 | 逆序完整返回、注入角色、并行后 user |
| `10dc94bbb` | GLM4.5 正文与调用之间补换行 | assistant/tool_call 不同系数及来源单测；不是额外 GLM4.5 真实模型验收 |
| `e6c9dbe01` | `glm5_next` 模板修复 | GLM-5.3-Flash 的实际注册、原生格式和 thinking 保留 |
| `c2bcc23a1` | 历史 EOS 监督处理 | 普通文本 mask/空历史/工具历史；未补齐新的 token-backed 终止符重叠全矩阵 |

完整提交 hash、case 对应与审查边界存于 `cross-model/update-audit.json`。非法/缺失/重复工具 ID、Anthropic 预处理、混合显式并行调用权重和 token-backed 新终止符矩阵未在本轮完成全面闭环。

## 6. 历史验证轨迹：失败、修复与复验

阶段报告不可相加形成通过数，也不能把静态检查、超时或 skip 当运行通过。

| 阶段 | 实际发生的事情 | 证据与准确结论 |
|---|---|---|
| 20261118 迁移 | 22 文件增量覆盖、冲突合并；AST/JSON/TOML/Shell/空白与索引检查；builder 当时 14 项用例通过 | `/data_train/vectorchen/github/ms-swift-migration-20261118/`；后续修改后的候选不能直接复用该 14 项结论。loss 测试曾因缺依赖/环境 I/O 在收集或启动阶段失败 |
| 20261119 首轮 Qwen 闭环 | 修复前导关闭 think、工具分隔符权重、OpenAI 注入来源与 ID 对齐；修正官方不接受 fork 参数及 tiny 模型夹具 | `/data_train/vectorchen/github/ms-swift-validation-20261119/run3/report.json`；14/14 官方默认及 role 等价，六组标准区域检查、84/84 注入；`regression3.log/xml` 为 93 passed、132 subtests passed、3 skipped。仅 Qwen 阶段 |
| 后续 CPU/缓存集成尝试 | 新增集成测试，静态检查通过；后台 600 秒后退出 124，日志为空，无 JUnit/collection/断言结果；观察到文件系统 I/O 等待，具体阻塞 import 未确认 | `continuation/validation-continuation.json`、`integration2.exit/log`；没有通过、主动 skip 或断言失败证据，不用于跨模型验收 |
| 跨模型初始复现 | DeepSeek processor 返回类型假设错误；新增空历史样例放到 system 前导致官方拒绝；Qwen 新 thinking 组合暴露分隔符被吞 | `cross-model/repro-*`；修正接口、合法消息顺序及实现，保留失败记录 |
| 跨模型 `run1-*` | 完整独立区域检查发现 V41 EOS/直接调用前缀，以及 oracle 对并行批次、GLM 内嵌头和 BPE 同权边界的误判 | 根据官方独立编码与原生格式区分实现缺陷和 oracle 缺陷；没有按实际输出直接修改预期来伪造通过 |
| 核心回归前几轮 | 首次引用不存在的测试路径，退出 4；后两轮 12 个新夹具失败，先误把注册表类当实例，后错误传构造参数 | `cross-model/regression.log`、`regression2.log`、`regression3.log`；按真实无参 agent 工厂修正，未跳过失败用例 |
| 跨模型 `run2-*` | Qwen/GLM 全部通过；V41 剩 `masked_tool_history` 严格对照差异和空历史工具分隔符区域差异 | 修复 masked assistant 不吸收工具前缀，并根据原始消息组完善独立 oracle，之后三模型全部重跑 |
| 修正夹具后的回归 | `regression4` 为 126 passed、132 subtests passed、3 skipped | 已通过，但不是后续最终修复的唯一证据；又执行 `regression5` |
| 最终稳定源码复验 | 9 个独立官方/default/role worker 并行、各 240 秒上限；3 个汇总进程检查退出码和 case/config 清单；核心回归限时 300 秒 | `cross-model/run3-{qwen,deepseek,glm}/`；9 worker 与3 collector 均退出 0，三份报告均 `passed=true`；`regression5` 退出 0 |

最终回归实际运行的文件：

- `tests/loss_scale/test_role_loss_scale.py`。
- `tests/loss_scale/test_role_migration.py`。
- `tests/utils/test_tool_user_followup.py`，官方原有测试，未修改。
- `tests/general/test_agent_template_tool_arg_types.py`，官方原有测试，未修改。
- `tests/general/test_role_cached_builder.py`。
- `tests/general/test_dataset_cache_safety.py`，官方原有测试，未修改。

`regression5.log/xml` 为 **126 passed、132 subtests passed、3 skipped、0 failures、0 errors**，运行约 140 秒。3 个 skip 消息均为未提供本地 Hunyuan-1.8B tokenizer；不影响 Qwen、DeepSeek-V4.1、GLM 的真实对照，但不能据此说全部测试均未跳过。V31/GLM4.5/4.7 等额外格式的局部单测通过，不等于相应真实模型完整验收。

### 6.1 环境与执行安全

验证节点为 `gpu-node-112`，Python 为 `/data_train/vectorchen/env/GLM/baseline/bin/python`。最终报告记录 `torch==2.14.1+cu132`、`transformers==5.16.1`、`tokenizers==0.23.3`、`qwen_vl_utils==0.0.14`、`decord==0.6.0`、`pytest==9.2.0`、`kernels==0.18.3`。

此前 Qwen 环境准备阶段按授权新增了缺失依赖；跨模型最终阶段及本文新增阶段没有再安装依赖。验证使用离线、禁 GPU、限 CPU 线程设置，加载的是模型目录 tokenizer/processor/config，不加载大模型权重。Swift 基线仍可能打印 `trust_remote_code=True` 提示；验证器在加载前拒绝模型目录自定义 Python 与 `auto_map`，本次三个原生目录通过检查，没有执行模型仓库自定义代码。不能将这个原生目录约束宣称为对任意远端模型代码的普遍安全保证。

任务采用后台限时运行与短轮询，失败退出码、初始化阻塞、原始失败样例和最终复验都保留，没有安装新版本来绕过断言、没有提交/推送或改动独立官方仓库。

## 7. 归档、重放与复验入口

### 7.1 最终证据位置

根目录：`/data_train/vectorchen/github/ms-swift-validation-20261119/cross-model/`。该目录在仓库外；迁移本仓库时需同时保留证据，下面文件名不代表仓库已自带全部运行产物。

| 文件/目录 | 含义 |
|---|---|
| `validation-cross-model-final.json` | 三模型最终汇总、修复、通过范围及未验收契约 |
| `run3-qwen/report.json`、`run3-deepseek/report.json`、`run3-glm/report.json` | 最终逐 case 无回退、严格等价、六组区域/注入、一致拒绝明细 |
| 各 `run3-*` 内 `official.json`、`fork_default.json`、`role.json` | 实际 IDs、labels、有效权重、解码与独立导入信息 |
| 各 `run3-*` 内 `cases.json`、`configs.json`、`config-*.json`、注入 JSONL | 合法派生样例、六组准确系数和物化输入 |
| 各 worker 的 `.log/.exit`、`collect.log`、`run.exit` | 实际执行日志和退出码，不仅是报告的布尔结论 |
| `regression5.log`、`regression5.xml`、`regression5.exit` | 最终核心回归与 skip 原因 |
| `model-manifest.json` | 三个真实模型配置/tokenizer 指纹与实际模板选择 |
| `update-audit.json` | 官方相关提交、case 与未覆盖边界 |
| `run3-source-snapshot.json` | 最终编码及回归期间 31 个候选文件与原样例未改变的核对依据 |
| `migration-validated-cross-model.patch`、`patch-replay.json` | 相对官方 HEAD 的完整已验证候选补丁及独立 index 重放证明 |
| `repro-*`、`run1-*`、`run2-*`、较早 regression 日志 | 真实失败、定位与修复过程，不作为最终成功证据 |

已验证补丁 SHA256 为 `44ad96ef75ea01803d6b1946a13a39b2ace9214a39e240ac998d7f19b36b5883`。使用专用临时 Git index 重放到官方 HEAD，树为 `3051868165ba0d2cc2173af0d58240eb6124b41f`，31 个候选文件与工作树逐文件一致，原始样例哈希未变；没有触碰用户 index，临时 index 已清理。

**该补丁不包含随后新增的本文，也不包含未完成运行验收的 `test_role_training_integration.py`；仍含现已撤销的 V31 适配及其测试。**补丁与旧报告原样保留为历史证据，不能直接作为恢复后的当前交付补丁；重新应用它会带回 V31 修改。不要将当前工作树自动视为历史31文件版本已经全部重验，修改任一被测源码、配置或样例后需重新核对/复验。

### 7.2 复验参数

入口为 [validate_role_equivalence.py](../validate_role_equivalence.py)。需使用上述兼容 Python 环境、独立官方与候选仓库、合法原例、真实模型目录，并选择第 5.1 节准确的 `--template-type` 和 `--agent-template`。

常规模式会依次运行三方；最终历史验收为九个独立 worker 并行后汇总。`--worker-timeout` 控制单 worker 上限；`--prepare-only` 仅写清单不测试，`--equivalence-only` 不包含完整非二值区域验收，`--collect-only` 仅核对并汇总已有清单/结果，不重新运行 worker。不能把这些局部模式的结果当作全部验收。

复跑应指定**新的输出目录**，保留本次历史证据；采用后台硬超时和短轮询，不使用长时间无输出的前台等待。本文不自动执行复验命令或真实缓存生产任务。

## 8. 仍需独立验收的事项

1. 三个真实大模型权重的 forward/backward 与完整训练 smoke test；tiny Qwen loss/梯度单测不能代替。
2. `test_role_training_integration.py` 的实际运行：完整 Trainer/SGD 一步、左右截断、普通 CLI cache 和 encoded Arrow cache 各自的保存/重载；此前仅有静态检查与启动超时记录。
3. 新模型格式的真实 tokenizer 对照，特别是 DeepSeek V3.1；共享继承/局部单测不是全模型证明。
4. 真实分布式缓存任务领取、heartbeat、恢复、export、复制、合并及训练启动侧 manifest 核对；历史 TOML 不代表当前生产环境。
5. 多模态、Jinja 训练后端、RLHF、GPU SP/CP/Megatron 和多节点路径。
6. 全量显式消息权重覆盖、非法/不完整工具 ID、Anthropic 输入、混合显式并行调用权重及 token-backed 终止符重叠矩阵。

补充能力说明见 [Agent-role-loss-scale.md](../../../docs/source/Customization/Agent-role-loss-scale.md)，原迁移过程见 [Loss-scale-migration-20261118.md](../../../docs/source/Customization/Loss-scale-migration-20261118.md)。这些链接用于追溯，不替代本文第 5～7 节明确的最终证据与验收边界。
