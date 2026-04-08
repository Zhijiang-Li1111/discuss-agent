# 005 — Prompt Optimization (Anthropic 三剑客标准)

> 状态：Draft
> 前置：spec-001~004 已实现
> 范围：按 Anthropic 三剑客标准重写框架内所有 prompt 和 tool description，不改引擎逻辑

---

## 1. 概述

discuss-agent 的引擎逻辑已经完备（spec-001~004），但**讨论质量取决于 prompt 质量**。当前的 prompt 存在多处偏离 Anthropic 最佳实践的问题，直接影响讨论产出。

本 spec 按三剑客标准（agent-prompt-guidelines / tool-description-standard / skill-writing-standard）逐一审计和重写：

1. **引擎内嵌 prompt**（express/challenge/host judgment/summary/compressor）
2. **YAML 用例 prompt**（agent system_prompt / host convergence/summary prompt）
3. **Tool descriptions**（research-pipeline 的 4 个 tool）

原则：**只改 prompt 文本，不改引擎代码逻辑。** 讨论循环、并行执行、错误处理、持久化全部不变。

---

## 2. 审计发现

### 2.1 引擎内嵌 prompt 问题

| 位置 | 当前代码 | 问题 | 三剑客原则 |
|------|----------|------|------------|
| `_express()` | `"第{n}轮：请分享你的分析和观点。"` | 太薄，没给角色 context，没说 Tool 可用，没说输出期望 | "把 agent 当聪明但缺上下文的新员工" |
| `_challenge()` | `"请审视上述观点，提出质疑或补充。"` | 没引导"死守立场"、没说要引用证据反驳、"补充"这个词鼓励和稀泥 | "给原则不给模式" |
| `_host_judge()` | `"请判断讨论是否收敛。返回JSON格式。"` | 没重申收敛标准、没给 JSON schema 示例、没说"不确定就判未收敛" | Poka-yoke 防错设计 |
| `_host_summarize()` | `"请生成最终选题报告。"` | 太泛，没指定输出结构，没告知讨论背景 | "给 context 让模型推理" |
| `_compress_round()` | `"将以下讨论轮次的发言浓缩为一段简洁的摘要..."` | 可以更好，但基本ok | — |

### 2.2 YAML 用例 prompt 问题（research-pipeline 选题配置）

| Agent | 问题 |
|-------|------|
| 热点猎手 system_prompt | 结构合理，但缺少"你可以用 Tool 查研报/热榜"的提示，Agent 可能不知道有 Tool |
| 深度研究员 system_prompt | 同上，且"引用研报原文"没说怎么获取原文（需要调 read_content Tool） |
| Host convergence_prompt | 收敛标准写得不错，但 JSON schema 只在 user message 里提了一次，没有 few-shot 示例 |
| Host summary_prompt | "每一部分都要具体、可执行"太宽泛，没给输出模板 |

### 2.3 Tool Description 问题（research-pipeline）

| Tool | 当前 | 问题 |
|------|------|------|
| `list_research` | 有 What/When/Parameters/Returns | ✅ 基本符合，但 `_CATEGORY_KEYWORDS` 过滤逻辑对 Agent 不透明 |
| `read_content` | 有完整 description | ✅ 基本符合，但没说"用 list_research 的结果来构造 file_path" |
| `get_trending` | 有 description | ⚠️ 没说返回什么格式、降级时 Agent 该怎么办 |
| `get_published` | 有 description | ⚠️ 没说返回格式是 `date | topic | argument` |

---

## 3. 设计

### 3.1 引擎 prompt 重写

引擎中的 prompt 是拼到 user message 里的，不是 system prompt。重写原则：

- **Express prompt**：告知 Agent 当前轮次、可用 Tool、输出期望（有论据的观点），不限制具体格式
- **Challenge prompt**：明确"反驳需要证据或逻辑"、"不做礼貌性认可"、必须回应指名的质疑
- **Host judge prompt**：重申收敛标准、给 JSON few-shot 示例、明确"不确定就判未收敛"
- **Host summary prompt**：给输出模板（4 个 section）、提供讨论背景

具体文本见 §4。

### 3.2 YAML 用例 prompt 指导

本 spec 不硬编码 YAML prompt 的具体文本（那是 research-pipeline 的事），但给出**改写指南**和**示例**，供 spec-006（research-pipeline 调优）执行。

指导原则：
- Agent system_prompt 必须告知"你有哪些 Tool 可用，什么时候用"
- Agent system_prompt 用自然语言描述原则，不用规则列表
- Host prompt 给 JSON 示例（收敛 + 未收敛各一个）
- Summary prompt 给输出模板

### 3.3 Tool Description 改写

按 tool-description-standard 的 5 要素重写所有 tool 的 docstring。改写在 research-pipeline 仓库执行，但标准和范例在本 spec 中定义。

---

## 4. 引擎 Prompt 重写（精确文本）

### 4.1 Express Prompt

**当前：**
```python
prompt = f"{context}\n\n{history_text}\n\n第{round_num}轮：请分享你的分析和观点。"
```

**重写为：**
```python
prompt = (
    f"{context}\n\n"
    f"{history_text}\n\n"
    f"--- 第{round_num}轮 表达 ---\n\n"
    f"请基于上述背景信息和讨论历史，阐述你的分析和判断。\n\n"
    f"要求：\n"
    f"- 每个论点都要有具体的证据或数据支撑。如果需要查阅原始资料，使用你的工具获取。\n"
    f"- 明确你的核心立场——一句话能概括的判断。\n"
    f"- 如果你对之前某位参与者的观点有问题，直接点名提出（如"@深度研究员 你提到…"）。"
)
```

**改动理由：**
- 加了"使用你的工具获取"——Agent 不一定知道自己有 Tool
- "每个论点都要有证据"——引导事实驱动
- "明确核心立场"——防止输出散漫
- "点名提出"——引导定向交锋，防止空对空

### 4.2 Challenge Prompt

**当前：**
```python
prompt = (
    f"以下是其他讨论者在第{round_num}轮的观点：\n\n"
    f"{others_text}\n\n请审视上述观点，提出质疑或补充。"
)
```

**重写为：**
```python
prompt = (
    f"以下是其他参与者在第{round_num}轮的表达：\n\n"
    f"{others_text}\n\n"
    f"--- 第{round_num}轮 反驳 ---\n\n"
    f"请逐一审视上述观点。对每位参与者的发言：\n\n"
    f"- 如果你认为对方的论据不充分、逻辑有漏洞或数据有问题，直接反驳并给出你的反面证据。\n"
    f"- 如果对方用了无法反驳的事实或逻辑击穿了你之前的论点，明确承认："我的[具体论点]被击穿了，因为[具体原因]。"\n"
    f"- 不做礼貌性认可——不说"你说的有道理但是"。要么被击穿就承认，要么没有就直接反驳。\n"
    f"- 如果对方向你提出了具体问题（@你），必须正面回答，不能回避。\n"
    f"- 需要验证对方引用的数据时，使用你的工具查阅原始资料。"
)
```

**改动理由：**
- 删掉了"补充"（鼓励和稀泥）
- 加了"不做礼貌性认可"——这是圆桌 skill 验证过的核心规则
- 加了"被击穿就承认"——强制二元态
- 加了"必须回答@你的问题"——防止选择性无视
- 加了"用工具查阅"——反驳需要事实

### 4.3 Host Judge Prompt

**当前：**
```python
prompt = (
    f"以下是讨论记录：\n\n{history_text}\n\n"
    f"请判断讨论是否收敛。返回JSON格式。"
)
```

**重写为：**
```python
prompt = (
    f"以下是完整的讨论记录：\n\n{history_text}\n\n"
    f"--- 收敛判断 ---\n\n"
    f"请判断讨论是否已收敛。收敛的含义是：各方不再提出实质性的新挑战，"
    f"核心结论趋于稳定。注意：\n\n"
    f"- 不同角度共存不等于未收敛——只要不再有根本性分歧即可。\n"
    f"- 如果最近一轮的反驳只是在重复已有论点或做细节修饰，这说明已收敛。\n"
    f"- 如果某方明确承认自己的立场被击穿，这是收敛的强信号。\n"
    f"- 如果你不确定是否收敛，判为未收敛。\n\n"
    f"返回以下 JSON 格式（只返回 JSON，不要其他内容）：\n\n"
    f'{{"converged": true, "reason": "各方已就X达成共识，最近一轮无实质性新挑战", "remaining_disputes": []}}\n\n'
    f"或者：\n\n"
    f'{{"converged": false, "reason": "A方的X论点尚未被充分回应", "remaining_disputes": ["具体分歧点1", "具体分歧点2"]}}'
)
```

**改动理由：**
- 重申了收敛标准（之前只在 system_prompt 里写了一次）
- 给了两个 JSON few-shot 示例（收敛 + 未收敛）——Anthropic 数据显示 examples 可将准确率从 72% 提升到 90%
- "只返回 JSON"——Poka-yoke，减少解析失败
- "不确定就判未收敛"——安全默认值

### 4.4 Host Summary Prompt

**当前：**
```python
prompt = f"以下是完整的讨论记录：\n\n{history_text}\n\n请生成最终选题报告。"
```

**重写为：**
```python
prompt = (
    f"以下是完整的讨论记录：\n\n{history_text}\n\n"
    f"--- 最终总结 ---\n\n"
    f"请基于全部讨论内容生成结构化的总结报告。报告必须包含以下四个部分：\n\n"
    f"## 1. 核心结论\n"
    f"讨论最终达成的核心共识是什么？如果有多个结论，按重要性排序。\n\n"
    f"## 2. 关键论据与证据\n"
    f"支撑核心结论的关键论据，标注提出者和数据来源。\n\n"
    f"## 3. 被击穿的论点\n"
    f"讨论中哪些论点被反驳方成功击穿？被谁击穿？用什么击穿的？\n\n"
    f"## 4. 未解决的分歧\n"
    f"如果存在未完全收敛的分歧，如实记录各方立场，供决策者裁决。"
)
```

**改动理由：**
- 输出模板比"请生成报告"精确得多——Agent 有明确的 section 可以填
- 模板是通用的（不绑死选题场景），适用于任何讨论用例
- "被击穿的论点"单独列出——这是对抗式讨论的核心价值
- "标注提出者和数据来源"——可追溯

### 4.5 Compressor Prompt

**当前基本可用，微调：**

**当前：**
```python
system_message=(
    "你是一个讨论记录压缩助手。"
    "将以下讨论轮次的发言浓缩为一段简洁的摘要，"
    "保留关键论点、证据和分歧。不要添加新信息。"
)
```

**重写为：**
```python
system_message=(
    "你是讨论记录压缩助手。将一轮讨论的完整发言浓缩为一段摘要（200-400字）。"
    "必须保留：每位参与者的核心立场、关键证据和数据、被击穿的论点、未解决的分歧。"
    "不得添加讨论中未出现的信息。不得评价参与者的观点。"
)
```

**改动理由：**
- 加了字数范围——防止压缩后仍然太长或太短
- "每位参与者的核心立场"——压缩不能丢掉谁说了什么
- "不得评价"——压缩器不是裁判

---

## 5. YAML 用例 Prompt 改写指南

以下指南适用于 research-pipeline 的 `configs/topic_selection.yaml`。具体文本在 spec-006 中定义，这里只给标准和示例。

### 5.1 Agent System Prompt 标准

每个 Agent 的 system_prompt 必须包含：

1. **角色定义**（2-3 句自然语言，不是一句话标签）
2. **核心原则**（3-5 条，每条带 WHY）
3. **Tool 使用指引**（明确列出可用的 Tool 和使用场景）
4. **讨论态度**（什么时候坚持、什么时候让步的原则）

**不要写的：**
- 规则列表（`MUST`, `ALWAYS`, `NEVER`）
- if-then 决策树
- 压缩箭头语法

**示例改写（热点猎手 Tool 指引部分）：**

```
## 工具使用

你有两类工具可以使用：
- 研报标题列表：查看最近的研报有哪些，用于发现研报中的潜在热点话题。
- 研报内容读取：当你需要验证某份研报是否真的包含有价值的内容时，读取全文确认。

讨论中如果对方质疑你的判断，先查阅原始资料再回应，不要凭印象辩论。
```

### 5.2 Host Prompt 标准

- Convergence prompt：收敛标准 + JSON schema + 两个 few-shot（收敛/未收敛）
- Summary prompt：输出模板（4 section）+ 内容要求

（引擎 prompt 已包含通用版本；YAML 中的 system_prompt 提供场景特定的维度和标准。）

---

## 6. Tool Description 改写标准

按 tool-description-standard 的 5 要素，每个 tool 的 docstring 必须包含：

1. **What** — 做什么
2. **When** — 什么时候用（什么时候不用）
3. **Parameters** — 参数含义和约束
4. **Returns** — 返回格式和限制
5. **Limitations** — 不做什么

格式：纯文本散文，至少 3-4 句，不用 markdown。

**示例改写（list_research）：**

当前 docstring 已接近标准，但缺少 When（不用）和 Limitations。改写后应包含：
- When to use: 讨论开始前了解近期有哪些研报可选
- When NOT to use: 不要用来获取研报内容（用 read_content）
- Limitation: 只扫描外资和内资研报目录中的行业/宏观/策略分类，个股研报不返回

**示例改写（read_content）：**

需要增加：
- When to use: 需要验证某份研报的具体内容、数据或结论时
- How to get file_path: 从 list_research 返回的条目推断文件路径，格式为"研报目录/外资或内资目录/子分类/YYYYMMDD-机构-标题.pdf"
- Limitation: 只能读取 research_dir 内的文件；扫描结果是提取文本而非格式化内容，表格和图表可能丢失

---

## 7. 验收标准

### 引擎 Prompt（discuss-agent 仓库）

- **AC-1.1**: `_express()` 的 prompt 包含"使用你的工具"提示、"证据支撑"要求、"核心立场"要求
- **AC-1.2**: `_challenge()` 的 prompt 包含"不做礼貌性认可"、"被击穿就承认"、"必须回答@你的问题"、"使用工具查阅"
- **AC-1.3**: `_host_judge()` 的 prompt 包含收敛标准重申、两个 JSON few-shot 示例（收敛 + 未收敛）、"不确定就判未收敛"
- **AC-1.4**: `_host_summarize()` 的 prompt 包含 4 个 section 的输出模板（核心结论/关键论据/被击穿的论点/未解决的分歧）
- **AC-1.5**: `_compress_round()` 的 system_message 包含字数范围、保留项清单（立场/证据/被击穿/分歧）
- **AC-1.6**: 所有 prompt 使用自然语言，不含 `MUST`/`ALWAYS`/`NEVER`/`CRITICAL` 等强调词

### Tool Description（research-pipeline 仓库）

- **AC-2.1**: `list_research` docstring 包含 5 要素（What/When/When NOT/Parameters/Returns/Limitations）
- **AC-2.2**: `read_content` docstring 告知 Agent 如何从 list_research 结果推断 file_path
- **AC-2.3**: `get_trending` docstring 说明降级场景下 Agent 应如何继续
- **AC-2.4**: `get_published` docstring 说明返回格式（date | topic | argument）
- **AC-2.5**: 所有 tool docstring 使用纯文本散文，不用 markdown 格式

### 通用

- **AC-3.1**: 引擎逻辑（循环、并行、错误处理、持久化）零改动
- **AC-3.2**: 所有现有测试通过（prompt 变更不应影响测试，因为测试 mock 了 LLM 调用）
- **AC-3.3**: spec-005 的 prompt 文本在代码中可被直接定位和 review（不分散在多个文件中的字符串拼接里——如果当前代码已经是这样，保持不变即可）

---

## 8. 不变条件

- 讨论循环逻辑不变
- 并行执行不变
- 错误处理策略不变
- 持久化归档不变
- Context 压缩策略不变（只改 compressor 的 system_message 措辞）
- Host 无 Tools 约束不变
- YAML 配置格式不变
- Tool 的功能和行为不变（只改 docstring/description）
