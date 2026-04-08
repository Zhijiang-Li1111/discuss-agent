# Agent Prompt & Tool Design Guidelines

> 基于 Anthropic 官方文档和工程实践的指导文件。
> 用于指导 LLMDrawIO 的 agent prompt 重构。

Sources:
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) — Erik Schluntz & Barry Zhang
- [Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)
- [Prompt Engineering Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Tool Use Implementation Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
- [Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)

---

## 1. System Prompt 设计

### 1.1 核心原则

> "Show your prompt to a colleague with minimal context on the task and ask them to follow it. If they'd be confused, Claude will be too."

把 Claude 当成一个**极其聪明但缺乏项目上下文的新员工**。需要的是 context，不是微操指令。

### 1.2 结构

推荐的 system prompt 结构：

```
1. Role — 身份、能力边界、行为风格（自然语言，不是一句话）
2. Principles — 少数关键原则 + WHY（不是规则列表）
3. Context — 精简的环境状态（详细信息按需查询）
4. Constraints — 真正的硬约束（安全、格式）
```

**长文本数据（如 diagram state）放在 prompt 顶部**，查询/指令放在底部。Anthropic 测试显示查询在末尾可提升约 30% 响应质量。

### 1.3 写法指南

| DO | DON'T |
|---|---|
| 说明 WHY，不只是 WHAT。Claude 会从解释中泛化 | 只列规则不解释原因 |
| 告诉 Claude 做什么 | 告诉 Claude 不做什么 |
| 自然语言描述原则 | 压缩箭头语法 `A → B` |
| 给 context 让模型推理 | 硬编码 if-then 决策树 |
| `"Use this tool when..."` | `"CRITICAL: You MUST use this tool..."` |

**反过度提示（Anti-Over-Prompting）**：Claude 4.5/4.6 对 system prompt 极其敏感。以前为防止 undertriggering 写的强调语言（`CRITICAL`, `MUST`, `ALWAYS`），在新模型上会导致 **overtriggering**。用平实的自然语言。

### 1.4 System Prompt 里应该放什么

- ✅ Role 定义（身份、风格、能力边界）
- ✅ 少数跨工具的路由原则（带 WHY）
- ✅ 交互风格指导（响应语言、格式偏好）
- ✅ 高层工具分类导航（有哪些工具类别）
- ❌ 具体工具使用规则（放 tool description）
- ❌ 大量 if-then 决策规则（让模型推理）
- ❌ 全量数据 dump（按需查询）

---

## 2. Tool Description 设计

### 2.1 核心原则

> "Descriptions are the most critical factor in tool performance."
> "For SWE-bench agent, spent MORE time optimizing tools than overall prompt."

Tool description 是 agent 性能的**最关键因素**。投入在 tool description 上的精力应该等于甚至超过 system prompt。

> "Think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good agent-computer interfaces (ACI)."

### 2.2 一个完整的 tool description 应包含

1. **What** — 这个工具做什么
2. **When** — 什么场景用它（以及什么场景**不**用它）
3. **Parameters** — 每个参数的含义和约束
4. **Returns** — 返回值的精确格式和限制
5. **Limitations** — 工具不做什么（避免幻觉假设）

目标：**至少 3-4 句话**，复杂工具更多。

### 2.3 好 vs 坏的例子

**Bad:**
```
"Gets the stock price for a ticker."
```

**Good:**
```
"Retrieves the current stock price for a given ticker symbol.
The ticker symbol must be a valid symbol for a publicly traded company
on a major US stock exchange like NYSE or NASDAQ. The tool will return
the latest trade price in USD. It should be used when the user asks
about the current or most recent price of a specific stock. It will
not provide any other information about the stock or company."
```

### 2.4 Poka-Yoke（防错设计）

> Anthropic found models made mistakes with relative filepaths; changed requirement to absolute filepaths. Result: flawless usage.

- 让错误在**结构上不可能发生**
- 用 `enum` 限制输入为有效值
- 用描述性参数名让意图一目了然
- 把它当成 "给初级开发者写文档"

### 2.5 工具整合

- **合并相关操作**为一个工具 + `action` 参数，而非拆成多个独立工具
  - ✅ `vertex(action="create"|"update"|"delete"|"move")`
  - ❌ `create_vertex`, `update_vertex`, `delete_vertex`, `move_vertex`
- 有意义的命名空间：`github_list_prs`, `slack_send_message`

### 2.6 Tool Response 设计

- 返回**语义化、稳定的标识符**（slug, UUID），不要不透明的内部引用
- 只包含 Claude 需要的字段（做下一步决策所需的信息）
- 避免臃肿的响应浪费 context
- 失败时设置 `is_error: true`

---

## 3. Context 管理

### 3.1 核心原则

> Agent 必须在每一步获取 "ground truth" — 通过工具结果、代码执行等。

不要假设状态，要查询状态。

### 3.2 注入 vs 按需查询

| 策略 | 适用场景 |
|---|---|
| **注入到 prompt** | 轻量、每轮都需要的信息（节点数、选中状态） |
| **按需查询（tool call）** | 重量级数据（完整 diagram state、节点详情） |
| **Observation masking** | 旧的工具输出替换为一行摘要，保留推理过程 |

### 3.3 Context Window 优化

Anthropic 观察到实际项目中 tool definitions 可达 **134K tokens**。策略：

- **Deferred tool loading**（工具搜索）— 常用 3-5 个始终加载，其余按需。可减少 ~85% context 消耗
- Diagram context 只注入 summary（类型、节点数、选中元素），不要全量 dump
- 用结构化格式存储中间状态（JSON for task status, git for checkpoints）

### 3.4 不要制造人为焦虑

- ❌ `Step 3 / 15. ⚠️ Approaching limit. Complete quickly.`
- ✅ 让模型专注于做对事情。如需限制迭代，用代码层面的 hard stop，不要用 prompt 施压

---

## 4. Agent 架构原则

### 4.1 核心哲学

> "Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs."

**从简单开始，只在能证明改善结果时才增加复杂度。**

### 4.2 Workflow vs Agent

| | Workflow | Agent |
|---|---|---|
| 控制流 | 代码预定义 | LLM 动态决定 |
| 适用 | 步骤可预测的任务 | 开放式问题 |
| 成本 | 低，可预测 | 高，可能复合错误 |

### 4.3 Workflow 模式（按复杂度递增）

1. **Prompt Chaining** — 固定顺序子任务，中间加 gate 校验
2. **Routing** — 分类输入，路由到专用 prompt/model
3. **Parallelization** — 独立子任务并行（sectioning）或同任务多次（voting）
4. **Orchestrator-Workers** — 中央 LLM 动态分配子任务
5. **Evaluator-Optimizer** — 生成 + 评估循环

### 4.4 反模式

- ❌ 简单 LLM + retrieval 能解决的场景用 agentic system
- ❌ 一个 LLM call 处理多个关注点（用 sectioning 拆分更好）
- ❌ 用框架但不理解底层代码
- ❌ 没有 sandbox 测试就部署 agent
- ❌ 过度工程：创建额外文件、不必要的抽象、没被要求的灵活性

---

## 5. 对 LLMDrawIO 的具体建议

### 5.1 System Prompt 重构

**当前问题：**

```
# Role
You are a diagram editing assistant.    ← 太薄

# Behavior
- Position 1-3 nodes → vertex move...   ← 硬编码规则，压缩语法
- Add single element → vertex/edge.     ← 应在 tool description
- Complex task (3+ steps) → plan...     ← if-then 替代推理
- Style multiple → ONE style call.      ← 应在 tool description
- Query diagram state before changes.   ← 没解释 WHY

# Current Diagram
```json                                  ← 全量 dump
{...thousands of tokens...}
```

# Iteration
Step 3/15. ⚠️ Complete quickly.          ← 人为焦虑
```

**建议方向：**

```
# Role
你是一个 Draw.io 图表编辑助手。你通过工具来创建、修改和排版图表。
Draw.io 是唯一的状态源（SSOT）— 用户可能在对话期间手动修改了图表，
所以在修改前先查询当前状态。

# Principles
- 先理解再行动：创建图表前，先判断图表类型（流程图、架构图、思维导图等），
  加载对应的 skill 获取领域知识。
- 批量优于逐个：工具支持批量操作，一次调用完成多个变更，而非逐个操作。
- 布局交给算法：超过 3 个节点的排版使用 layout 工具，不要手动计算坐标。

# Context
{{lightweight diagram summary — node count, types, selection}}
```

### 5.2 Tool Description 增强

把当前 system prompt 中的规则下沉到对应工具：

| 当前在 system prompt | 移到哪里 |
|---|---|
| "Style multiple elements → ONE style call" | `style` tool description |
| "Move/delete multiple → ONE batch call" | `batch` tool description |
| "Position 1-3 nodes → vertex move" | `vertex` tool description |
| "Coordinates inside containers are relative" | `vertex` + `batch` tool description |
| "Complex task → plan tool first" | `plan` tool description |
| "Create → determine type → load skill" | `load_skill` tool description |

### 5.3 Diagram Context 精简

当前：每轮注入完整 JSON dump（可达数千 token）

建议：
- 注入轻量 summary：`"Diagram: 12 vertices, 15 edges, 2 containers. Selection: node 'Validate Input'"`
- 提供 `get_diagram_state` 工具让 agent 按需查询完整状态
- 或者在首轮注入完整状态，后续轮次只注入 summary

### 5.4 去掉 Iteration 压力

当前：`Step 3/15. ⚠️ Approaching limit.`

建议：
- 删除 step 计数和警告
- 用代码层面的 hard stop（`MAX_ITERATIONS`）控制，不在 prompt 层面施压
- 如果确实需要提示，用中性语言：`"This is iteration 8. If the task is mostly complete, consider wrapping up."`

---

## 6. 关键语录

> "Descriptions are the most critical factor in tool performance."

> "For SWE-bench agent, spent MORE time optimizing tools than overall prompt."

> "Think about how much effort goes into HCI, and plan to invest just as much effort in creating good ACI."

> "Put yourself in the model's shoes — is it obvious how to use this tool?"

> "Start simple and add complexity only when demonstrably improving outcomes."

> "Ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error."
