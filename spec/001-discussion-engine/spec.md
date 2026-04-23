# 001 - 多 Agent 对抗式讨论引擎

> 来源：DESIGN.md v0.3
> 状态：待审核
> 范围：核心框架 + 选题用例配置（不含深度调研、写作、发布模块）

---

## 1. 概述

实现一个通用的多 Agent 对抗式讨论框架。框架通过 YAML 配置驱动，支持 N 个讨论 Agent 并行表达、交叉反驳，由 Host Agent 判断收敛并生成最终总结。

第一个用例：研报选题讨论（从 IMA 研报 + 热榜数据中筛选公众号选题）。

---

## 2. 系统边界

**In scope（本 spec）：**
- 讨论循环引擎（表达 → 反驳 → Host 判断 → 循环/退出）
- 跨 Agent 共享 Context 管理与压缩
- YAML 配置加载 → Agno Agent 实例化
- 讨论记录持久化归档
- 4 个 Tools 定义（研报列表、研报内容、热榜、已发布历史）
- 选题用例的 YAML 配置文件
- CLI 入口

**Out of scope：**
- 深度调研系统
- 财经写作 Agent
- 发布流程
- IMA MCP Server 集成（待接入，v2）
- 质量反馈回路（v2）
- Agent 同质化对策（v2，观察实际效果后再定）
- 研报标题预处理标准化（v2）

---

## 3. 开放问题决议（本版）

| 问题 | 决议 | 理由 |
|------|------|------|
| 最大/最小轮数 | min=2, max=5，YAML 可配 | DESIGN.md 建议值，跑了再调 |
| 压缩触发时机 | 接近 context window 上限时触发 | 前几轮不浪费成本做无谓摘要，只在真正需要时压缩 |
| 质量反馈回路 | defer v2 | 需要先有数据积累 |
| 同质化风险 | defer v2 | 需观察实际效果 |
| 研报标题预处理 | defer v2 | 先用原始标题 |

---

## 4. 架构

### 4.1 模块划分

```
discuss_agent/
  ├── __init__.py
  ├── engine.py          # DiscussionEngine: 讨论循环主逻辑
  ├── context.py         # ContextManager: 共享 Context 注入 + 压缩
  ├── config.py          # ConfigLoader: YAML → 数据类
  ├── persistence.py     # Archiver: JSON/MD 归档
  ├── tools/
  │   ├── __init__.py
  │   ├── research_list.py   # 研报标题列表
  │   ├── research_content.py# 研报内容读取
  │   ├── trending.py        # 热榜数据
  │   └── published.py       # 已发布历史
  └── main.py            # CLI 入口
configs/
  └── topic_selection.yaml   # 选题用例配置
```

### 4.2 依赖

- `agno`：Agent 类、Tool calling、Session Summary、SqliteDb、MCP
- `anthropic`：Claude API（通过 Agno 间接使用）
- `pyyaml`：配置解析
- `asyncio`：并行执行
- `pdftotext`（系统依赖，poppler-utils）：research_content Tool 需要

### 4.3 核心数据结构

```python
@dataclass
class AgentUtterance:
    agent_name: str
    content: str

@dataclass
class RoundRecord:
    round_num: int
    expressions: list[AgentUtterance]       # Step 1 表达
    challenges: list[AgentUtterance]        # Step 2 反驳
    host_judgment: dict | None              # Host 收敛判断 JSON
    is_summary: bool = False                # True = 该轮已被压缩为摘要
    summary_text: str | None = None         # 压缩后的摘要文本

@dataclass
class DiscussionResult:
    converged: bool                         # 是否收敛
    rounds_completed: int                   # 完成的轮数
    archive_path: str                       # 归档目录路径
    summary: str | None                     # 最终总结（收敛时有值）
    remaining_disputes: list[str]           # 未解决分歧
    terminated_by_error: bool = False       # 是否因错误终止
```

### 4.4 数据流

```
YAML 配置
    ↓
ConfigLoader → Agent 实例 (×N) + Host 实例 (×1)
    ↓
ContextManager 注入 Round 0 共享 Context
    ↓
DiscussionEngine 循环:
    ├─ Step 1: 并行表达（所有 Agent）
    ├─ Step 2: 并行反驳（所有 Agent）
    ├─ ContextManager 压缩历史
    ├─ Archiver 持久化本轮
    └─ Host 判断收敛？→ 是: 生成总结 / 否: 下一轮
    ↓
Archiver 保存 summary.md
```

---

## 5. 组件规格

### 5.1 ConfigLoader (`config.py`)

**输入：** YAML 文件路径

**输出：** `DiscussionConfig` 数据类

**YAML 结构：**

```yaml
discussion:
  min_rounds: 2
  max_rounds: 5
  model: "claude-sonnet-4-20250514"  # 开发调试用，生产切 opus

agents:
  - name: "热点猎手"
    system_prompt: |
      你是一个敏锐的财经热点追踪者...
  - name: "深度研究员"
    system_prompt: |
      你是一个严谨的行业研究分析师...

host:
  convergence_prompt: |
    你是讨论的主持人...
  summary_prompt: |
    基于讨论全程，生成最终选题报告...

tools:
  - research_list
  - research_content
  - trending
  - published

context:
  research_dir: "~/research-data/"
  published_file: "PUBLISHED.md"
  research_days: 2  # 取最近几天的研报
```

**验收标准：**
- AC-1.1: 能解析上述 YAML 结构到 `DiscussionConfig` 类
- AC-1.2: 缺少必填字段时抛出明确错误信息
- AC-1.3: `min_rounds` 和 `max_rounds` 有默认值（2 和 5）

### 5.2 DiscussionEngine (`engine.py`)

**职责：** 编排讨论循环

**接口：**

```python
class DiscussionEngine:
    def __init__(self, config: DiscussionConfig): ...
    async def run(self) -> DiscussionResult: ...
```

**行为：**

1. 从 config 创建 N 个 Agno Agent + 1 个 Host Agent
2. 注入 Round 0 共享 Context
3. 循环 Round 1 ~ max_rounds:
   a. **Step 1 表达：** 并行调用所有 Agent，传入当前 Context + 历史发言，收集观点
   b. **Step 2 反驳：** 并行调用所有 Agent，每个 Agent 收到本轮**其他** Agent 的表达（排除自己的），收集质疑
   c. 将本轮表达和反驳追加到共享聊天记录
   d. 调用 ContextManager 压缩历史
   e. 持久化本轮数据
   f. 调用 Host 判断收敛（Host 收到压缩后的历史，最近 2 轮为完整发言，足以判断是否有实质性新挑战）
   g. 如果收敛且 round >= min_rounds：Host 生成总结，退出
   h. 如果达到 max_rounds：输出当前状态，退出
4. 返回 `DiscussionResult`

**并行执行：** 表达和反驳步骤中，所有 Agent 通过 `asyncio.gather` 并行调用。

**验收标准：**
- AC-2.1: 2 个 Agent + 1 Host 能完成完整的讨论循环（表达→反驳→Host判断→收敛退出）
- AC-2.2: 不到 min_rounds 时即使 Host 判断收敛也继续
- AC-2.3: 达到 max_rounds 时即使未收敛也退出，输出当前状态
- AC-2.4: 表达步骤中 N 个 Agent 并行执行（asyncio.gather）
- AC-2.5: 反驳步骤中 N 个 Agent 并行执行（asyncio.gather）
- AC-2.6: 每个 Agent 在表达时能看到完整的历史发言
- AC-2.7: 每个 Agent 在反驳时只看到本轮**其他** Agent 的表达（不含自己的）

### 5.3 ContextManager (`context.py`)

**职责：** 共享 Context 的注入和压缩

**Round 0 注入内容（元数据，不含原文）：**
- 已发布文章历史（topic、论点、日期）
- 近 N 天研报标题列表（日期-机构-标题）
- 当前热榜数据（实时拉取）

**压缩策略：**
- 保留最近 2 轮完整发言
- 更早轮次压缩为摘要（用 LLM 生成，每轮一段话）
- 当累积的 context 接近 window 上限时触发压缩（按 token 计数）

**接口：**

```python
class ContextManager:
    def __init__(self, config: DiscussionConfig): ...
    async def build_initial_context(self) -> str: ...
    async def compress(self, history: list[RoundRecord], current_round: int) -> list[RoundRecord]: ...
```

**验收标准：**
- AC-3.1: `build_initial_context` 返回包含已发布历史、研报标题、热榜数据的文本
- AC-3.2: 共享 Context 只包含元数据（标题级别），不包含研报原文
- AC-3.3: 压缩后保留最近 2 轮完整发言
- AC-3.4: 第 3 轮及以前的发言被压缩为摘要
- AC-3.5: 压缩摘要由 LLM 生成（通过 Agno Agent 调用）

### 5.4 Archiver (`persistence.py`)

**职责：** 讨论过程持久化到本地文件系统

**目录结构：**
```
discussions/
  └─ {YYYY-MM-DD_HHMM}/
      ├─ config.yaml
      ├─ context.md
      ├─ rounds/
      │   ├─ round_1_express.json
      │   ├─ round_1_challenge.json
      │   ├─ round_1_host.json
      │   └─ ...
      └─ summary.md
```

**接口：**

```python
class Archiver:
    def __init__(self, base_dir: str = "discussions"): ...
    def start_session(self, config: DiscussionConfig) -> str: ...  # 返回 session 目录路径
    def save_round(self, round_num: int, phase: str, data: dict) -> None: ...
    def save_context(self, context: str) -> None: ...
    def save_summary(self, summary: str) -> None: ...
```

**验收标准：**
- AC-4.1: `start_session` 创建 `{YYYY-MM-DD_HHMM}/` 目录和 `rounds/` 子目录
- AC-4.2: `start_session` 将当前配置复制为 `config.yaml`
- AC-4.3: 每轮的表达、反驳、Host 判断分别保存为独立 JSON 文件
- AC-4.4: 最终总结保存为 `summary.md`
- AC-4.5: 初始 Context 保存为 `context.md`

### 5.5 Tools（`tools/`）

讨论 Agent 在所有步骤（表达、反驳）均可调用 Tool，不做时机限制。Agent 自行判断是否需要查证据。Host Agent 无 Tool 访问权限。

Tool 按 Anthropic tool-description-standard 编写（纯文本描述，3-4+ 句，含 What/When/Parameters/Returns）。

#### 5.5.1 research_list

**功能：** 列出指定天数内的研报标题
**输入：** `days: int`（默认 2）, `research_dir: str`
**输出：** `日期-机构-标题` 格式的列表
**来源：** 扫描 `~/research-data/` 目录下的 PDF 文件名

**验收标准：**
- AC-5.1: 返回最近 N 天的研报标题列表
- AC-5.2: 格式为 `日期-机构-标题`
- AC-5.3: 只返回行业和宏观类研报（按文件路径/命名过滤）

#### 5.5.2 research_content

**功能：** 读取指定研报的文本内容
**输入：** `file_path: str`
**输出：** 研报全文（pdftotext 提取）
**验收标准：**
- AC-5.4: 通过 pdftotext 提取 PDF 文本内容
- AC-5.5: 文件不存在时返回明确错误信息

#### 5.5.3 trending

**功能：** 获取当前热榜数据
**输入：** 无
**输出：** 热门话题列表
**来源：** NewsNow MCP Server（`ourongxing/newsnow`）
**降级方案：** MCP 未部署时，返回提示信息"热榜服务暂不可用"

**验收标准：**
- AC-5.6: 通过 MCP 调用返回热榜数据
- AC-5.7: MCP 不可用时返回降级提示而非崩溃

#### 5.5.4 published

**功能：** 读取已发布文章历史
**输入：** `file_path: str`（默认 `PUBLISHED.md`）
**输出：** 已发布话题列表（topic、论点、日期）
**验收标准：**
- AC-5.8: 解析 PUBLISHED.md 并返回结构化的历史列表
- AC-5.9: 文件不存在时返回空列表

### 5.6 Host Agent

**职责：**
1. 收敛判断：读取本轮所有发言，判断是否收敛
2. 最终总结：收敛后生成选题报告

**约束：Host 无 Tools 访问权限。** 只读取讨论记录，不调用外部服务。

**收敛判断输出格式：**
```json
{
  "converged": true/false,
  "reason": "各 Agent 已就...",
  "remaining_disputes": ["..."]
}
```

**最终总结输出内容：**
- 推荐话题 + 为什么值得写
- 多角度文章骨架
- 每个角度的核心证据和来源
- 未解决的分歧（如有）

**验收标准：**
- AC-6.1: Host 返回结构化 JSON 收敛判断
- AC-6.2: Host 判断不明确时默认"未收敛"
- AC-6.3: 收敛后生成包含上述 4 项内容的总结

### 5.7 CLI 入口 (`main.py`)

**用法：**
```bash
python -m discuss_agent configs/topic_selection.yaml
```

**行为：**
1. 解析命令行参数（YAML 配置路径）
2. 加载配置
3. 运行讨论引擎
4. 输出结果路径

**验收标准：**
- AC-7.1: 接受 YAML 配置文件路径作为命令行参数
- AC-7.2: 讨论结束后打印归档目录路径
- AC-7.3: 配置文件不存在时报错退出

---

## 6. 错误处理

| 场景 | 处理 | AC |
|------|------|----|
| 单个 Agent LLM 调用失败 | 重试 1 次，仍失败则该 Agent 本轮跳过 | AC-8.1 |
| Tool 调用失败 | Agent 收到错误，继续发言 | AC-8.2 |
| 全部 Agent 失败 | 终止讨论，保存错误日志 | AC-8.3 |
| Agent 返回格式不符 | 重试 1 次 | AC-8.4 |
| Host 返回格式不符 | 重试 1 次，仍失败则默认"未收敛" | AC-8.5 |
| Host 判断不明确 | 默认"未收敛" | AC-8.6 |
| 讨论因错误终止 | 已归档的轮次保留，summary.md 不生成，DiscussionResult.terminated_by_error=True | AC-8.7 |

---

## 7. 第一个用例配置：选题讨论

**Agent 设定：**

| Agent | 人设定位 |
|-------|----------|
| 热点猎手 | 敏锐追踪大众关注的财经热点，关注传播性和话题度 |
| 深度研究员 | 严谨的行业分析师，关注研报的实质性洞察和数据 |

（具体 prompt 内容在实现时按 Anthropic prompt 工程标准编写，不在 spec 中硬编码）

**Host 设定：** 公众号主编，平衡热度与深度，最终给出选题建议。

---

## 8. 验收标准汇总

共 29 条验收标准（AC-1.1 ~ AC-8.7），覆盖：
- 配置加载 (3)
- 讨论引擎 (7)
- Context 管理 (5)
- 持久化 (5)
- Tools (9)
- Host (3)
- CLI (3)
- 错误处理 (7)

**端到端验收：** 使用 `configs/topic_selection.yaml` 配置，2 个 Agent + 1 Host 完成一次完整讨论，产出归档目录（含 config.yaml、context.md、rounds/*.json、summary.md）。
