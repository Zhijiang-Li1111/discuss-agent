# 多 Agent 对抗式讨论框架 — 设计文档

> 维护者：研报豆 📑
> 版本：v0.3
> 日期：2026-04-08
> 状态：设计定稿

---

## 一、目标

构建一个**通用的多 Agent 对抗式讨论框架**，通过多视角并行讨论、交叉反驳、收敛判断，产出高质量的结论。

第一个用例：从每日同步的研报中筛选公众号选题，最终发布在独立的财经类公众号上，对标 YouTube「老厉害」。

---

## 二、核心理念

- **框架通用，配置可插拔** — 同一套代码，通过配置文件切换不同讨论场景
- **对抗出真知** — 多视角碰撞比单一 Agent 思考更可靠
- **事实驱动** — 所有讨论基于可引用的数据和原文，不基于情感
- **圆桌自己写，框架管底层** — Agno 提供 Agent 原子能力，讨论编排完全自研
- **大众热点 × 专业研报 = 好选题**（选题用例的核心逻辑）

---

## 三、技术选型

### 底层框架：Agno（agno-agi/agno）

**只用 Agno 的原子能力，不用它的 Team/Workflow：**

| 用 Agno 的 | 不用 Agno 的 |
|-----------|-------------|
| Agent 类（prompt + model + tools） | TeamMode.broadcast |
| Tool calling 封装 | Workflow 编排 |
| Session Summary（context 压缩） | AgentOS |
| SqliteDb 持久化 | |
| MCP 集成 | |
| Anthropic Claude 原生支持 | |

选择理由：
- Context 压缩内置（`enable_session_summaries=True`），解决多轮讨论 token 爆炸
- 持久化一行代码（`db=SqliteDb(...)`）
- 120+ 预置 Toolkit
- MCP Server 原生支持
- 39K Stars，活跃维护

### 模型：Claude Opus
- 讨论 Agent：Claude Opus（不考虑成本，用最强模型）
- Host Agent：Claude Opus（收敛判断需要强理解力）

### Prompt 标准：Anthropic 三剑客
- 参考：`references/anthropic/agent-prompt-guidelines.md`
- 不硬编码限制条件，给模型足够判断空间
- Agent 通过人设 prompt 引导立场，不通过规则限死

---

## 四、框架架构

### 4.1 分层设计

```
┌─────────────────────────────────────────────┐
│           圆桌讨论引擎（自研）                 │
│                                             │
│  ├─ 讨论循环编排                              │
│  ├─ 并行表达 / 交叉反驳                       │
│  ├─ Host 收敛判断                            │
│  ├─ 讨论记录持久化                            │
│  └─ 配置加载                                 │
├─────────────────────────────────────────────┤
│           Agno 原子能力层                     │
│                                             │
│  ├─ Agent 类（prompt + model + tools）       │
│  ├─ Tool calling                            │
│  ├─ Session Summary（context 压缩）          │
│  ├─ SqliteDb（持久化）                       │
│  └─ MCP 集成                                │
├─────────────────────────────────────────────┤
│           外部服务                            │
│                                             │
│  ├─ Anthropic Claude API                    │
│  ├─ NewsNow MCP Server（热榜）              │
│  ├─ IMA MCP Server（知识库）                 │
│  └─ 本地研报文件（pdftotext）                 │
└─────────────────────────────────────────────┘
```

### 4.2 框架与配置分离

**框架固定（自研代码）：**
- 并行表达 → 交叉反驳的循环
- 共享聊天记录管理
- Host 编排与收敛判断
- 跨 Agent context 压缩
- 讨论过程持久化存档

**配置可变（YAML）：**
- Agent 数量和人设 prompt
- Host 的收敛判断 prompt 和总结 prompt
- Tools 列表
- 初始 Context 的来源和格式
- 最大轮数 / 最小轮数
- 输出格式模板
- 模型选择

---

## 五、讨论系统设计

### 5.1 参与者

**N+1 个 LLM 实例，全部 Claude Opus：**

```
讨论 Agent（×N）
  ├─ 同一份代码
  ├─ 同一套 Tools
  ├─ 同一份共享 Context
  └─ 唯一区别：人设 system prompt

Host Agent（×1）
  ├─ 无 Tools
  ├─ 只读聊天记录
  ├─ 判断收敛
  └─ 生成最终总结
```

### 5.2 讨论流程

```
Round 0: 注入共享 Context

Round 1+: 每轮两步

  Step 1: 表达（并行）
    所有 Agent 基于 Context + 历史发言 → 输出观点 + 证据
    （可调 Tool）

  Step 2: 反驳（并行）
    每个 Agent 看到其他人本轮的表达 → 质疑
    （可调 Tool）

  Host 判断收敛？
    → 是：生成最终总结，退出
    → 否：进入下一 Round
    → 达到最大轮数：输出当前状态，交老板裁决
```

### 5.3 Tools

所有步骤（表达、反驳）均可调用 Tool，不做时机限制。Agent 自行判断是否需要查证据。

### 5.4 收敛定义

**各 Agent 互相认可对方的观点，不再提出实质性的新挑战。**

Host 判断标准：
- 反驳轮中是否有实质性的新质疑？（不是重复翻炒已有论点）
- 同一话题下不同角度可以共融，不要求排序一致
- Host 返回结构化输出：`{converged: bool, reason: string, remaining_disputes: [...]}`

### 5.5 讨论规则（通过 prompt 引导，不硬编码）

按 Anthropic prompt 工程标准，在人设 prompt 中引导：
- 论点需要有证据支撑
- 坚持自己的立场和视角
- 被逻辑、数据、事实说服时可以调整观点
- 看到其他人的观点时，先审视其论据是否充分

### 5.6 Host 最终总结

Host 的输出 = 下一步调研和写作的输入：

- 推荐话题 + 为什么值得写
- 多角度文章骨架（整合各 Agent 的不同角度）
- 每个角度的核心证据和来源
- 未解决的分歧（如有）

---

## 六、Context 管理

### 6.1 共享 Context（Round 0 注入）

**只放元数据，不放原文：**

| 内容 | 格式 |
|------|------|
| 已发布文章历史 | topic、论点、日期、可续写方向 |
| 近 2 天研报标题列表 | `日期-机构-标题`，仅行业和宏观 |
| 当前热榜数据 | 实时拉取，不用缓存 |

### 6.2 聊天记录

- 指讨论 Agent 之间的历史发言，不是其他来源
- 每轮表达和反驳的内容，累积到共享记录中
- 所有 Agent 在表达/反驳时看到完整的历史发言

### 6.3 Context 压缩

使用 Agno 的 Session Summary 机制：
- 单 Agent 维度：Agno 内置，`enable_session_summaries=True`
- 跨 Agent 共享讨论记录的压缩：自研（~150 行），定期用 LLM 将早期轮次摘要化

压缩策略：
- 保留最近 2 轮的完整发言
- 更早的轮次压缩为摘要（每轮一段话）

### 6.4 Tool 返回结果

Tool 返回什么就是什么，不人为截断。Opus 200K 窗口 + Session Summary 压缩机制兜底。早期轮次的 Tool 返回结果会在后续压缩时被摘要化。

---

## 七、持久化

### 7.1 Agno Session 持久化

```python
db = SqliteDb(table_name="discussions", db_file="discussions.db")
```

### 7.2 讨论记录归档

每次讨论全程归档到本地文件系统：

```
discussions/
  └─ {YYYY-MM-DD_HHMM}/
      ├─ config.yaml              # 本次使用的配置
      ├─ context.md               # 初始注入的共享 Context
      ├─ rounds/
      │   ├─ round_1_express.json  # 第1轮表达（所有 Agent）
      │   ├─ round_1_challenge.json# 第1轮反驳（所有 Agent）
      │   ├─ round_1_host.json     # Host 判断
      │   ├─ round_2_express.json
      │   └─ ...
      └─ summary.md               # 最终总结（如果收敛）
```

---

## 八、错误处理

| 场景 | 处理方式 |
|------|----------|
| 单个 Agent LLM 调用失败 | 重试 1 次，仍失败则该 Agent 本轮跳过 |
| Tool 调用失败 | Agent 收到错误信息，继续发言（没有证据就说"未能验证"） |
| 全部 Agent 失败 | 终止讨论，输出错误日志 |
| Agent 返回格式不符 | 尝试解析，失败则重试 1 次 |
| Host 判断不明确 | 默认"未收敛"，继续下一轮 |

---

## 九、用例配置

通过 YAML 配置文件定义具体用例的 Agent prompt、Tools、Context 来源。
参见 `examples/` 目录。

---

## 十、数据依赖

| 数据源 | 状态 | 获取方式 |
|--------|------|----------|
| IMA 研报同步 | ✅ 已完成 | cron 每天 18:00，`~/ima-downloads/` |
| 研报去水印 | ✅ 已完成 | 同步后自动执行 |
| NewsNow 热榜 | 待部署 | MCP Server，`ourongxing/newsnow` |
| IMA MCP | 待接入 | `highkay/tencent-ima-copilot-mcp` |
| 已发布历史 | ✅ 已建立 | `PUBLISHED.md` |

---

## 十一、自研代码估算

| 模块 | 代码量 | 说明 |
|------|--------|------|
| 讨论循环引擎 | ~200 行 | 并行表达→反驳→Host判断的循环 |
| 跨 Agent context 管理 | ~150 行 | 共享聊天记录的压缩和注入 |
| 配置加载 | ~50 行 | YAML → Agno Agent 实例化 |
| 持久化和归档 | ~100 行 | JSON 归档 + 目录管理 |
| Tools 定义 | ~100 行 | 4 个 Tool 的实现 |
| **合计** | **~600 行** | |

---

## 十二、设计原则

1. **圆桌自研，底层用 Agno** — 讨论逻辑完全自己写，Agno 只管 Agent 原子能力
2. **框架是代码，场景是配置** — 可复用到任何多视角讨论场景
3. **Context 精简，Tool 按需** — 共享层只放标题，原文通过 Tool 获取
4. **并行执行** — 表达和反驳都可并行，提高效率
5. **Prompt 遵循 Anthropic 标准** — 给模型空间，不硬编码规则
6. **全程归档** — 每次讨论持久化，可回溯、可审计

---

## 十三、后续模块（本文档不覆盖）

- 深度调研系统（选题确定后，读多份研报，生成调研报告）
- 财经写作豆（新 Agent，独立风格，写公众号文章）
- 发布流程（审核 → 排版 → 发公众号）

---

## 十四、开放问题（实现时再定）

- 最大/最小轮数具体值（建议 min=2, max=5，跑了再调）
- 跨 Agent context 压缩的触发时机（每轮？还是超过 token 阈值？）
- 讨论结果的质量反馈回路（老板采纳/拒绝如何回流）
- Agent 同质化风险（同模型可能太 agreeable，需观察实际效果）
- 研报标题预处理（外资翻译标题太长太乱，是否需要标准化）
