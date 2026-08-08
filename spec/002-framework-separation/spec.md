# 002 - Framework / Business Separation

> 状态：待审核
> 前置：spec-001 已实现
> 范围：将 discuss-agent 拆分为通用框架（公开）+ 业务代码（私有），暴露干净的插件 API

---

## 1. 概述

discuss-agent 目前把通用讨论框架和研报选题业务代码混在同一个仓库。本 spec 将两者分离：

- **discuss-agent**（公开仓库）：只保留通用引擎、context 管理、host、持久化、配置加载。不包含任何业务相关的 tools、prompts 或 configs。
- **research-pipeline**（私有仓库）：包含所有研报选题相关代码（tools、YAML 配置、agent prompts），依赖 discuss-agent 作为框架。

框架通过一个简洁的注册 API 让外部代码插入自定义 tools 和 context 构建逻辑。

---

## 2. 系统边界

**In scope：**
- 从 discuss-agent 中移除所有业务代码
- 将 `ContextManager.build_initial_context` 从硬编码改为可插拔
- 将 `ContextConfig` 从业务字段改为通用 dict
- 将 tool 注册从硬编码 registry 改为运行时注册
- discuss-agent 提供完整 README（说明、安装、配置、示例）
- research-pipeline 仓库的结构定义和依赖方式
- 端到端验证：research-pipeline 导入 discuss-agent 运行完整讨论

**Out of scope：**
- research-pipeline 的 CI/CD 配置
- discuss-agent 的 PyPI 发布（用 git 依赖）
- 新功能开发（本 spec 是纯重构）
- 引擎内部逻辑变更（保持 spec-001 的行为不变）

---

## 3. 当前耦合点分析

| 文件 | 耦合问题 |
|------|----------|
| `tools/__init__.py` | 硬编码导入 4 个业务 tool 类，硬编码 `TOOL_REGISTRY` |
| `context.py` | 直接导入 `ResearchListTools`、`PublishedTools`、`TrendingTools`，`build_initial_context` 硬编码调用这三个 tool |
| `config.py` | `ContextConfig` 包含业务字段（`research_dir`、`published_file`、`research_days`） |
| `configs/topic_selection.yaml` | 业务配置 |
| `tools/research_list.py` | 业务 tool |
| `tools/research_content.py` | 业务 tool |
| `tools/trending.py` | 业务 tool |
| `tools/published.py` | 业务 tool |

不耦合（无需改动逻辑）：`models.py`、`persistence.py`、`main.py`、`__main__.py`。

注意：`engine.py` 的讨论循环逻辑不变，但其 `__init__` 中的 `from discuss_agent.tools import get_tools` 导入和 tool 实例化逻辑需要改为使用 plugin registry。

---

## 4. 设计

### 4.1 插件 API：Python entry_points

使用 Python 标准的 `entry_points`（`pyproject.toml` 中的 `[project.entry-points]`）作为唯一的注册机制。这是 Python 生态中插件发现的标准做法（pytest、setuptools、pip 都用此方式）。

**工作方式：**

1. 框架定义 entry_point group：`"discuss_agent.plugins"`
2. 外部包在自己的 `pyproject.toml` 里声明一个 entry_point，指向一个 `register` 函数
3. 框架启动时通过 `importlib.metadata.entry_points` 发现并调用所有已注册的 `register` 函数
4. `register` 函数接收一个 `PluginRegistry` 对象，向其中注册 tools 和 context builder

**框架侧接口（discuss-agent 提供）：**

```python
# discuss_agent/registry.py

class PluginRegistry:
    """Plugin registration interface for the discussion framework."""

    def register_tool(self, name: str, tool_class: type) -> None:
        """Register a tool class by name.

        The tool_class must be an Agno Toolkit subclass.
        When the YAML config's tools list references this name,
        the framework instantiates this class with context=config.context.
        """

    def register_context_builder(
        self, builder: Callable[[dict], Awaitable[str]]
    ) -> None:
        """Register a function that builds initial shared context.

        The builder receives the YAML config's 'context' dict
        and returns an assembled context string.
        Only one context builder is active; last registered wins.
        """

    def get_tool_class(self, name: str) -> type:
        """Look up a registered tool class. Raises ValueError if not found."""

    def get_context_builder(self) -> Callable[[dict], Awaitable[str]] | None:
        """Return the registered context builder, or None."""
```

**外部包侧（research-pipeline 实现）：**

```toml
# research-pipeline/pyproject.toml
[project.entry-points."discuss_agent.plugins"]
research = "research_pipeline:register"
```

```python
# research_pipeline/__init__.py
from discuss_agent.registry import PluginRegistry

def register(registry: PluginRegistry) -> None:
    from research_pipeline.tools import (
        ResearchListTools, ResearchContentTools,
        TrendingTools, PublishedTools,
    )
    registry.register_tool("research_list", ResearchListTools)
    registry.register_tool("research_content", ResearchContentTools)
    registry.register_tool("trending", TrendingTools)
    registry.register_tool("published", PublishedTools)

    from research_pipeline.context import build_research_context
    registry.register_context_builder(build_research_context)
```

**为什么选 entry_points 而不是其他方式：**

| 方案 | 优劣 |
|------|------|
| **entry_points（选定）** | Python 标准机制，`pip install` 即自动发现插件，无需额外配置。YAML 保持干净（tools 仍然只写名称字符串）。缺点：开发时需 `pip install -e .` |
| YAML 中写 Python dotted path | 用户需在 YAML 中写 `class: "package.module.ClassName"` 字符串，混入 Python 细节，违反配置/代码分离原则。YAML 变复杂 |
| 显式 Python 脚本调用 register() | 需要写启动脚本或修改 main.py，不适合作为库分发 |

### 4.2 YAML 配置格式

**YAML 格式保持不变**——`tools` 仍然是字符串名称列表，`context` 仍然是自由格式的 dict。框架不解析 `context` 的内部字段，直接透传给 context builder。

```yaml
# 格式与 spec-001 完全相同
tools:
  - research_list
  - research_content
  - trending
  - published

context:
  research_dir: "~/research-data/"
  published_file: "PUBLISHED.md"
  research_days: 2
```

**变化只在实现层**：名称解析从硬编码 `TOOL_REGISTRY` 改为 `PluginRegistry` 查找。用户感知不到格式变化。

### 4.3 Config 变更

**`ContextConfig` 改为通用 dict：**

```python
@dataclass
class DiscussionConfig:
    min_rounds: int
    max_rounds: int
    model: str
    agents: list[AgentConfig]
    host: HostConfig
    tools: list[str]       # 不变：仍然是名称列表
    context: dict          # 从 ContextConfig → dict，透传给 context builder
```

**`ConfigLoader` 变更：**
- 不再为 `context` 块做字段级验证（不再要求 `research_dir` 等）
- `context` 块作为 `dict` 原样保存到 `DiscussionConfig.context`
- `context` 键变为可选，缺省时默认为空 dict `{}`
- **约束：** `context` dict 的值必须是 YAML 可序列化的基础类型（str、int、float、bool、list、dict），因为 Archiver 会将配置序列化保存
- 其余验证逻辑（`agents`、`host`、`tools`、`discussion`）不变
- 从 `_REQUIRED_TOP_KEYS` 中移除 `"context"`

### 4.4 Tool 实例化

当前 `get_tools()` 中对 `research_list` 和 `research_content` 有特殊的参数传递逻辑（手动传 `research_dir`、`allowed_dir`）。这种逻辑属于业务侧。

**新方案：** 框架统一用 `tool_class(context=config.context)` 实例化所有 tool。**所有 tool 类都必须接受 `context: dict | None = None` 作为构造参数**，自行从中取所需字段。不需要 context 的 tool 忽略该参数即可。

```python
# 框架侧（engine.py 中）
def _create_tools(self) -> list:
    registry = load_plugins()  # 从 entry_points 加载
    tools = []
    for name in self._config.tools:
        cls = registry.get_tool_class(name)
        tools.append(cls(context=self._config.context))
    return tools
```

```python
# 业务侧 tool 示例：需要 context 参数的 tool
class ResearchListTools(Toolkit):
    def __init__(self, context: dict | None = None):
        super().__init__(name="research_list")
        ctx = context or {}
        self._research_dir = ctx.get("research_dir", "~/research-data/")

# 业务侧 tool 示例：不需要 context 参数的 tool
class PublishedTools(Toolkit):
    def __init__(self, context: dict | None = None):
        super().__init__(name="published")
        # 不使用 context，但必须接受该参数
```

**迁移要求：** 现有 4 个 tool 类的构造函数签名都必须改为 `__init__(self, context: dict | None = None)`。当前各 tool 的签名各不相同（`research_dir: str`、`allowed_dir: str`、无参数、`mcp_url: str | None`），迁移时统一改为从 `context` dict 中获取所需配置。

### 4.5 ContextManager 变更

当前 `ContextManager.__init__` 直接实例化 3 个业务 tool。`build_initial_context` 硬编码调用它们。

**变更：** `ContextManager` 不再自行构建 context。它从 plugin registry 获取已注册的 context builder 函数，委托给它。

```python
class ContextManager:
    def __init__(self, config: DiscussionConfig, context_builder: Callable | None):
        self._config = config
        self._context_builder = context_builder

    async def build_initial_context(self) -> str:
        if self._context_builder:
            return await self._context_builder(self._config.context)
        logger.warning("No context builder registered. Starting discussion with empty context.")
        return ""  # 无 context builder 时返回空字符串
```

压缩逻辑（`compress`、`_compress_round`）不受影响，保持原样。

### 4.6 Engine 变更

`DiscussionEngine.__init__` 的变更：
- 调用 `load_plugins()` **一次**，获取单个 `PluginRegistry` 实例
- 从该 registry 获取 tool 类并实例化（替代硬编码的 `get_tools`）
- 从同一 registry 获取 context builder，传给 `ContextManager` 构造函数
- **不得**在 ContextManager 或其他地方再次调用 `load_plugins()`

讨论循环逻辑（`run`、`_express`、`_challenge`、`_host_judge`、`_host_summarize`）零改动。

### 4.7 插件加载时机

框架在 `DiscussionEngine.__init__` 时调用 `load_plugins()` 一次，发现并执行所有 entry_points。此后 registry 是只读的。

```python
# discuss_agent/registry.py
def load_plugins() -> PluginRegistry:
    registry = PluginRegistry()
    eps = importlib.metadata.entry_points(group="discuss_agent.plugins")
    if not eps:
        logger.warning(
            "No discuss_agent.plugins entry points found. "
            "Did you install your plugin package with 'pip install -e .'?"
        )
    for ep in eps:
        register_fn = ep.load()  # 插件加载失败时 let exception propagate（fail fast）
        register_fn(registry)    # register 函数异常也 fail fast，不静默跳过
    return registry
```

**错误处理：** `load_plugins()` 采用 fail-fast 策略。如果任何插件的 `ep.load()` 或 `register_fn(registry)` 抛出异常，直接向上传播，终止启动。理由：插件注册失败意味着后续讨论必然缺少 tools 或 context builder，静默跳过只会导致更隐蔽的错误。

### 4.8 discuss-agent 包结构（重构后）

```
discuss_agent/
  ├── __init__.py          # 导出公共 API
  ├── __main__.py          # python -m discuss_agent
  ├── main.py              # CLI 入口
  ├── models.py            # 数据结构（不变）
  ├── config.py            # ConfigLoader（ContextConfig → dict）
  ├── engine.py            # DiscussionEngine（从 registry 获取 tools/builder）
  ├── context.py           # ContextManager（委托给 context builder）
  ├── persistence.py       # Archiver（不变）
  └── registry.py          # PluginRegistry + load_plugins()（新增）
```

**删除：**
- `tools/` 整个目录（4 个业务 tool + registry __init__）
- `configs/topic_selection.yaml`

**新增：**
- `registry.py`

### 4.9 research-pipeline 仓库结构

```
research-pipeline/
  ├── pyproject.toml             # 声明 entry_point + 依赖 discuss-agent
  ├── research_pipeline/
  │   ├── __init__.py            # register() 函数
  │   ├── context.py             # build_research_context()
  │   └── tools/
  │       ├── __init__.py
  │       ├── research_list.py   # 从 discuss-agent 迁移
  │       ├── research_content.py
  │       ├── trending.py
  │       └── published.py
  ├── configs/
  │   └── topic_selection.yaml   # 从 discuss-agent 迁移
  └── tests/
```

**依赖声明：**

```toml
[project]
name = "research-pipeline"
dependencies = [
    "discuss-agent @ git+https://github.com/Zhijiang-Li1111/discuss-agent.git",
]
```

注意：`agno`、`pyyaml`、`anthropic` 是 discuss-agent 的依赖，会传递安装，research-pipeline 不需要重复声明。如果 research-pipeline 的 tools 直接导入 agno（用 `Toolkit` 基类），则需要声明 `agno` 依赖。

### 4.10 discuss-agent 公共 API 导出

`discuss_agent/__init__.py` 导出框架用户需要的类型：

```python
from discuss_agent.engine import DiscussionEngine
from discuss_agent.models import AgentUtterance, RoundRecord, DiscussionResult
from discuss_agent.config import ConfigLoader, DiscussionConfig, AgentConfig, HostConfig
from discuss_agent.registry import PluginRegistry, load_plugins
```

注意：`ContextManager`、`Archiver` 是框架内部实现，不导出。外部消费者只需要 `DiscussionEngine`（运行讨论）、`PluginRegistry`（注册插件）、`ConfigLoader`（加载配置）和数据类型。

### 4.11 discuss-agent README

README.md 包含：

1. **项目介绍**：一句话说明 + 核心特性列表
2. **安装**：`pip install git+https://...` 方式
3. **快速开始**：最小可运行示例（YAML 配置 + 运行命令）
4. **配置说明**：YAML 结构文档（discussion、agents、host、tools、context）
5. **插件开发**：如何写 Agno Toolkit 子类 + context builder + entry_point 注册
6. **API 参考**：`PluginRegistry`、`DiscussionEngine`、`DiscussionResult` 的接口说明

注意：重构后 `pdftotext`（poppler-utils）不再是 discuss-agent 的系统依赖，README 不应提及。该依赖属于 research-pipeline。

---

## 5. 验收标准

### 框架清洁度（discuss-agent）

- **AC-1.1**: discuss-agent 包内不包含 `tools/` 目录或业务 tool 代码
- **AC-1.2**: discuss-agent 包内不包含 `configs/` 目录或业务配置文件
- **AC-1.3**: discuss-agent 代码中不存在对 `research_list`、`research_content`、`trending`、`published`、`research-data`、`PUBLISHED.md`、`pdftotext` 的导入或硬编码引用
- **AC-1.4**: `context.py` 不直接导入任何 tool 类，`build_initial_context` 委托给已注册的 context builder

### 插件 API

- **AC-2.1**: `registry.py` 提供 `PluginRegistry` 类，支持 `register_tool(name, cls)` 和 `register_context_builder(builder)` 方法
- **AC-2.2**: `load_plugins()` 通过 `importlib.metadata.entry_points(group="discuss_agent.plugins")` 发现并加载所有已注册插件
- **AC-2.3**: YAML 中 `tools:` 列表引用的 tool name 必须在 plugin registry 中已注册，否则抛出含名称的 `ValueError`
- **AC-2.4**: Tool 类通过 `cls(context=config.context)` 统一实例化，框架不负责特定 tool 的参数传递
- **AC-2.5**: 无插件注册时（无 entry_points、tools 列表也为空），引擎仍能正常运行
- **AC-2.6**: 无 context builder 注册时，`build_initial_context()` 返回空字符串并输出 warning 日志
- **AC-2.7**: 无 entry_points 发现时，`load_plugins()` 输出 warning 日志提示用户检查是否安装了插件包
- **AC-2.8**: 插件的 `register` 函数抛出异常时，`load_plugins()` 不捕获异常，直接 fail fast 终止启动

### 配置

- **AC-3.1**: `DiscussionConfig.context` 类型为 `dict`，框架不解析其内部字段
- **AC-3.2**: YAML 的 `context:` 块原样透传给 context builder，不做字段级验证
- **AC-3.3**: `context` 键为可选，缺省时默认为空 dict `{}`
- **AC-3.4**: `tools` 字段仍为 `list[str]`（名称列表），YAML 格式不变
- **AC-3.5**: 其余配置验证（`agents`、`host`、`discussion`）逻辑和错误信息不变

### research-pipeline 侧

- **AC-4.1**: research-pipeline 通过 `pyproject.toml` 的 `[project.entry-points."discuss_agent.plugins"]` 声明插件
- **AC-4.2**: research-pipeline 的 `register()` 函数注册 4 个 tools 和 1 个 context builder
- **AC-4.3**: research-pipeline 依赖 `discuss-agent @ git+https://github.com/Zhijiang-Li1111/discuss-agent.git`
- **AC-4.4**: 4 个 tool 的功能和行为与重构前完全一致

### 公共 API

- **AC-5.1**: `discuss_agent/__init__.py` 导出 `DiscussionEngine`、`DiscussionResult`、`ConfigLoader`、`DiscussionConfig`、`PluginRegistry`
- **AC-5.2**: 导出的类型在 research-pipeline 中可正常导入和使用

### README

- **AC-6.1**: README 包含项目介绍、安装方式、快速开始示例、配置说明、插件开发指南、API 参考 6 个部分
- **AC-6.2**: 快速开始示例包含最小可运行的 YAML 配置和运行命令

### 端到端验证

- **AC-7.1**: 在 research-pipeline 目录下执行 `pip install -e .`（同时安装 discuss-agent 依赖）后，从 research-pipeline 目录运行 `python -m discuss_agent configs/topic_selection.yaml` 能启动完整讨论
- **AC-7.2**: 讨论流程（表达 → 反驳 → Host 判断 → 收敛/退出）与重构前行为一致
- **AC-7.3**: 讨论归档输出目录结构与重构前一致（`config.yaml`、`context.md`、`rounds/*.json`、`summary.md`）

### 测试

- **AC-8.1**: discuss-agent 的现有 engine、config、context、persistence 测试更新后全部通过
- **AC-8.2**: discuss-agent 的 tool 相关测试移至 research-pipeline 或删除
- **AC-8.3**: research-pipeline 有独立的测试验证 4 个 tools 功能正确

---

## 6. 迁移策略

两个仓库同步变更，不需要兼容过渡期：

1. 先在 discuss-agent 中完成框架改造（添加 registry.py、修改 config/context/engine、删除 tools/）
2. 创建 research-pipeline 仓库，将业务代码迁入
3. 在 research-pipeline 中 `pip install -e .` 验证端到端

不需要版本号兼容或 deprecation path——当前只有一个消费者（research-pipeline），且两个仓库同一个人维护。

---

## 7. 不变条件

以下 spec-001 的行为在本次重构中不变：
- 讨论循环逻辑（表达 → 反驳 → Host 判断 → 收敛/继续）
- 并行执行（`asyncio.gather`）
- 错误处理策略（重试、跳过、终止）
- 持久化归档目录结构
- Context 压缩策略（保留最近 2 轮，120K token 阈值）
- CLI 用法（`python -m discuss_agent <yaml>`）
- Host 无 Tools 约束

---

## 8. 验收标准汇总

共 27 条验收标准（AC-1.1 ~ AC-8.3），覆盖：
- 框架清洁度 (4)
- 插件 API (8)
- 配置 (5)
- research-pipeline (4)
- 公共 API (2)
- README (2)
- 端到端验证 (3)
- 测试 (3)
