# 003 - Runtime Configuration Flexibility

> 状态：待审核
> 前置：spec-002 已实现
> 范围：用 YAML dotted-path 替代 entry_points 进行 tool 注册 + 支持 per-agent tool 配置

---

## 1. 概述

spec-002 引入了 Python entry_points 作为 tool 注册机制。实践中发现两个问题：

1. **entry_points 需要 `pip install -e .`**：每次改 tool 都得重装，开发体验差。改为在 YAML 中直接写 Python dotted path，框架用 `importlib.import_module` 运行时加载，无需任何安装步骤。
2. **所有 agent 共享同一套 tools**：无法为不同 agent 配置不同的 tool 集合。增加 per-agent tool 配置：agent 继承全局 tools，可额外添加或禁用特定 tool。

两个变更都在 discuss-agent 框架侧。research-pipeline 只需更新 YAML 配置和 pyproject.toml（移除 entry_points 声明）。

---

## 2. 系统边界

**In scope：**
- YAML `tools` 格式从名称列表改为 dotted-path 对象列表
- YAML `agents` 增加 `extra_tools` 和 `disable_tools` 可选字段
- `registry.py` 改为基于 `importlib.import_module` 的运行时加载（移除 entry_points 依赖）
- `config.py` 更新数据类和解析逻辑
- `engine.py` 更新 tool 实例化为 per-agent
- 顶层 `context_builder` 字段：YAML 中用 dotted path 指定 context builder（替代 entry_points 注册）
- research-pipeline 的 pyproject.toml 和 YAML 配置更新
- discuss-agent README 更新
- 所有测试更新

**Out of scope：**
- 讨论引擎循环逻辑变更
- 新 tool 开发
- context builder 被 per-agent 化（context builder 保持全局唯一）

---

## 3. 当前状态（spec-002 之后）

| 组件 | 当前行为 |
|------|----------|
| `tools` YAML 格式 | `list[str]` 名称列表，如 `["research_list", "trending"]` |
| tool 发现 | 通过 `entry_points(group="discuss_agent.plugins")` 发现 |
| tool 实例化 | `registry.get_tool_class(name)(context=config.context)` |
| agent tool 分配 | 所有 agent 共享同一个 `tools` 列表 |
| context builder | 通过 entry_points 注册的 `register()` 函数注册 |
| `PluginRegistry` | 存储 name→class 映射和 context builder |

---

## 4. 设计

### 4.1 YAML 格式变更

**当前格式（spec-002）：**
```yaml
tools:
  - research_list
  - trending

context:
  research_dir: "~/ima-downloads/"

agents:
  - name: "热点猎手"
    system_prompt: ...
```

**新格式：**
```yaml
discussion:
  min_rounds: 2
  max_rounds: 5
  model: "claude-sonnet-4-20250514"

tools:
  - path: research_pipeline.tools.research_list.ResearchListTools
  - path: research_pipeline.tools.research_content.ResearchContentTools
  - path: research_pipeline.tools.trending.TrendingTools
  - path: research_pipeline.tools.published.PublishedTools

context_builder: research_pipeline.context.build_research_context

context:
  research_dir: "~/ima-downloads/"
  published_file: "PUBLISHED.md"
  research_days: 2

agents:
  - name: "热点猎手"
    system_prompt: ...
    extra_tools:
      - path: some_package.tools.WebSearchTools
    disable_tools:
      - research_pipeline.tools.trending.TrendingTools

  - name: "深度研究员"
    system_prompt: ...
    # 继承全局 tools，无增减
```

**变更要点：**
- `tools` 从 `list[str]` 改为 `list[dict]`，每个 dict 有 `path` 字段（Python dotted path 指向 Toolkit 子类）
- 新增顶层 `context_builder` 字段（可选）：dotted path 指向 context builder 函数。独立于 `context` 块，避免保留字冲突。
- `context` 块保持纯业务数据 dict，不包含任何框架字段。
- `agents[].extra_tools`：可选，`list[dict]`，格式同全局 `tools`。这些 tool 追加到该 agent 的 tool 集合。
- `agents[].disable_tools`：可选，`list[str]`，每个元素是要禁用的 tool 的 dotted path。从该 agent 的 tool 集合中移除匹配的 tool。`disable_tools` 用裸字符串而非 `{path: ...}` 格式，因为它只需标识要移除的 path，不需要实例化对象。

### 4.2 Tool 解析与实例化

**解析流程：**
1. `ConfigLoader` 解析 `tools` 列表为 `list[ToolConfig]`，每个 `ToolConfig` 有 `path: str`
2. `ConfigLoader` 验证每个 tool entry 必须是 dict 且包含 `path` 键，否则抛出 `ValueError`
3. `tools` 仍为 _REQUIRED_TOP_KEYS，缺省时报错。空列表 `tools: []` 合法。
4. `DiscussionEngine.__init__` 用 `importlib.import_module` 动态导入每个 path 对应的类
5. 每个 agent 的 tool 集合 = 全局 tools + extra_tools − disable_tools
6. 如果同一个 dotted path 出现在全局 tools 和 extra_tools 中，最终列表中该 tool 保留一份（去重，按 path 字符串）
7. 用 `cls(context=config.context)` 实例化

**动态导入逻辑：**
```python
def import_from_path(dotted_path: str):
    """Import a class or function from a dotted path like 'package.module.Name'."""
    module_path, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
```

导入失败时抛出明确错误，包含 dotted path 和原始异常信息。
`dotted_path` 中必须包含至少一个 `.`，否则在 `rsplit` 前验证并抛出 `ValueError`。

### 4.3 Per-agent Tool 计算

对每个 agent：
1. 从全局 `tools` 列表获取所有 tool 类
2. 追加 `extra_tools` 中的 tool 类
3. 移除 `disable_tools` 中列出的 dotted path 对应的 tool 类
4. 用 `cls(context=config.context)` 实例化最终的 tool 列表
5. 传给 `Agent(tools=per_agent_tools)`

`disable_tools` 中的 path 必须精确匹配全局或 extra_tools 中某个 tool 的 path。Warning 判断时机：在合并全局 + extra_tools 后、执行过滤前，遍历 `disable_tools` 列表，对不在合并列表中的 path 发出 warning。

### 4.4 Context Builder 加载

当前 context builder 通过 entry_points 注册。改为 YAML 顶层 `context_builder` 字段指定 dotted path：

```yaml
context_builder: research_pipeline.context.build_research_context

context:
  research_dir: "~/ima-downloads/"
```

- `context_builder` 是顶层可选字段，独立于 `context` 块。这样 `context` 块永远是纯业务数据，不存在保留字冲突风险。
- 如果未指定，`build_initial_context()` 返回空字符串 + warning 日志（与当前行为一致）
- 框架用 `import_from_path(builder_path)` 加载 builder 函数
- `context` dict 原样传给 builder 函数（不需要做任何字段过滤）

### 4.5 Registry 变更

`PluginRegistry` 类和 `load_plugins()` 函数不再需要。用一个简单的 `import_from_path()` 公共函数替代。

`registry.py` 重写为：只保留 `import_from_path()` 函数。删除 `PluginRegistry` 类和 `load_plugins()` 函数。

### 4.6 Config 数据类变更

```python
@dataclass
class ToolConfig:
    path: str  # Python dotted path to Toolkit subclass

@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    extra_tools: list[ToolConfig]      # 默认空列表
    disable_tools: list[str]           # 默认空列表，存储 dotted path 字符串

@dataclass
class DiscussionConfig:
    min_rounds: int
    max_rounds: int
    model: str
    agents: list[AgentConfig]
    host: HostConfig
    tools: list[ToolConfig]            # 全局 tool 列表
    context: dict                      # 纯业务数据，透传给 context builder
    context_builder: str | None        # dotted path，可选
```

### 4.7 Engine 变更

`DiscussionEngine.__init__` 变更：
- 不再调用 `load_plugins()`
- 动态导入全局 tools 和每个 agent 的 extra_tools
- 为每个 agent 计算独立的 tool 列表
- 从 `config.context_builder` 动态导入 context builder

```python
# 伪代码
global_tool_classes = [(tc.path, import_from_path(tc.path)) for tc in config.tools]

for ac in config.agents:
    # Start with global tools
    agent_tools = list(global_tool_classes)
    # Add extra tools
    for tc in ac.extra_tools:
        agent_tools.append((tc.path, import_from_path(tc.path)))
    # Remove disabled tools
    disabled = set(ac.disable_tools)
    agent_tools = [(p, cls) for p, cls in agent_tools if p not in disabled]
    # Instantiate
    tool_instances = [cls(context=config.context) for _, cls in agent_tools]
    # Create agent
    Agent(name=ac.name, ..., tools=tool_instances)
```

### 4.8 __init__.py 公共 API 变更

- 移除导出：`PluginRegistry`、`load_plugins`
- 新增导出：`ToolConfig`、`import_from_path`

### 4.9 research-pipeline 变更（独立仓库 `/home/zhijiang/.openclaw/repos/research-pipeline/`）

- `pyproject.toml`：移除 `[project.entry-points."discuss_agent.plugins"]` 部分
- `research_pipeline/__init__.py`：删除 `register()` 函数（不再需要）
- `configs/topic_selection.yaml`：更新为新的 YAML 格式
- `tests/test_registration.py`：删除（entry_points 注册测试不再需要）

### 4.10 README 变更

- 更新 Configuration 部分：新的 `tools` 格式和顶层 `context_builder` 字段
- 更新 Plugin Development 部分：不再需要 entry_points，只需确保 Python path 可导入
- 更新 API Reference：移除 `PluginRegistry`，新增 `ToolConfig`
- 更新 Quick Start 示例

---

## 5. 验收标准

### Tool 加载（dotted path）

- **AC-1.1**: YAML `tools` 接受 `list[dict]` 格式，每个 dict 有 `path` 字段
- **AC-1.2**: 框架通过 `importlib.import_module` 根据 `path` 动态导入 Tool 类
- **AC-1.3**: 导入失败时抛出明确错误，包含 dotted path 和原始异常
- **AC-1.4**: `tools: []` 时引擎正常运行（agent 无 tool）
- **AC-1.5**: entry_points 不再用于 tool 注册（`load_plugins` 函数删除）

### Per-agent tools

- **AC-2.1**: agent 默认继承所有全局 tools
- **AC-2.2**: `extra_tools` 中的 tool 追加到该 agent 的 tool 集合
- **AC-2.3**: `disable_tools` 中的 dotted path 从该 agent 的 tool 集合中移除
- **AC-2.4**: `extra_tools` 和 `disable_tools` 都是可选字段，缺省时为空列表
- **AC-2.5**: `disable_tools` 中的 path 不匹配任何已有 tool 时，输出 warning 日志不报错
- **AC-2.6**: 不同 agent 可以有不同的 tool 集合

### Context builder

- **AC-3.1**: 顶层 `context_builder` 字段指定 context builder 的 dotted path
- **AC-3.2**: `context_builder` 字段可选，未指定时返回空字符串 + warning
- **AC-3.3**: `context` dict 原样传给 builder 函数（不做字段过滤）

### Config

- **AC-4.1**: `DiscussionConfig.tools` 类型为 `list[ToolConfig]`
- **AC-4.2**: `AgentConfig` 新增 `extra_tools: list[ToolConfig]` 和 `disable_tools: list[str]`
- **AC-4.3**: `ToolConfig` 有 `path: str` 字段

### research-pipeline

- **AC-5.1**: pyproject.toml 不再包含 `[project.entry-points]` 部分
- **AC-5.2**: `research_pipeline/__init__.py` 不再包含 `register()` 函数
- **AC-5.3**: `configs/topic_selection.yaml` 使用新格式（dotted path tools、顶层 `context_builder`）
- **AC-5.4**: research-pipeline 功能不变，tools 行为与之前一致

### 公共 API

- **AC-6.1**: `discuss_agent/__init__.py` 不再导出 `PluginRegistry`、`load_plugins`
- **AC-6.2**: `discuss_agent/__init__.py` 导出 `ToolConfig`、`import_from_path`

### 测试

- **AC-7.1**: discuss-agent 所有测试通过
- **AC-7.2**: research-pipeline 所有测试通过
- **AC-7.3**: 有测试验证 per-agent tool 计算（extra_tools 追加、disable_tools 移除）
- **AC-7.4**: 有测试验证 dotted path 导入失败的错误消息

### README

- **AC-8.1**: README 反映新的 YAML 格式和插件开发流程

---

## 6. 迁移

同步修改两个仓库。不需要向后兼容——当前只有 research-pipeline 一个消费者。

---

## 7. 不变条件

- 讨论循环逻辑不变
- 并行执行不变
- 错误处理策略不变
- 持久化归档不变
- Context 压缩策略不变
- Host 无 Tools 约束不变

---

## 8. 验收标准汇总

共 28 条验收标准（AC-1.1 ~ AC-8.1），覆盖：
- Tool 加载 (5)
- Per-agent tools (6)
- Context builder (3)
- Config (3)
- research-pipeline (4)
- 公共 API (2)
- 测试 (4)
- README (1)
