# 006 — Research Pipeline Tuning (研报选题管道调优)

> 状态：Implemented
> 前置：spec-001~005 已实现
> 范围：修复 research-pipeline 的数据扫描逻辑和配置，使选题讨论能基于真实研报数据运行

---

## 1. 概述

spec-001~005 完成了讨论框架的引擎和 prompt 优化，但第一次端到端测试（2026-04-08_2220）发现 **context 全部为空**：

```
## 已发布历史
暂无已发布文章记录。

## 近期研报标题
未找到近期研报。

## 当前热榜
热榜服务暂不可用，请基于研报数据和自身判断进行讨论。
```

框架引擎能跑，但选题用例从未产出过基于真实数据的讨论。本 spec 诊断并修复所有阻塞问题。

---

## 2. 诊断发现

### 2.1 研报扫描工具（research_list）的文件名解析失败

`~/ima-downloads/` 的实际目录和文件命名与工具假设不符：

| 假设 | 现实 | 影响 |
|------|------|------|
| 顶级目录名包含 "外资研报" 或 "内资研报" | 实际目录名为 `六、高盛、花旗、瑞银、摩根等外资研报🌏` 和 `七、中金、中信、华泰等内资研报🌈` | ✅ 子串匹配能命中，这部分没有问题 |
| 文件名格式统一为 `YYYYMMDD-机构-标题.pdf` | 存在至少 4 种格式变体 | ❌ 大量文件被跳过 |

文件名格式变体：

1. **标准格式**：`20260401-摩根大通-全球指数邮件.pdf`（大部分文件）
2. **下划线格式**：`20260408_巴克莱银行_HY研究_印度机场_中东动荡.pdf`（部分外资研报）
3. **前缀格式**：`深度研报-20260402-国信证券-宏观经济深度报告.pdf`（内资深度研报）
4. **6位年份**：`260401-高盛-中国人民银行报告.pdf`（部分外资研报，省略 "20" 前缀）

旧代码使用 `stem.split("-", 2)` 按固定位置解析，只能匹配格式 1，其余全部丢失。

### 2.2 搜索窗口太窄

`research_days: 2` 只看今天和昨天。研报同步虽然是每天跑，但实际有间歇，2 天窗口经常返回零结果。

### 2.3 published_file 路径问题

配置中 `published_file: "PUBLISHED.md"` 使用相对路径，运行时 CWD 可能不在 research-pipeline 目录下，导致找不到文件。

### 2.4 热榜服务未部署

NewsNow MCP Server 尚未部署，`get_trending` 始终返回降级消息。这是已知状态，不在本 spec 修复范围内，但需要确保 agent prompt 明确告知 agent 即使没有热榜数据也可以基于市场认知继续讨论。

### 2.5 Agent System Prompt 缺少工具使用指引

spec-005 指出 agent system_prompt 缺少"你有哪些 Tool 可用，什么时候用"的提示。agent 可能不知道自己有 Tool 来查研报和热榜。

---

## 3. 修复方案

### 3.1 重写文件名解析器

将硬编码的 `split("-", 2)` 替换为正则表达式解析器 `_parse_pdf_filename()`，支持所有 4 种变体：

```python
_DATE_PATTERN = re.compile(
    r"^(?:.*?[-_])?(\d{6,8})[-_](.+?)[-_](.+)\.pdf$", re.IGNORECASE
)
```

- 可选前缀（如 `深度研报-`）
- 6 位或 8 位日期（6 位自动补 `20` 前缀）
- 分隔符支持 `-` 和 `_`
- 提取为 `(date_str_8, institution, title)` 三元组

### 3.2 增大搜索窗口

`research_days` 从 2 改为 7，确保有足够的研报覆盖。

### 3.3 使用绝对路径

`published_file` 改为绝对路径 `/home/zhijiang/.openclaw/repos/research-pipeline/PUBLISHED.md`。

### 3.4 创建 PUBLISHED.md 初始文件

在 research-pipeline 根目录创建一个空表格结构的 PUBLISHED.md：

```markdown
# 已发布文章记录

| 日期 | 话题 | 核心论点 |
|------|------|----------|
```

### 3.5 重写 Agent System Prompt

按 spec-005 的指导标准重写两个 agent 的 system_prompt：
- 增加完整的工具使用指引（列出每个 tool 及使用场景）
- 调整角色描述为 2-3 句自然语言
- 增加讨论态度指引（何时坚持、何时让步）
- 明确即使热榜不可用也可基于市场认知讨论

### 3.6 API 配置

添加 `api_key: "env:ANTHROPIC_API_KEY"` 到 discussion 配置块。

---

## 4. 验收标准

- **AC-1**: `list_research(days=7)` 对 `~/ima-downloads/` 返回 >100 条结果（修复前为 0）
- **AC-2**: 所有 4 种文件名格式均被正确解析（有针对每种格式的单元测试）
- **AC-3**: 端到端运行 `python -m discuss_agent configs/topic_selection.yaml` 产出包含真实研报数据的讨论
- **AC-4**: Context 的"近期研报标题"部分非空
- **AC-5**: PUBLISHED.md 存在且 get_published 能正常读取（返回"暂无"而非报错）
- **AC-6**: 所有现有测试 + 新增测试通过
- **AC-7**: Agent system_prompt 包含工具使用指引

---

## 5. 不变条件

- 讨论引擎逻辑不变（engine.py 零改动）
- Tool 的 API 接口不变（参数和返回类型兼容）
- 目录结构匹配逻辑的子串检查方式不变（"外资研报"/"内资研报"）
- 行业/宏观/策略的分类过滤逻辑不变
- 热榜降级行为不变

---

## 6. 测试新增

| 测试 | 覆盖范围 |
|------|----------|
| `test_standard_format` | `YYYYMMDD-机构-标题.pdf` 格式解析 |
| `test_underscore_format` | `YYYYMMDD_机构_标题.pdf` 格式解析 |
| `test_prefixed_format` | `深度研报-YYYYMMDD-机构-标题.pdf` 格式解析 |
| `test_six_digit_year` | `YYMMDD-机构-标题.pdf` 日期补全 |
| `test_finds_reports_with_emoji_dir_names` | 真实目录名（带 emoji）的扫描 |
| `test_finds_prefixed_filenames` | 前缀格式在完整扫描流程中的匹配 |
| `test_finds_underscore_filenames` | 下划线格式在完整扫描流程中的匹配 |
