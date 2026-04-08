# Anthropic Prompt Engineering References

三份核心文档，用于指导我们所有 agent 的 prompt 优化：

1. `writing-tools-for-agents.md` - Tool description 设计原则
2. `prompting-best-practices.md` - 通用 prompt 工程
3. URLs:
   - https://www.anthropic.com/engineering/writing-tools-for-agents
   - https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices

核心原则摘要：
- 把 agent 当成聪明但全新的员工，需要显式指令
- 用 XML 标签结构化 prompt
- 3-5 个 few-shot examples 极大提升准确性
- Tool description 要像给新同事写文档一样写
- 给原则而非模式（principles over patterns）
- 引导思考过程而非指定思考内容
