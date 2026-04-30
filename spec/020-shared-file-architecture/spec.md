# spec-020: discuss-agent v2 — 共享文件 + 持续对话架构

## 概述

将 discuss-agent 从"无状态轮次制"改为"共享文件 + 持续对话 + 程序 merge + host 裁决"架构。

## 核心设计原则

1. **每个 agent 有自己的 message history**（multi-turn conversation）
2. **主文件 claims.md 是单一真相来源**，只有程序能写
3. **程序推送增量更新**给 agent，不需要重读全文
4. **Agent 可以随时调用 tool**（search、grep、read）
5. **收敛由程序+host LLM 双层判断**

## 数据流

```
Round 1（初始化）:
  程序 → Agent A message[0]: "你是XX分析师。议题如下。请提出观点。"
  Agent A: [思考] [调tool搜索] [输出 CLAIM/观点]
  Agent A message history: [system, user_r1, assistant_r1]
  
  程序 → Agent B/C/D: 同上（并发）
  
  所有 agent 完成后:
  程序: parse 各 agent 输出 → merge 到 claims.md → host LLM 判断收敛

Round 2（增量推送）:
  程序 → Agent A message[2]: 
    "有更新：
     - CLAIM:X 被 B 反驳了：'B的反驳内容...'
     - CLAIM:Y 被 host 裁决为 CLOSED:共识
     - C 提出了新 CLAIM:Z
     当前所有 OPEN claims: [X, Z]
     请回应。"
  Agent A: [看到 B 的反驳] [记得自己上轮说了什么] [回应]
  Agent A message history: [system, user_r1, assistant_r1, user_r2, assistant_r2]

  程序: merge → host 判断 → 更新
  
Round N:
  程序 → Agent A message[2N-2]:
    "最新更新：
     - CLAIM:X 你和 B 对3904万头的数据来源达成一致
     - CLAIM:Z 还在 OPEN，没人反驳你
     当前 OPEN claims: [Z]
     请回应或确认。"
  ...
```

## Agent 的 message history 结构

```python
agent_a_messages = [
    # Round 1
    {"role": "user", "content": "你是养殖产业分析师。议题：猪周期...请提出观点"},
    {"role": "assistant", "content": "##CLAIM:能繁去化 [OPEN]## ..."},
    
    # Round 2 — 程序推送增量
    {"role": "user", "content": "更新：CLAIM:能繁去化被反驳了...请回应"},
    {"role": "assistant", "content": "[REBUTTAL TO:...] 我补充数据..."},
    
    # Round 3
    {"role": "user", "content": "更新：CLAIM:能繁去化已CLOSED:共识..."},
    {"role": "assistant", "content": "确认。关于CLAIM:Z我有新观点..."},
]
```

**每个 agent 的 history 只包含自己的对话**——不包含其他 agent 说了什么的全文。其他人的反驳通过"增量更新"摘要送进来。

## 程序推送给 Agent 的增量更新格式

**OPEN claims 全文直接放 prompt，CLOSED 只通知状态变化。**

随着讨论推进，OPEN claims 越来越少，prompt 自然越来越短。

```markdown
## 第{N}轮更新

状态变化：
- [已关闭] CLAIM:牧原成本优势 — CLOSED:共识
- [新增] CLAIM:饲料产量下降

以下是当前所有 OPEN claims 的完整讨论内容：

##CLAIM:能繁去化进度 [OPEN]##
[FROM:养殖产业分析师 @R1] 当前3904万头，目标3650万...
  [REBUTTAL FROM:财务审计分析师 @R1] 数据需交叉验证...
  [RESPONSE FROM:养殖产业分析师 @R2] 农业农村部月报第X页...
  [ACCEPT FROM:财务审计分析师 @R2] 确认数据来源可靠

##CLAIM:饲料产量下降 [OPEN]##
[FROM:侧面验证分析师 @R2] Q1猪饲料产量同比-8%...

## 你的任务
对每个 OPEN claim，你必须回应（除非你是提出者且没人反驳）：
- [REBUTTAL TO:关键词] 反驳 + 证据
- [ACCEPT TO:关键词] 接受 + 理由
- [NEW_CLAIM:关键词] 提出新论点
可以随时用 research_search 或 web_search 搜索证据。
```

**设计原则：**
- OPEN claims 全文进 prompt（含 FROM 标签，知道谁说了什么）
- CLOSED claims 不进 prompt（只通知关闭了）
- grep/read 保留给特殊需求（如想看 CLOSED 论点的完整讨论过程）
- 随着 CLOSE 增多，prompt 自然缩短

## 主文件 claims.md 格式

```markdown
# 讨论主文件

## 议题
猪周期当前位置与牧原股份分析

---

##CLAIM:能繁去化进度 [OPEN]##
[FROM:养殖产业分析师 @R1] 当前3904万头，目标3650万...
  [REBUTTAL FROM:财务审计分析师 @R1] 数据需交叉验证...
  [RESPONSE FROM:养殖产业分析师 @R2] 数据来源：农业农村部月报第X页...
  [ACCEPT FROM:财务审计分析师 @R2] 确认数据来源可靠

##CLAIM:牧原成本优势 [CLOSED:共识]##
[FROM:财务审计分析师 @R1] 头均14.5元 vs 行业16-17元
  [ACCEPT FROM:养殖产业分析师 @R1] 招商证券研报确认
  [ACCEPT FROM:宏观策略分析师 @R1] 同意
[HOST @R1] 裁决：各方一致认可。关闭。

##CLAIM:饲料产量下降 [OPEN]##
[FROM:侧面验证分析师 @R2] Q1猪饲料产量同比-8%...
```

## 程序 merge 逻辑

```python
class ClaimsManager:
    def __init__(self, claims_file: str):
        self.claims_file = claims_file
        self.claims: dict[str, Claim] = {}
    
    def parse(self):
        """解析 claims.md"""
        
    def merge_round(self, agent_outputs: list[AgentOutput]):
        """将一轮所有agent输出merge到主文件"""
        for output in agent_outputs:
            for response in output.parsed_responses:
                if response.type == "REBUTTAL":
                    self.claims[response.target].add_entry(
                        type="REBUTTAL",
                        from_agent=output.agent_name,
                        round=self.current_round,
                        content=response.content
                    )
                elif response.type == "ACCEPT":
                    self.claims[response.target].add_entry(
                        type="ACCEPT", ...
                    )
                elif response.type == "NEW_CLAIM":
                    self.claims[response.keyword] = Claim(
                        keyword=response.keyword,
                        status="OPEN",
                        entries=[Entry(type="FROM", ...)]
                    )
        self.save()
    
    def get_open_claims(self) -> list[Claim]:
        """返回所有 OPEN 的 claims"""
        return [c for c in self.claims.values() if c.status == "OPEN"]
    
    def close_claim(self, keyword: str, verdict: str, reason: str):
        """Host 裁决关闭"""
        self.claims[keyword].status = f"CLOSED:{verdict}"
        self.claims[keyword].add_entry(
            type="HOST", content=f"裁决：{reason}"
        )
        self.save()
    
    def generate_update_for_agent(self, agent_name: str, prev_round: int) -> str:
        """生成增量更新文本，只包含自上次以来的变化"""
        updates = []
        for claim in self.claims.values():
            new_entries = [e for e in claim.entries 
                         if e.round > prev_round and e.from_agent != agent_name]
            if new_entries:
                updates.append(format_update(claim, new_entries))
        return "\n".join(updates)
```

## Host 收敛判断

**程序层**（先执行，不调 LLM）：
```python
def check_convergence_precondition(claims: ClaimsManager) -> bool:
    """所有 OPEN claim 都被所有 agent 回应了吗？"""
    open_claims = claims.get_open_claims()
    for claim in open_claims:
        responding_agents = {e.from_agent for e in claim.entries if e.round == current_round}
        if len(responding_agents) < len(all_agents) - 1:  # 提出者不需要回应自己
            return False
    return True
```

**LLM层**（只在程序判断"可能收敛"时调用）：
```
Host prompt:
"以下是当前所有 OPEN claims 的讨论记录。
请对每个 claim 裁决：
- CLOSED:共识 — 各方达成一致
- CLOSED:分歧 — 讨论充分但立场不同，记录分歧
- CONTINUE — 仍需讨论

输出 JSON: [{claim: "X", verdict: "CLOSED:共识", reason: "..."}]"
```

## Agent 对话管理

```python
class AgentConversation:
    """管理单个 agent 的 multi-turn 对话"""
    
    def __init__(self, agent_config, system_prompt: str):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.model = build_claude(agent_config)
    
    async def send(self, user_message: str) -> str:
        """发送消息并获取回复，自动维护 history"""
        self.messages.append({"role": "user", "content": user_message})
        response = await self.model.create(messages=self.messages)
        assistant_msg = response.content
        self.messages.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg
    
    def get_history_length(self) -> int:
        return len(self.messages)
```

## 配置兼容

```yaml
# v2 模式
mode: shared_file  # 新增，默认 "rounds"（v1行为）

# 其他配置不变
agents: [...]
tools: [...]
limitation: "..."
```

## 总结

| | v1（当前） | v2（新） |
|--|-----------|---------|
| Agent 状态 | 无状态，每次新 prompt | **有状态，multi-turn history** |
| 上下文传递 | 全文塞 prompt | **增量更新，只推送变化** |
| 收敛判断 | LLM 拍脑袋 | **程序检查 + LLM 裁决** |
| 并发安全 | N/A | **Agent 并发读，程序串行写** |
| 信息完整性 | 截断/摘要丢信息 | **主文件保存全量** |
| Prompt 长度 | 随轮次增长 | **随 OPEN claims 减少而缩短** |
| 可追溯 | 按轮次 | **按 CLAIM + FROM + 轮次** |
