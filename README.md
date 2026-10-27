# AI 自动面试机器人

招聘批量筛选场景的实时语音面试系统。Python / FastAPI 全栈。

**完整设计说明见 [`design/架构设计.md`](design/架构设计.md)**

---

## 这个系统在解决什么

不是"让机器人问问题"——那不难。难在四件事，都已落地为可运行代码：

| 难点 | 解法 | 位置 |
|---|---|---|
| 抢话 / 冷场 | 事件驱动状态机，**端点检测与打断检测分离建模** | `app/domain/turn.py` |
| 固定题库 = 模板化面试 | **LLM 只抽信号，确定性引擎做决策**，追问沿深度阶梯自适应 | `app/domain/strategy.py` |
| 黑箱分数无法申诉 | 证据锚定 + rubric 行为锚点 + 置信度有来源 | `app/domain/scoring.py` |
| 优化方向搞反 | 实测：**ASR 占成本 96.2%，LLM 只占 2.1%** | 设计文档 §7 |

---

## 快速开始

```bash
pip install fastapi uvicorn websockets pytest

python -m uvicorn app.main:app --port 8765
# 浏览器打开 http://127.0.0.1:8765
```

不配任何 API Key 也能跑通 —— 自动降级为本地 `MockLLM`，全链路离线可用。

**接真实模型**（可选）：

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-xxx
export LLM_MODEL=gpt-4o-mini
```

兼容 OpenAI / vLLM / Ollama 及各家 OpenAI 兼容网关。

---

## 测试

```bash
python -m pytest tests -q      # 28 项单元 + 集成测试
python tests/ws_smoke.py       # WebSocket 端到端协议冒烟
```

测试覆盖的都是真实事故场景：候选人思考停顿、咳嗽声误判、自己口水话误判为插话、机器人刚开口就被打断、ASR 与端点信号竞态、超时强制收尾。

---

## 目录

```
app/
  domain/      纯逻辑，零 I/O，可穷举测试
    turn.py        轮次状态机 + 端点检测
    rubric.py      能力模型 + 评分档位 + 追问阶梯
    strategy.py    追问策略引擎（确定性决策）
    scoring.py     评分引擎 + 红旗检测
  adapters/
    llm.py         LLM 协议 + Mock + OpenAI 兼容实现
  session.py       会话编排
  main.py          FastAPI + WebSocket
web/index.html     浏览器端（Web Speech + Web Audio VAD）
design/架构设计.md  完整设计文档
```

---

## 设计要点速览

**1. 端点与打断分离**

```
端点检测 = 静音多久算说完   (时间维度, 700ms)
打断检测 = 检测到人声算插话 (能量维度, 320ms 持续)
```

混在一起会导致候选人插话时机器人还在念稿。

**2. LLM 只抽信号，不做决策**

```
回答 → [LLM 抽信号] → AnswerSignals → [规则引擎决策] → Decision → [LLM 写成话术]
```

决策层是纯函数，可单测、可回放、可审计。`Decision.reason` 记录"为什么问这个"。

**3. 每个分数都要能回答三问**

1. 依据哪句原话？（quote + 时间戳，可回听）
2. 对应 rubric 哪一档？（行为锚点）
3. 有多确信？（置信度，且置信度本身有来源）

**4. 红旗阈值宁严勿松**

数值矛盾用 2 倍阈值 —— 1.5 倍 vs 4 倍是真矛盾，2 倍 vs 2.1 倍往往只是口径差异。红旗噪音会让 HR 忽略所有红旗，整个风控体系就废了。

---

## 生产化前必读

- [ ] **知情同意**：开场白必须告知"这是 AI 面试官，正在录音"并记录同意时间戳
- [ ] **校准集**：收集 100+ 人工面试评分，算与 AI 评分的相关性（目标 r ≥ 0.75）—— 没有校准，HR 不会信
- [ ] **人工复核工作台**：AI 结论不能是唯一依据，覆写需留档
- [ ] **敏感维度拦截**：面试中涉及婚育、年龄的问题，转写标记后单独人工复核，不进自动评分
- [ ] **ASR 自建部署**：成本降 83%，是第一杠杆
