"""LLM 适配层。

关键约束：**LLM 输出全部走结构化 schema**。
自由文本解析（正则抠 JSON、猜字段）是生产事故的头号来源，
一旦模型换个说法，整个流程静默失效且不报错。

提供两种实现：
- ``LLMClient`` 协议：业务层只依赖这个接口
- ``MockLLM``：离线可跑，用于单测与本地演示，不依赖任何网络
- ``OpenAICompatLLM``：兼容 OpenAI / vLLM / Ollama / 通义等 OpenAI 风格接口
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..domain.scoring import Evidence
from ..domain.strategy import AnswerSignals, Decision


@runtime_checkable
class LLMClient(Protocol):
    async def extract_signals(
        self, question: str, answer: str, dimension_hint: str
    ) -> AnswerSignals: ...

    async def extract_evidence(
        self, question: str, answer: str, dimension: str, ts_start_ms: int, ts_end_ms: int
    ) -> list[Evidence]: ...

    async def compose_question(
        self, decision: Decision, resume: str, tone: str = "专业友好"
    ) -> str: ...

    async def compose_opening(self, resume: str, job_title: str) -> str: ...

    async def compose_closing(self, report_summary: str) -> str: ...


SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hit_dimensions": {"type": "array", "items": {"type": "string"}},
        "depth_hint": {"type": "integer", "minimum": 1, "maximum": 5},
        "is_restatement": {"type": "boolean"},
        "is_evasive": {"type": "boolean"},
        "contradicts_previous": {"type": "boolean"},
        "contradiction_detail": {"type": "string"},
        "has_quantification": {"type": "boolean"},
        "answer_chars": {"type": "integer"},
        "summary": {"type": "string"},
    },
    "required": ["hit_dimensions", "depth_hint", "summary"],
}

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "level_hint": {"type": "integer", "minimum": 1, "maximum": 5},
                    "rationale": {"type": "string"},
                },
                "required": ["quote", "level_hint", "rationale"],
            },
        }
    },
    "required": ["items"],
}

SYSTEM_PROMPT = """你是一位资深技术面试官，正在协助完成一场结构化初筛。

硬性要求：
1. 只输出符合 schema 的 JSON，不要任何解释性文字或代码块标记。
2. 引用原话必须逐字摘自候选人回答，禁止改写或润色。
3. level_hint 依据 rubric 档位锚点判断，不是依据语气自信度或语气词。
4. 判断"空洞"看是否有具体动作、时间、数字，而不是看回答长度。
5. 识别矛盾只针对事实性冲突（数字、时间、角色归属），措辞差异不算矛盾。"""


@dataclass
class MockLLM:
    """离线可跑的假实现。基于启发式规则，足以驱动全链路与单测。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def extract_signals(
        self, question: str, answer: str, dimension_hint: str
    ) -> AnswerSignals:
        self.calls.append(("extract_signals", {"q": question, "a": answer[:50]}))
        length = len(answer)
        has_num = any(ch.isdigit() for ch in answer)
        has_quant = any(
            k in answer for k in ("%", "毫秒", "ms", "倍", "万", "人", "提升", "降到", "上线")
        )
        if length < 30:
            depth = 1
        elif length < 120:
            depth = 2
        elif has_quant:
            depth = 4
        else:
            depth = 3

        dims = [dimension_hint] if dimension_hint else []
        signals = AnswerSignals(
            hit_dimensions=dims,
            depth_hint=depth,
            is_restatement=length < 30 and question[:10] in answer,
            is_evasive=False,
            contradicts_previous=False,
            has_quantification=has_quant or has_num,
            answer_chars=length,
            summary=answer[:60],
        )
        return signals

    async def extract_evidence(
        self, question: str, answer: str, dimension: str, ts_start_ms: int, ts_end_ms: int
    ) -> list[Evidence]:
        self.calls.append(("extract_evidence", {"dimension": dimension}))
        if len(answer) < 20:
            return []
        quote = answer[: min(120, len(answer))]
        has_quant = any(k in answer for k in ("%", "倍", "提升", "降到", "万"))
        level = 4 if has_quant else (3 if len(answer) > 120 else 2)
        return [
            Evidence(
                dimension=dimension,
                quote=quote,
                ts_start_ms=ts_start_ms,
                ts_end_ms=ts_end_ms,
                level_hint=level,
                rationale="回答包含具体动作与结果，命中该档行为锚点" if level >= 3 else "仅有笼统描述",
                source="llm",
            )
        ]

    async def compose_question(
        self, decision: Decision, resume: str, tone: str = "专业友好"
    ) -> str:
        self.calls.append(("compose_question", {"intent": decision.prompt_intent}))
        # 策略意图里的括号标注是给模型看的，不该出现在候选人耳边
        intent = re.sub(r"（[^）]*）", "", decision.prompt_intent)
        intent = re.sub(r"考察维度【[^】]*】", "", intent).strip()
        parts = [p for p in intent.split("。") if p.strip()]
        return parts[-1].strip() if parts else "请再详细说说。"

    async def compose_opening(self, resume: str, job_title: str) -> str:
        return (
            f"您好，我是本场 {job_title} 面试的 AI 面试官。"
            "整个过程大约 20 到 25 分钟，我们会围绕您的项目经历和技术判断展开。"
            "请问您现在方便开始吗？"
        )

    async def compose_closing(self, report_summary: str) -> str:
        return "感谢您的时间，本次面试就到这里。后续结果会由招聘团队综合评估后通知您。"


@dataclass
class OpenAICompatLLM:
    """OpenAI 风格接口的通用实现。

    兼容 OpenAI、vLLM、Ollama、以及大量国内厂商的 OpenAI 兼容网关。
    """

    base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    timeout_s: float = 30.0
    temperature: float = 0.2

    async def _chat_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], temperature: float | None = None
    ) -> dict[str, Any]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    async def _chat_text(self, messages: list[dict[str, str]]) -> str:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    async def extract_signals(
        self, question: str, answer: str, dimension_hint: str
    ) -> AnswerSignals:
        data = await self._chat_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"维度候选：{dimension_hint or '未知'}\n"
                        f"面试官提问：{question}\n"
                        f"候选人回答：{answer}\n\n"
                        "抽取信号。"
                    ),
                },
            ],
            SIGNAL_SCHEMA,
        )
        return AnswerSignals.from_dict(data)

    async def extract_evidence(
        self, question: str, answer: str, dimension: str, ts_start_ms: int, ts_end_ms: int
    ) -> list[Evidence]:
        data = await self._chat_json(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"考察维度：{dimension}\n"
                        f"面试官提问：{question}\n"
                        f"候选人回答：{answer}\n\n"
                        "抽取支撑该维度的证据，最多 3 条。"
                    ),
                },
            ],
            EVIDENCE_SCHEMA,
        )
        return [
            Evidence(
                dimension=dimension,
                quote=it.get("quote", ""),
                ts_start_ms=ts_start_ms,
                ts_end_ms=ts_end_ms,
                level_hint=int(it.get("level_hint", 3)),
                rationale=it.get("rationale", ""),
                source="llm",
            )
            for it in data.get("items", [])
            if it.get("quote")
        ]

    async def compose_question(
        self, decision: Decision, resume: str, tone: str = "专业友好"
    ) -> str:
        resume_brief = resume[:500] if resume else ""
        return await self._chat_text(
            [
                {
                    "role": "system",
                    "content": (
                        f"你是一位{tone}的技术面试官，正在提问。"
                        "只输出要说的那一句话，不要解释、不要编号、不要客套。"
                        "长度控制在 60 字以内。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"提问意图：{decision.prompt_intent}\n"
                        f"候选人简历摘要：{resume_brief}\n"
                        f"你刚才的提问：{resume}\n\n"
                        "生成这句提问。"
                    ),
                },
            ]
        )

    async def compose_opening(self, resume: str, job_title: str) -> str:
        return await self._chat_text(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一位专业友好的面试官，正在开场。只输出开场白本身，"
                        "80 字以内，说明身份、预计时长、流程，然后问是否方便开始。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"岗位：{job_title}\n简历摘要：{resume[:500]}",
                },
            ]
        )

    async def compose_closing(self, report_summary: str) -> str:
        return await self._chat_text(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一位专业友好的面试官，正在结束面试。只输出结束语，"
                        "60 字以内，感谢并说明后续流程。不要透露任何评估内容或分数。"
                    ),
                },
                {"role": "user", "content": f"面试已结束。内部摘要：{report_summary}"},
            ]
        )
