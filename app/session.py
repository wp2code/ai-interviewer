"""会话编排层：把状态机、策略引擎、评分引擎与模型适配粘合成一场面试。

分层原则：
- domain/  纯逻辑，无 I/O，可单测
- adapters/ 与外部世界打交道（LLM / ASR / TTS），全部走协议接口
- session.py  只做编排与状态维护，不含业务规则
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .adapters.llm import LLMClient, MockLLM
from .domain.rubric import COMPETENCY_BY_KEY, COMPETENCIES, InterviewPlan
from .domain.scoring import Evidence, InterviewReport, ScoringEngine
from .domain.strategy import AnswerSignals, Decision, MoveKind, StrategyEngine
from .domain.turn import (
    Action,
    ActionKind,
    Event,
    EventKind,
    TurnMachine,
    TurnPolicy,
    TurnState,
)


@dataclass
class SessionConfig:
    job_title: str = "后端工程师"
    resume: str = ""
    total_budget_ms: int = 25 * 60_000
    language: str = "zh-CN"


@dataclass
class TurnRecord:
    turn_index: int
    question: str
    answer: str
    ts_start_ms: int
    ts_end_ms: int
    dimension: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "question": self.question,
            "answer": self.answer,
            "ts_start_ms": self.ts_start_ms,
            "ts_end_ms": self.ts_end_ms,
            "dimension": self.dimension,
        }


@dataclass
class InterviewSession:
    """一场面试的完整状态。"""

    session_id: str
    config: SessionConfig
    llm: LLMClient
    machine: TurnMachine
    strategy: StrategyEngine
    scorer: ScoringEngine
    evidence: dict[str, list[Evidence]] = field(
        default_factory=lambda: {k: [] for k in COMPETENCY_BY_KEY}
    )
    turns: list[TurnRecord] = field(default_factory=list)
    turn_texts: list[tuple[int, str]] = field(default_factory=list)
    last_question: str = ""
    last_decision: Decision | None = None
    started_at: float = field(default_factory=time.time)
    _t0: float = field(default_factory=time.monotonic)
    _turn_index: int = 0
    _pending: dict[str, Any] = field(default_factory=dict)

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    # ------------------------------------------------------------------
    async def handle_action(self, action: Action) -> list[dict[str, Any]]:
        """执行一个动作，返回要发给前端的消息。"""
        if action.kind is ActionKind.SAY:
            return [
                {
                    "type": "say",
                    "text": action.text,
                    "ts": self.elapsed_ms(),
                }
            ]
        if action.kind is ActionKind.LISTEN:
            return [{"type": "listen", "ts": self.elapsed_ms()}]
        if action.kind is ActionKind.CUT:
            return [{"type": "cut", "ts": self.elapsed_ms()}]
        if action.kind is ActionKind.FINALIZE_ASR:
            return [{"type": "finalize_asr", "ts": self.elapsed_ms()}]
        if action.kind is ActionKind.FINISH:
            report = self.build_report()
            return [
                {"type": "final_report", "report": report.to_dict(), "ts": self.elapsed_ms()}
            ]
        if action.kind is ActionKind.SCORE:
            return []
        if action.kind is ActionKind.DECIDE:
            await self._decide_and_speak(action.meta.get("utterance", ""))
            return []
        return []

    async def _decide_and_speak(self, answer: str) -> None:
        """决策层：抽信号 → 策略决策 → 生成话术 → 走状态机开口。"""
        dim_hint = self.last_decision.dimension if self.last_decision else ""
        signals: AnswerSignals = await self.llm.extract_signals(
            self.last_question, answer, dim_hint
        )
        self.strategy.observe(signals)

        decision = self.strategy.decide(signals, elapsed_ms=self.elapsed_ms())
        self.last_decision = decision

        if decision.kind is MoveKind.CLOSE:
            closing = await self.llm.compose_closing("")
            transition = self.machine.apply(Event(EventKind.BUDGET_EXHAUSTED, self.elapsed_ms()))
            for act in transition.actions:
                await self._emit(act)
            return

        # 抽证据（异步做，与生成话术并行）
        if decision.dimension and answer:
            evid_task = asyncio.create_task(
                self.llm.extract_evidence(
                    self.last_question,
                    answer,
                    decision.dimension,
                    self.elapsed_ms(),
                    self.elapsed_ms(),
                )
            )
        else:
            evid_task = None

        question_text = await self.llm.compose_question(
            decision, self.last_question
        )

        if evid_task:
            try:
                items = await evid_task
                for e in items:
                    self.evidence.setdefault(e.dimension, []).append(e)
            except Exception:
                pass  # 证据抽取失败不阻断面试

        self.last_question = question_text
        transition = self.machine.apply(
            Event(EventKind.MOVE_READY, self.elapsed_ms(), {"text": question_text})
        )
        for act in transition.actions:
            await self._emit(act)

    async def _emit(self, action: Action) -> None:
        for msg in await self.handle_action(action):
            self._pending.setdefault("outbox", []).append(msg)

    def drain(self) -> list[dict[str, Any]]:
        msgs = self._pending.pop("outbox", [])
        return msgs

    def build_report(self) -> InterviewReport:
        weights = {c.key: c.weight for c in COMPETENCIES}
        return self.scorer.build_report(self.evidence, weights, self.turn_texts)

    def record_turn(self, answer: str, ts_start_ms: int, ts_end_ms: int) -> None:
        self._turn_index += 1
        rec = TurnRecord(
            turn_index=self._turn_index,
            question=self.last_question,
            answer=answer,
            ts_start_ms=ts_start_ms,
            ts_end_ms=ts_end_ms,
            dimension=self.last_decision.dimension if self.last_decision else "",
        )
        self.turns.append(rec)
        self.turn_texts.append((ts_start_ms, answer))


class SessionManager:
    """会话生命周期管理。生产环境应换成 Redis + 数据库持久化。"""

    def __init__(self) -> None:
        self._sessions: dict[str, InterviewSession] = {}

    def create(
        self,
        session_id: str,
        config: SessionConfig | None = None,
        llm: LLMClient | None = None,
    ) -> InterviewSession:
        config = config or SessionConfig()
        policy = TurnPolicy(total_budget_ms=config.total_budget_ms)
        session = InterviewSession(
            session_id=session_id,
            config=config,
            llm=llm or MockLLM(),
            machine=TurnMachine(policy),
            strategy=StrategyEngine(
                InterviewPlan(total_budget_ms=config.total_budget_ms)
            ),
            scorer=ScoringEngine(),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> InterviewSession | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
