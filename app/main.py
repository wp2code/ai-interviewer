"""FastAPI 服务：WebSocket 会话接口 + 报告查询接口。

协议（前后端 JSON 消息）：

客户端 → 服务端
  {"type": "start", "job_title": "...", "resume": "..."}
  {"type": "vad", "speech_prob": 0.0~1.0, "t_ms": 12345}   # 浏览器端 VAD 帧
  {"type": "asr_final", "text": "...", "ts_start_ms": 0, "ts_end_ms": 5000}
  {"type": "bot_speech_done"}
  {"type": "barge_in", "speech_ms": 800}
  {"type": "stop"}

服务端 → 客户端
  {"type": "say", "text": "..."}          # 需要播报
  {"type": "listen"}                       # 开始聆听
  {"type": "cut"}                          # 立即停止播报
  {"type": "final_report", "report": {...}}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .adapters.llm import MockLLM, OpenAICompatLLM
from .domain.turn import (
    Action,
    ActionKind,
    EndpointDetector,
    Event,
    EventKind,
    TurnPolicy,
)
from .session import SessionConfig, SessionManager

app = FastAPI(title="AI 面试机器人")
manager = SessionManager()


def _build_llm():
    """有配 LLM 环境变量就用真实模型，否则用本地 mock，保证开箱可跑。"""
    import os

    if os.getenv("LLM_API_KEY") or os.getenv("LLM_BASE_URL"):
        return OpenAICompatLLM()
    return MockLLM()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    with open("web/index.html", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws/interview")
async def interview_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = None
    detector: EndpointDetector | None = None
    pending_vad: list[dict[str, Any]] = []

    async def send(msg: dict[str, Any]) -> None:
        await ws.send_text(json.dumps(msg, ensure_ascii=False))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send({"type": "error", "message": "非法 JSON"})
                continue

            mtype = msg.get("type")

            if mtype == "start":
                sid = uuid.uuid4().hex[:12]
                config = SessionConfig(
                    job_title=msg.get("job_title") or "后端工程师",
                    resume=msg.get("resume") or "",
                )
                session = manager.create(sid, config, _build_llm())
                detector = EndpointDetector(TurnPolicy(total_budget_ms=config.total_budget_ms))
                await send({"type": "session_started", "session_id": sid})

                transition = session.machine.apply(Event(EventKind.START, session.elapsed_ms()))
                for action in transition.actions:
                    await _dispatch(session, action, send)
                continue

            if session is None or detector is None:
                await send({"type": "error", "message": "会话未初始化"})
                continue

            if mtype == "vad":
                frame = msg.get("speech_prob", 0.0)
                t_ms = int(msg.get("t_ms", session.elapsed_ms()))
                from .domain.turn import VadFrame

                events = detector.feed(VadFrame(t_ms, float(frame)))
                for ev in events:
                    transition = session.machine.apply(ev)
                    for action in transition.actions:
                        await _dispatch(session, action, send)
                continue

            if mtype == "asr_final":
                text = msg.get("text", "").strip()
                ts_start = int(msg.get("ts_start_ms", session.elapsed_ms()))
                ts_end = int(msg.get("ts_end_ms", session.elapsed_ms()))
                if text:
                    session.record_turn(text, ts_start, ts_end)
                transition = session.machine.apply(
                    Event(
                        EventKind.UTTERANCE_FINAL,
                        session.elapsed_ms(),
                        {"text": text},
                    )
                )
                for action in transition.actions:
                    await _dispatch(session, action, send)
                # DECIDE 是异步的，等它完成再把话术推给前端
                await asyncio.sleep(0)
                for out in session.drain():
                    await send(out)
                continue

            if mtype == "bot_speech_done":
                transition = session.machine.apply(
                    Event(EventKind.BOT_SPEECH_DONE, session.elapsed_ms())
                )
                for action in transition.actions:
                    await _dispatch(session, action, send)
                for out in session.drain():
                    await send(out)
                continue

            if mtype == "barge_in":
                transition = session.machine.apply(
                    Event(
                        EventKind.BARGE_IN,
                        session.elapsed_ms(),
                        {"speech_ms": int(msg.get("speech_ms", 0))},
                    )
                )
                for action in transition.actions:
                    await _dispatch(session, action, send)
                continue

            if mtype == "timeout":
                transition = session.machine.apply(
                    Event(EventKind.TIMEOUT, session.elapsed_ms())
                )
                for action in transition.actions:
                    await _dispatch(session, action, send)
                continue

            if mtype == "stop":
                transition = session.machine.apply(
                    Event(EventKind.STOP, session.elapsed_ms())
                )
                for action in transition.actions:
                    await _dispatch(session, action, send)
                for out in session.drain():
                    await send(out)
                continue

            await send({"type": "error", "message": f"未知消息类型：{mtype}"})

    except WebSocketDisconnect:
        pass
    finally:
        if session is not None:
            manager.drop(session.session_id)


async def _dispatch(session, action: Action, send) -> None:
    msgs = await session.handle_action(action)
    for m in msgs:
        await send(m)


@app.get("/api/report/{session_id}")
async def get_report(session_id: str) -> dict[str, Any]:
    session = manager.get(session_id)
    if session is None:
        return {"error": "会话不存在或已过期"}
    return session.build_report().to_dict()
