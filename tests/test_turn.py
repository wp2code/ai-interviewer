"""状态机与端点检测的行为测试。

这些用例覆盖的都是真实事故场景：
- 候选人思考时停顿（误判说完）
- 候选人被自己的口水话打断（误判打断）
- 机器人刚开口就被打断
- 面试超时必须收尾
"""

from __future__ import annotations

from app.domain.turn import (
    EndpointDetector,
    Event,
    EventKind,
    TurnMachine,
    TurnPolicy,
    TurnState,
    VadFrame,
)


def kinds(actions) -> list[str]:
    return [a.kind.value for a in actions]


# --------------------------------------------------------------------------
# 端点检测
# --------------------------------------------------------------------------


def test_silence_ends_turn_after_threshold():
    det = EndpointDetector(TurnPolicy(silence_end_ms=700))
    events = det.feed(VadFrame(0, 0.9))
    assert [e.kind for e in events] == [EventKind.SPEECH_STARTED]  # 开始说话

    det.feed(VadFrame(100, 0.9))
    det.feed(VadFrame(200, 0.9))
    events = det.feed(VadFrame(300, 0.0))   # 进入静音
    events += det.feed(VadFrame(1000, 0.0))  # 静音满 700ms
    assert [e.kind for e in events] == [EventKind.SPEECH_ENDED]
    # 语音时长 = 从开口到进入静音，不含尾部静音
    assert events[0].payload["speech_ms"] == 300


def test_short_silence_does_not_end_turn():
    """候选人说一半停顿 300ms（<700ms 阈值），应保持同一轮。"""
    det = EndpointDetector(TurnPolicy(silence_end_ms=700))
    det.feed(VadFrame(0, 0.9))
    det.feed(VadFrame(100, 0.0))
    det.feed(VadFrame(300, 0.0))  # 静音 200ms，仍在阈值内
    events = det.feed(VadFrame(400, 0.9))  # 又开口 → 同一轮
    assert [e.kind for e in events] == []


def test_short_speech_is_not_treated_as_turn():
    """咳嗽/杂音（<250ms）不应产生端点，避免机器人抢话。"""
    det = EndpointDetector(TurnPolicy(silence_end_ms=700, min_speech_ms=250))
    det.feed(VadFrame(0, 0.9))
    det.feed(VadFrame(100, 0.0))
    events = det.feed(VadFrame(900, 0.0))
    assert [e.kind for e in events] == []


# --------------------------------------------------------------------------
# 状态机
# --------------------------------------------------------------------------


def test_happy_path_flow():
    m = TurnMachine()
    t = m.apply(Event(EventKind.START, 0))
    assert t.state is TurnState.GREETING
    assert kinds(t.actions) == ["say", "listen"]

    t = m.apply(Event(EventKind.BOT_SPEECH_DONE, 3000))
    assert t.state is TurnState.LISTENING

    t = m.apply(Event(EventKind.SPEECH_STARTED, 5000))
    assert t.state is TurnState.LISTENING

    t = m.apply(Event(EventKind.SPEECH_ENDED, 15000))
    assert t.state is TurnState.THINKING
    assert kinds(t.actions) == ["finalize_asr"]

    t = m.apply(Event(EventKind.UTTERANCE_FINAL, 15500, {"text": "我做过一个电商系统"}))
    assert kinds(t.actions) == ["score", "decide"]

    t = m.apply(Event(EventKind.MOVE_READY, 16500, {"text": "讲讲你的具体职责"}))
    assert t.state is TurnState.SPEAKING
    assert kinds(t.actions) == ["say"]


def test_barge_in_cuts_bot_speech():
    m = TurnMachine(TurnPolicy(barge_in_min_speech_ms=320, barge_in_grace_ms=400))
    m.apply(Event(EventKind.START, 0))
    m.apply(Event(EventKind.BOT_SPEECH_DONE, 1000))
    m.apply(Event(EventKind.SPEECH_STARTED, 2000))
    m.apply(Event(EventKind.SPEECH_ENDED, 8000))
    m.apply(Event(EventKind.UTTERANCE_FINAL, 8200, {"text": "回答"}))
    m.apply(Event(EventKind.MOVE_READY, 9000, {"text": "追问一下"}))

    # 机器人在 9000ms 开口，10000ms 时被打断（距开口 1000ms > 400ms 宽限期）
    t = m.apply(Event(EventKind.BARGE_IN, 10000, {"speech_ms": 500}))
    assert t.state is TurnState.LISTENING
    assert "cut" in kinds(t.actions)


def test_barge_in_within_grace_period_is_ignored():
    """机器人刚开口 200ms 内被打断，应忽略 —— 否则开场就被自己的回声切掉。"""
    m = TurnMachine(TurnPolicy(barge_in_grace_ms=400))
    m.apply(Event(EventKind.START, 0))
    m.apply(Event(EventKind.BOT_SPEECH_DONE, 1000))
    m.apply(Event(EventKind.SPEECH_STARTED, 2000))
    m.apply(Event(EventKind.SPEECH_ENDED, 8000))
    m.apply(Event(EventKind.UTTERANCE_FINAL, 8200, {"text": "回答"}))
    m.apply(Event(EventKind.MOVE_READY, 9000, {"text": "追问"}))

    t = m.apply(Event(EventKind.BARGE_IN, 9200, {"speech_ms": 500}))
    assert t.state is TurnState.SPEAKING
    assert "cut" not in kinds(t.actions)


def test_short_barge_in_is_treated_as_filler():
    """"嗯嗯"这种 200ms 的插话不能打断。"""
    m = TurnMachine(TurnPolicy(barge_in_min_speech_ms=320))
    m.apply(Event(EventKind.START, 0))
    m.apply(Event(EventKind.BOT_SPEECH_DONE, 1000))
    m.apply(Event(EventKind.SPEECH_STARTED, 2000))
    m.apply(Event(EventKind.SPEECH_ENDED, 8000))
    m.apply(Event(EventKind.UTTERANCE_FINAL, 8200, {"text": "回答"}))
    m.apply(Event(EventKind.MOVE_READY, 9000, {"text": "追问"}))

    t = m.apply(Event(EventKind.BARGE_IN, 12000, {"speech_ms": 150}))
    assert t.state is TurnState.SPEAKING
    assert "cut" not in kinds(t.actions)


def test_budget_exhaustion_forces_closing():
    m = TurnMachine(TurnPolicy(total_budget_ms=60_000, closing_tail_ms=30_000))
    m.apply(Event(EventKind.START, 0))
    t = m.apply(Event(EventKind.BUDGET_EXHAUSTED, 60_000))
    assert t.state is TurnState.CLOSING
    assert "say" in kinds(t.actions)


def test_closing_finishes_session():
    m = TurnMachine()
    m.apply(Event(EventKind.START, 0))
    m.apply(Event(EventKind.BUDGET_EXHAUSTED, 1000))
    m.apply(Event(EventKind.MOVE_READY, 1500, {"text": "收尾"}))
    t = m.apply(Event(EventKind.BOT_SPEECH_DONE, 4000))
    assert t.state is TurnState.ENDED
    assert "finish" in kinds(t.actions)


def test_speech_ended_ignored_outside_listening():
    m = TurnMachine()
    m.apply(Event(EventKind.START, 0))
    t = m.apply(Event(EventKind.SPEECH_ENDED, 1000))
    assert t.state is TurnState.GREETING
    assert kinds(t.actions) == ["none"]


def test_listen_timeout_prompts_candidate():
    m = TurnMachine()
    m.apply(Event(EventKind.START, 0))
    m.apply(Event(EventKind.BOT_SPEECH_DONE, 1000))
    t = m.apply(Event(EventKind.TIMEOUT, 65000))
    assert t.state is TurnState.SPEAKING
    assert "say" in kinds(t.actions)
