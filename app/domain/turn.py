"""轮次控制：实时语音面试的状态机与端点检测。

三个设计决策（决定了它是玩具还是能上生产的系统）：

1. **状态机是纯函数**。事件进 → 转换出，不碰 I/O、不碰网络。
   这样才能穷举测试，才敢在生产里跑。

2. **端点检测（候选人说完没）与打断检测（候选人插话没）分开建模**。
   很多实现混在一起，结果候选人插话时机器人还在念稿，体感直接崩。
   端点 = 时间维度的"静音多久算说完"；打断 = 能量维度的"是否检测到人声"。

3. **感知延迟预算**：候选人说完 → 机器人开口 < 1.2s。
   这意味着不能等 ASR 完整结果才决策，必须用 VAD 静音直接切。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TurnState(str, Enum):
    IDLE = "idle"
    GREETING = "greeting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CLOSING = "closing"
    ENDED = "ended"


class EventKind(str, Enum):
    START = "start"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    UTTERANCE_FINAL = "utterance_final"
    MOVE_READY = "move_ready"
    BOT_SPEECH_DONE = "bot_speech_done"
    BARGE_IN = "barge_in"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STOP = "stop"


class ActionKind(str, Enum):
    SAY = "say"
    LISTEN = "listen"
    FINALIZE_ASR = "finalize_asr"
    SCORE = "score"
    DECIDE = "decide"
    CUT = "cut"
    FINISH = "finish"
    NONE = "none"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    at_ms: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    state: TurnState
    actions: tuple[Action, ...]
    reason: str


@dataclass
class TurnPolicy:
    """一轮对话的全部可调参数，集中在一处便于调优与 A/B。"""

    # 端点检测
    silence_end_ms: int = 700          # 静音多久判定"候选人说完"
    silence_resume_ms: int = 260        # 短静音后又有声，视为同一轮（处理"呃...那个..."）
    speech_threshold: float = 0.5      # VAD 语音概率阈值
    min_speech_ms: int = 250           # 短于此视为咳嗽/杂音
    max_turn_ms: int = 90_000          # 单轮硬上限，防止跑飞

    # 打断
    barge_in_min_speech_ms: int = 320  # 插话至少持续这么久才算真插话
    barge_in_grace_ms: int = 400       # 机器人开口后多久内不响应打断（防误切）

    # 时长预算
    total_budget_ms: int = 25 * 60_000
    closing_tail_ms: int = 90_000      # 剩余时间少于此时进入收尾

    # 节奏
    think_timeout_ms: int = 8_000      # 决策超时 → 追问"您还在吗"
    speak_timeout_ms: int = 60_000


# --------------------------------------------------------------------------
# 端点检测
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VadFrame:
    t_ms: int
    speech_prob: float


class EndpointDetector:
    """把 VAD 帧流切成"说话段"。

    它只关心两件事：什么时候开始说话、什么时候确定说完。
    打断检测不在这里 —— 那是播放器侧的职责。
    """

    def __init__(self, policy: TurnPolicy) -> None:
        self.policy = policy
        self._speaking = False
        self._speech_start_ms = 0
        self._last_speech_ms = 0
        self._silence_start_ms: int | None = None

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def feed(self, frame: VadFrame) -> list[Event]:
        events: list[Event] = []
        is_speech = frame.speech_prob >= self.policy.speech_threshold

        if is_speech:
            self._last_speech_ms = frame.t_ms
            if self._silence_start_ms is not None:
                self._silence_start_ms = None

            if not self._speaking:
                # 连续静音期内又开口 → 同一轮的后续补充，不是新的一轮
                self._speaking = True
                self._speech_start_ms = frame.t_ms
                events.append(Event(EventKind.SPEECH_STARTED, frame.t_ms))

        elif self._speaking:
            if self._silence_start_ms is None:
                self._silence_start_ms = frame.t_ms
            else:
                silence = frame.t_ms - self._silence_start_ms
                if silence >= self.policy.silence_end_ms:
                    # 语音时长 = 从开口到进入静音，不含尾部静音
                    speech_ms = self._silence_start_ms - self._speech_start_ms
                    if speech_ms >= self.policy.min_speech_ms:
                        events.append(
                            Event(
                                EventKind.SPEECH_ENDED,
                                frame.t_ms,
                                {"speech_ms": speech_ms},
                            )
                        )
                    # 无论是否达阈值都要复位，否则一次咳嗽会让检测器卡在
                    # "说话中" 状态，后续真实发言的开头会被吞掉（不触发 SPEECH_STARTED）
                    self._speaking = False
                    self._silence_start_ms = None
        return events

    def force_end(self, at_ms: int) -> Event | None:
        if not self._speaking:
            return None
        duration = at_ms - self._speech_start_ms
        self._speaking = False
        self._silence_start_ms = None
        return Event(EventKind.SPEECH_ENDED, at_ms, {"speech_ms": duration, "forced": True})


# --------------------------------------------------------------------------
# 状态机
# --------------------------------------------------------------------------


class TurnMachine:
    """事件驱动的轮次状态机。纯逻辑，无 I/O。"""

    def __init__(self, policy: TurnPolicy | None = None) -> None:
        self.policy = policy or TurnPolicy()
        self.state = TurnState.IDLE
        self.started_at_ms: int | None = None
        self.bot_speak_started_ms: int | None = None
        self.history: list[Transition] = []
        # 收尾模式：一旦触发预算，进入收尾流程，最终必须走到 ENDED
        self._closing_mode = False
        self._final_remarks = False

    # -- helpers ---------------------------------------------------------
    def elapsed_ms(self, now_ms: int) -> int:
        if self.started_at_ms is None:
            return 0
        return now_ms - self.started_at_ms

    def _transition(
        self, state: TurnState, actions: tuple[Action, ...], reason: str
    ) -> Transition:
        self.state = state
        t = Transition(state=state, actions=actions, reason=reason)
        self.history.append(t)
        return t

    # -- main ------------------------------------------------------------
    def apply(self, event: Event) -> Transition:
        handler = getattr(self, f"_on_{event.kind.value}", None)
        if handler is None:
            return self._transition(self.state, (Action(ActionKind.NONE),), "未知事件")
        return handler(event)  # type: ignore[no-any-return]

    # -- handlers --------------------------------------------------------
    def _on_start(self, event: Event) -> Transition:
        self.started_at_ms = event.at_ms
        return self._transition(
            TurnState.GREETING,
            (
                Action(ActionKind.SAY, "您好，我是本场面试的 AI 面试官。整个过程大约 20 到 25 分钟，我们会围绕您的项目经历和技术判断展开。请问您现在方便开始吗？"),
                Action(ActionKind.LISTEN, meta={"timeout_ms": 60_000}),
            ),
            "开场",
        )

    def _on_speech_started(self, event: Event) -> Transition:
        if self.state in (TurnState.GREETING, TurnState.CLOSING):
            # 开场/收尾语被打断：礼貌收住
            return self._transition(
                TurnState.LISTENING,
                (Action(ActionKind.CUT), Action(ActionKind.LISTEN)),
                "开场语被打断，转入聆听",
            )
        if self.state is TurnState.SPEAKING:
            return self._on_barge_in(event)
        if self.state is TurnState.LISTENING:
            return self._transition(self.state, (Action(ActionKind.NONE),), "继续聆听")
        return self._transition(self.state, (Action(ActionKind.NONE),), "忽略")

    def _on_speech_ended(self, event: Event) -> Transition:
        if self.state is not TurnState.LISTENING:
            return self._transition(self.state, (Action(ActionKind.NONE),), "非聆听态，忽略端点")
        return self._transition(
            TurnState.THINKING,
            (Action(ActionKind.FINALIZE_ASR),),
            f"端点到达（说话 {event.payload.get('speech_ms', 0)}ms），请求 ASR 收尾",
        )

    def _on_utterance_final(self, event: Event) -> Transition:
        # 允许在 LISTENING 态直接接收最终转写。
        # 原因：ASR 收尾与 VAD 端点是两条独立的信号链，存在竞态——
        # ASR 结果可能先于端点判定到达。若此时丢弃转写，整轮回答就丢了。
        # THINKING 态收到则说明端点已到，正常进入评分与决策。
        if self.state is TurnState.LISTENING:
            return self._transition(
                TurnState.THINKING,
                (
                    Action(ActionKind.SCORE, meta={"utterance": event.payload.get("text", "")}),
                    Action(ActionKind.DECIDE, meta={"utterance": event.payload.get("text", "")}),
                ),
                "转写先于端点到达，直接进入决策",
            )
        if self.state is not TurnState.THINKING:
            return self._transition(self.state, (Action(ActionKind.NONE),), "非思考态，忽略转写")
        return self._transition(
            TurnState.THINKING,
            (
                Action(ActionKind.SCORE, meta={"utterance": event.payload.get("text", "")}),
                Action(ActionKind.DECIDE, meta={"utterance": event.payload.get("text", "")}),
            ),
            "拿到最终转写，开始评分与决策",
        )

    def _on_move_ready(self, event: Event) -> Transition:
        text = event.payload.get("text", "")
        if not text:
            return self._transition(
                TurnState.THINKING,
                (Action(ActionKind.LISTEN, meta={"timeout_ms": self.policy.think_timeout_ms}),),
                "决策返回空，转为等待",
            )
        if self.state is TurnState.THINKING:
            self.bot_speak_started_ms = event.at_ms
            self._final_remarks = self._closing_mode
            return self._transition(
                TurnState.SPEAKING,
                (Action(ActionKind.SAY, text),),
                "决策就绪，开口",
            )
        if self.state is TurnState.CLOSING:
            self.bot_speak_started_ms = event.at_ms
            self._final_remarks = True
            return self._transition(
                TurnState.SPEAKING,
                (Action(ActionKind.SAY, text),),
                "收尾语，开始播报",
            )
        return self._transition(self.state, (Action(ActionKind.NONE),), "非思考态，忽略")

    def _on_bot_speech_done(self, event: Event) -> Transition:
        if self.state is TurnState.GREETING:
            return self._transition(
                TurnState.LISTENING,
                (Action(ActionKind.LISTEN),),
                "开场白播报完毕，转入聆听",
            )
        if self.state is TurnState.SPEAKING:
            if self._final_remarks:
                return self._transition(
                    TurnState.ENDED,
                    (Action(ActionKind.FINISH),),
                    "收尾语播报完毕，面试结束",
                )
            return self._transition(
                TurnState.LISTENING,
                (Action(ActionKind.LISTEN),),
                "播报完毕，转入聆听",
            )
        if self.state is TurnState.CLOSING:
            # 收尾问题已问出，等候选人回答
            return self._transition(
                TurnState.LISTENING,
                (Action(ActionKind.LISTEN),),
                "收尾问题已问出，等待回答",
            )
        return self._transition(self.state, (Action(ActionKind.NONE),), "忽略")

    def _on_barge_in(self, event: Event) -> Transition:
        speech_ms = event.payload.get("speech_ms", 0)
        if speech_ms < self.policy.barge_in_min_speech_ms:
            return self._transition(self.state, (Action(ActionKind.NONE),), "插话过短，视为语气词")
        if (
            self.bot_speak_started_ms is not None
            and event.at_ms - self.bot_speak_started_ms < self.policy.barge_in_grace_ms
        ):
            return self._transition(self.state, (Action(ActionKind.NONE),), "宽限期内，不打断")
        return self._transition(
            TurnState.LISTENING,
            (Action(ActionKind.CUT), Action(ActionKind.LISTEN)),
            "检测到有效打断，截断播报",
        )

    def _on_timeout(self, event: Event) -> Transition:
        if self.state is TurnState.LISTENING:
            return self._transition(
                TurnState.SPEAKING,
                (Action(ActionKind.SAY, "不好意思，我这边没有听到声音，您现在方便说话吗？"),),
                "聆听超时，触发唤醒",
            )
        if self.state is TurnState.THINKING:
            return self._transition(
                TurnState.SPEAKING,
                (Action(ActionKind.SAY, "让我再想一下。"),),
                "决策超时",
            )
        return self._transition(self.state, (Action(ActionKind.NONE),), "忽略超时")

    def _on_budget_exhausted(self, event: Event) -> Transition:
        if self.state in (TurnState.ENDED, TurnState.CLOSING):
            return self._transition(self.state, (Action(ActionKind.NONE),), "已在收尾")
        self._closing_mode = True
        return self._transition(
            TurnState.CLOSING,
            (
                Action(
                    ActionKind.SAY,
                    "好的，我们时间差不多到了。最后想请您补充一句：如果只能挑一件事证明您能胜任这个岗位，您会挑哪件？",
                ),
                Action(ActionKind.LISTEN),
            ),
            "触达总时长预算，进入收尾",
        )

    def _on_stop(self, event: Event) -> Transition:
        return self._transition(
            TurnState.ENDED, (Action(ActionKind.FINISH),), "外部中止"
        )
