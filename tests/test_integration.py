"""端到端集成测试：模拟一场完整面试，验证编排层把各域串起来。

用 MockLLM 离线跑，不依赖网络。验证重点：
- 事件顺序正确（说话 → 决策 → 开口 → 回到聆听）
- 证据被正确累积到对应维度
- 最终能产出带证据的报告
"""

from __future__ import annotations

import asyncio

from app.adapters.llm import MockLLM
from app.domain.turn import Event, EventKind, TurnState
from app.session import SessionConfig, SessionManager


async def run_full_interview() -> None:
    mgr = SessionManager()
    session = mgr.create(
        "test-session",
        SessionConfig(job_title="后端工程师", resume="5 年后端，高并发交易系统"),
        MockLLM(),
    )

    answers = [
        "我主导过电商大促的支付链路重构。峰值下单量是平时二十倍，"
        "支付接口 P99 延迟从 800 毫秒涨到 5 秒。我把同步扣减改成异步队列削峰，"
        "给热点商品做本地缓存预热，日志落盘改成异步。优化后 P99 降到 120 毫秒，"
        "大促期间零资损。",
        "这个方案里我具体负责后端服务。前端是另一个同事做的。"
        "我主导了整体技术选型和上线方案，缓存方案是我提的。",
        "当时备选方案有同步改 Redis 集群和直接扩容机器两个。我放弃了扩容，"
        "因为成本太高而且治标不治本。选了异步队列，因为它能同时解决削峰和可观测性。",
    ]

    # 开场
    t = session.machine.apply(Event(EventKind.START, 0))
    assert t.state is TurnState.GREETING
    assert t.actions[0].kind.value == "say"

    # 开场白播完
    t = session.machine.apply(Event(EventKind.BOT_SPEECH_DONE, 3000))
    assert t.state is TurnState.LISTENING

    for i, ans in enumerate(answers):
        base = 5000 + i * 40000
        session.machine.apply(Event(EventKind.SPEECH_STARTED, base))
        session.record_turn(ans, base, base + 8000)
        t = session.machine.apply(
            Event(EventKind.UTTERANCE_FINAL, base + 8000, {"text": ans})
        )
        assert t.state is TurnState.THINKING

        # 执行 DECIDE 动作（编排层内部逻辑）
        for action in t.actions:
            await session.handle_action(action)
        session.drain()

        assert session.last_question, f"第 {i + 1} 轮未生成追问"
        assert session.last_decision is not None
        assert session.last_decision.reason, "决策必须带可审计的理由"

        t = session.machine.apply(
            Event(EventKind.MOVE_READY, base + 9000, {"text": session.last_question})
        )
        assert t.state is TurnState.SPEAKING
        t = session.machine.apply(Event(EventKind.BOT_SPEECH_DONE, base + 14000))
        assert t.state is TurnState.LISTENING

    # 证据累积检查
    total_evidence = sum(len(v) for v in session.evidence.values())
    assert total_evidence > 0, "未累积到任何证据"

    # 维度覆盖检查：不应所有维度都是 0
    report = session.build_report()
    covered = [d for d in report.dimensions if d.level > 0]
    assert covered, "没有任何维度被覆盖"

    # 每条证据都要能回溯
    for d in report.dimensions:
        for e in d.evidence:
            assert e.quote, "证据必须有原话"
            assert e.ts_start_ms >= 0

    # 报告结构完整
    assert report.recommendation in ("advance", "hold", "reject")
    assert 0 <= report.total_score_100 <= 100
    assert 0 <= report.overall_confidence <= 1
    d = report.to_dict()
    assert "dimensions" in d and "red_flags" in d

    print(f"✓ 完整面试跑通：{len(answers)} 轮")
    print(f"✓ 覆盖维度：{[x.dimension_name for x in covered]}")
    print(f"✓ 证据条数：{total_evidence}")
    print(f"✓ 综合分：{report.total_score_100} | 置信度：{report.overall_confidence}")
    print(f"✓ 结论：{report.recommendation}")
    print(f"✓ 红旗：{[f.title for f in report.red_flags] or '无'}")
    print(f"✓ 末轮提问：{session.last_question[:60]}")


def test_full_interview_flow():
    asyncio.run(run_full_interview())


def test_budget_closes_interview():
    """时间预算耗尽必须能收尾并产出报告。"""
    from app.domain.turn import TurnPolicy

    async def run():
        mgr = SessionManager()
        session = mgr.create("budget-test", SessionConfig(total_budget_ms=1), MockLLM())
        t = session.machine.apply(Event(EventKind.START, 0))
        session.machine.apply(Event(EventKind.BOT_SPEECH_DONE, 1000))
        t = session.machine.apply(Event(EventKind.BUDGET_EXHAUSTED, 5000))
        assert t.state is TurnState.CLOSING
        # 收尾问题问完 → 等回答
        t = session.machine.apply(Event(EventKind.BOT_SPEECH_DONE, 9000))
        assert t.state is TurnState.LISTENING
        # 候选人回答 → 决策 → 收尾语
        session.machine.apply(Event(EventKind.SPEECH_ENDED, 20000))
        session.machine.apply(
            Event(EventKind.UTTERANCE_FINAL, 20500, {"text": "我擅长高并发系统设计"})
        )
        t = session.machine.apply(
            Event(EventKind.MOVE_READY, 21000, {"text": "感谢您的时间。"})
        )
        assert t.state is TurnState.SPEAKING
        t = session.machine.apply(Event(EventKind.BOT_SPEECH_DONE, 22000))
        assert t.state is TurnState.ENDED
        assert any(a.kind.value == "finish" for a in t.actions)

    asyncio.run(run())
