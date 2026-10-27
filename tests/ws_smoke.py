"""WebSocket 端到端冒烟测试：连真实服务器，走完整协议。

验证前后端协议约定真的能跑通，而不是只在单测里自洽。
"""

from __future__ import annotations

import asyncio
import json

import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8765/ws/interview"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "job_title": "后端工程师",
            "resume": "5 年后端，高并发交易系统",
        }, ensure_ascii=False))

        got_session = False
        got_say = 0
        got_listen = False

        # 1) 应先收到 session_started
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "session_started", msg
        got_session = True

        # 2) 开场白 say
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "say", msg
        got_say += 1
        print(f"  开场白: {msg['text'][:40]}...")

        # 3) listen
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "listen", msg
        got_listen = True

        # 4) 告知服务端开场白播完了
        await ws.send(json.dumps({"type": "bot_speech_done"}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "listen", msg

        # 5) 模拟一轮回答
        answer = (
            "我主导过电商大促的支付链路重构。峰值下单量是平时二十倍，"
            "支付接口 P99 延迟从 800 毫秒涨到 5 秒。我把同步扣减改成异步队列削峰，"
            "给热点商品做本地缓存预热，日志落盘改成异步。优化后 P99 降到 120 毫秒。"
        )
        await ws.send(json.dumps({
            "type": "asr_final",
            "text": answer,
            "ts_start_ms": 5000,
            "ts_end_ms": 20000,
        }, ensure_ascii=False))

        # 服务端应回一个 say（追问）
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        assert msg["type"] == "say", msg
        got_say += 1
        print(f"  追问: {msg['text'][:50]}")
        assert "聚焦" not in msg["text"], "策略标注不应泄漏到候选人话术"
        assert "考察维度" not in msg["text"]

        # 6) 结束 → 拿报告
        await ws.send(json.dumps({"type": "stop"}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        assert msg["type"] == "final_report", msg
        report = msg["report"]
        print(f"  报告: 综合分 {report['total_score_100']} "
              f"置信度 {report['overall_confidence']} 结论 {report['recommendation']}")
        assert report["dimensions"], "报告应含维度分"
        assert report["recommendation"] in ("advance", "hold", "reject")

        assert got_session and got_say >= 2 and got_listen
        print("✓ WebSocket 全链路协议验证通过")


if __name__ == "__main__":
    asyncio.run(main())
