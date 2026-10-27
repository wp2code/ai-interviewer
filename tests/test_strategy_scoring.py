"""策略引擎与评分引擎测试。"""

from __future__ import annotations

from app.domain.rubric import COMPETENCIES, COMPETENCY_BY_KEY, InterviewPlan
from app.domain.scoring import (
    Evidence,
    ScoringEngine,
    detect_filler_density,
    detect_we_ratio,
    find_contradictions,
)
from app.domain.strategy import AnswerSignals, MoveKind, StrategyEngine


# --------------------------------------------------------------------------
# 策略引擎
# --------------------------------------------------------------------------


def test_first_decision_enters_a_dimension():
    e = StrategyEngine()
    d = e.decide(AnswerSignals(), elapsed_ms=0)
    assert d.kind is MoveKind.ASK_MAIN
    assert d.dimension in COMPETENCY_BY_KEY


def test_gateway_competency_is_prioritized():
    """问题解决与技术深度是门槛维度，未覆盖时应优先于学习与成长。"""
    e = StrategyEngine()
    d = e.decide(AnswerSignals(), elapsed_ms=0)
    assert d.dimension in ("problem_solving", "technical_depth")


def test_empty_answer_triggers_rephrasing_not_deepening():
    """空洞回答应换角度重问，而不是直接加深度。"""
    e = StrategyEngine()
    e.decide(AnswerSignals(), elapsed_ms=0)
    d = e.decide(AnswerSignals(is_evasive=True, answer_chars=10), elapsed_ms=1000)
    assert d.kind is MoveKind.DEEPEN
    assert "具体" in d.prompt_intent or "切入" in d.prompt_intent


def test_contradiction_takes_priority_over_everything():
    e = StrategyEngine()
    e.decide(AnswerSignals(), elapsed_ms=0)
    d = e.decide(
        AnswerSignals(
            contradicts_previous=True,
            contradiction_detail="性能提升 30% 和 3 倍",
            hit_dimensions=["technical_depth"],
        ),
        elapsed_ms=1000,
    )
    assert d.kind is MoveKind.PROBE_CONTRADICTION
    assert d.urgency >= 0.9


def test_budget_triggers_close():
    e = StrategyEngine(InterviewPlan(total_budget_ms=60_000, closing_tail_ms=30_000))
    d = e.decide(AnswerSignals(), elapsed_ms=40_000)
    assert d.kind is MoveKind.CLOSE


def test_deepening_follows_ladder_then_advances():
    e = StrategyEngine()
    first = e.decide(AnswerSignals(), elapsed_ms=0)
    key = first.dimension

    # 高质量回答 → 该维度深度推进
    e.observe(AnswerSignals(hit_dimensions=[key], depth_hint=4, has_quantification=True))
    d2 = e.decide(AnswerSignals(hit_dimensions=[key], depth_hint=4), elapsed_ms=1000)
    assert d2.kind is MoveKind.DEEPEN
    assert d2.dimension == key
    assert d2.depth >= 1

    # 覆盖度达标后应切换到别的维度
    for _ in range(6):
        e.observe(AnswerSignals(hit_dimensions=[key], depth_hint=4, has_quantification=True))
    d3 = e.decide(AnswerSignals(hit_dimensions=[key], depth_hint=4), elapsed_ms=2000)
    assert d3.kind in (MoveKind.ASK_MAIN, MoveKind.DEEPEN, MoveKind.CLOSE)


# --------------------------------------------------------------------------
# 评分引擎
# --------------------------------------------------------------------------


def test_dimension_score_requires_evidence():
    s = ScoringEngine()
    comp = COMPETENCY_BY_KEY["technical_depth"]
    ds = s.score_dimension(comp, [])
    assert ds.level == 0
    assert ds.confidence == 0.0
    assert ds.level_label == "未覆盖"
    assert "无法评分" in ds.rationale


def test_higher_evidence_gives_higher_score():
    s = ScoringEngine()
    comp = COMPETENCY_BY_KEY["problem_solving"]
    low = s.score_dimension(
        comp,
        [Evidence("problem_solving", "我们一起做了个系统", 0, 100, 2, "笼统")],
    )
    high = s.score_dimension(
        comp,
        [
            Evidence(
                "problem_solving",
                "我把接口延迟从800ms降到120ms，用了本地缓存加批量预热",
                0,
                100,
                4,
                "有量化结果",
            )
        ],
    )
    assert high.level > low.level
    assert high.score_100 > low.score_100
    assert high.confidence > low.confidence


def test_confidence_drops_with_contradictory_evidence():
    s = ScoringEngine()
    comp = COMPETENCY_BY_KEY["impact"]
    consistent = s.score_dimension(
        comp,
        [
            Evidence("impact", "延迟降到100ms", 0, 100, 4, ""),
            Evidence("impact", "错误率下降一半", 200, 300, 4, ""),
        ],
    )
    spread = s.score_dimension(
        comp,
        [
            Evidence("impact", "延迟降到100ms", 0, 100, 4, ""),
            Evidence("impact", "完全没解决问题", 200, 300, 1, ""),
        ],
    )
    assert spread.confidence < consistent.confidence


def test_we_ratio_detects_collective_attribution():
    text = "我们团队我们一起我们负责这个系统我们做了很多"
    assert detect_we_ratio(text) > 0.7
    assert detect_we_ratio("我自己设计的架构，我自己写的核心代码") < 0.3


def test_filler_density_detects_hesitation():
    assert detect_filler_density("嗯那个这个就是说要做的那个事情") > 5
    assert detect_filler_density("我把接口延迟从800毫秒降到了120毫秒") < 1


def test_numeric_contradiction_is_flagged():
    turns = [(0, "我们把接口延迟从200ms优化到50ms"), (60000, "最后延迟是500ms左右")]
    flags = find_contradictions(turns)
    assert len(flags) == 1
    assert flags[0].severity == "high"


def test_similar_numbers_are_not_flagged():
    """1.5 倍 vs 2 倍属于口径差异，不该报红旗，否则 HR 会忽略所有红旗。"""
    turns = [(0, "耗时1.5秒"), (60000, "耗时2秒")]
    assert find_contradictions(turns) == []


def test_report_recommendation_blocks_on_high_severity_flag():
    s = ScoringEngine()
    evidence = {
        k: [Evidence(k, "我把延迟从800ms降到120ms，稳定性也提升了", 0, 100, 4, "有量化")]
        for k in COMPETENCY_BY_KEY
    }
    weights = {c.key: c.weight for c in COMPETENCIES}
    turns = [(0, "延迟是800ms"), (1000, "延迟降到120ms"), (2000, "耗时900ms")]
    report = s.build_report(evidence, weights, turns)
    assert report.recommendation == "hold"
    assert any(f.severity == "high" for f in report.red_flags)


def test_clean_evidence_yields_advance():
    s = ScoringEngine()
    evidence = {
        k: [
            Evidence(k, "我把延迟从800ms降到120ms，错误率降到0.01%", 0, 100, 4, "有量化"),
            Evidence(k, "我主导了方案选型，放弃了同步写入改为异步队列", 200, 300, 4, "有取舍"),
        ]
        for k in COMPETENCY_BY_KEY
    }
    weights = {c.key: c.weight for c in COMPETENCIES}
    report = s.build_report(evidence, weights, [(0, "我把延迟从800ms降到120ms")])
    assert report.recommendation == "advance"
    assert report.total_score_100 >= 72
