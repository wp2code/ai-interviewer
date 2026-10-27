"""评分引擎：证据锚定 + 档位映射 + 一致性红旗检测。

**核心立场：不产出黑箱分数，产出可复核的证据链。**

一个评分必须能回答三个问题：
1. 依据哪句原话？（证据 quote + 时间戳，可回听）
2. 对应 rubric 的哪一档？（behavioral anchor 引用）
3. 这个分数有多确信？（置信度，且置信度本身有来源）

如果答不上这三点，HR 一旦质疑就崩盘 —— 而且用 AI 做初筛最怕的就是这个。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .rubric import COMPETENCY_BY_KEY, Competency, RubricLevel


@dataclass
class Evidence:
    """一条可回溯的证据。"""

    dimension: str
    quote: str
    ts_start_ms: int
    ts_end_ms: int
    level_hint: int  # 1~5
    rationale: str = ""
    source: str = "llm"  # llm | rule | human

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "quote": self.quote,
            "ts_start_ms": self.ts_start_ms,
            "ts_end_ms": self.ts_end_ms,
            "level_hint": self.level_hint,
            "rationale": self.rationale,
            "source": self.source,
        }


@dataclass
class RedFlag:
    code: str
    severity: str  # low | medium | high
    title: str
    detail: str
    ts_start_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "ts_start_ms": self.ts_start_ms,
        }


@dataclass
class DimensionScore:
    dimension: str
    dimension_name: str
    level: int
    level_label: str
    anchor: str
    score_100: float
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "dimension_name": self.dimension_name,
            "level": self.level,
            "level_label": self.level_label,
            "anchor": self.anchor,
            "score_100": round(self.score_100, 1),
            "confidence": round(self.confidence, 3),
            "evidence": [e.to_dict() for e in self.evidence],
            "rationale": self.rationale,
        }


@dataclass
class InterviewReport:
    dimensions: list[DimensionScore]
    total_score_100: float
    overall_confidence: float
    red_flags: list[RedFlag]
    recommendation: str  # advance | hold | reject
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "total_score_100": round(self.total_score_100, 1),
            "overall_confidence": round(self.overall_confidence, 3),
            "red_flags": [f.to_dict() for f in self.red_flags],
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
        }


# --------------------------------------------------------------------------
# 一致性与红旗检测（确定性规则，不依赖 LLM）
# --------------------------------------------------------------------------

_FIRST_PERSON = re.compile(r"我(?!们)|本人")
_COLLECTIVE = re.compile(r"(我们|咱们|团队|小组|部门|他们)")
_FILLERS = ("嗯", "呃", "那个", "这个", "就是说", "然后呢", "怎么说呢")

# 数字型指标关键词，用于矛盾检测
_METRIC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(%|ms|毫秒|秒|分钟|小时|天|万|亿|k|K|w|W|倍|人|次)")

# 指标词典：数字前若出现这些词，则归为同一指标，可跨轮次比对
_METRIC_LEXICON = (
    "延迟",
    "耗时",
    "响应时间",
    "响应",
    "性能",
    "成功率",
    "错误率",
    "失败率",
    "转化率",
    "成本",
    "营收",
    "利润",
    "人数",
    "流量",
    "并发",
    "请求数",
    "内存",
    "cpu",
    "CPU",
    "磁盘",
    "带宽",
    "时长",
    "占用",
)


def detect_we_ratio(text: str) -> float:
    """集体人称占比。过高说明个人贡献边界不清 —— 团队协作题里的经典注水信号。"""
    first = len(_FIRST_PERSON.findall(text))
    collective = len(_COLLECTIVE.findall(text))
    total = first + collective
    if total == 0:
        return 0.0
    return collective / total


def detect_filler_density(text: str) -> float:
    """口头禅密度（每百字命中数）。"""
    if not text:
        return 0.0
    hits = sum(text.count(f) for f in _FILLERS)
    return hits * 100.0 / max(len(text), 1)


def extract_metrics(text: str) -> list[tuple[str, float]]:
    """抽取 (指标名, 数值)。

    指标名从数字**前方的词典**中回溯识别，而不是简单地取前一个词。
    否则"接口延迟从200ms"与"延迟是500ms"会被当成两个不同指标，矛盾检测就失效了。
    找不到词典命中时，退化为"未知指标+单位"，同单位之间仍可比较。
    """
    out: list[tuple[str, float]] = []
    for m in _METRIC_RE.finditer(text):
        try:
            value = float(m.group(1))
        except ValueError:
            continue
        unit = m.group(2)
        prefix = text[max(0, m.start() - 16) : m.start()]
        name = ""
        best_pos = -1
        for kw in _METRIC_LEXICON:
            pos = prefix.rfind(kw)
            if pos > best_pos:
                best_pos = pos
                name = kw
        if not name:
            name = f"未知指标_{unit}"
        out.append((name, value))
    return out


def find_contradictions(turn_texts: list[tuple[int, str]]) -> list[RedFlag]:
    """同一指标在不同轮次出现显著不同的数值 → 前后矛盾。

    阈值取 2 倍差距：1.5 倍和 4 倍是量级差异，2 倍与 2.1 倍往往只是口径差异，
    不该报，否则红旗噪音会让 HR 忽略所有红旗。
    """
    flags: list[RedFlag] = []
    buckets: dict[str, list[tuple[int, float]]] = {}
    for idx, (_ts, text) in enumerate(turn_texts):
        for name, value in extract_metrics(text):
            buckets.setdefault(name, []).append((idx, value))

    for name, items in buckets.items():
        if len(items) < 2:
            continue
        # 同一指标在同一轮内出现多次（如"从200ms优化到50ms"）不算矛盾，
        # 只比较不同轮次之间的取值
        by_turn: dict[int, list[float]] = {}
        for turn_idx, value in items:
            by_turn.setdefault(turn_idx, []).append(value)
        cross = [(i, v[-1]) for i, v in sorted(by_turn.items())]
        if len(cross) < 2:
            continue
        values = [v for _, v in cross]
        lo, hi = min(values), max(values)
        if lo <= 0:
            continue
        if hi / lo >= 2.0:
            first_i, first_v = cross[0]
            last_i, last_v = cross[-1]
            flags.append(
                RedFlag(
                    code="numeric_contradiction",
                    severity="high",
                    title=f"「{name}」指标前后不一致",
                    detail=(
                        f"第 {first_i + 1} 轮提到 {first_v:g}，"
                        f"第 {last_i + 1} 轮提到 {last_v:g}，相差 {hi / lo:.1f} 倍。"
                        "请人工核实是否为统计口径差异。"
                    ),
                    ts_start_ms=turn_texts[first_i][0],
                )
            )
    return flags


def detect_evidence_quality_flags(
    turn_texts: list[tuple[int, str]], min_len: int = 40
) -> list[RedFlag]:
    flags: list[RedFlag] = []
    for idx, (ts, text) in enumerate(turn_texts):
        if len(text) < min_len and len(text) > 0:
            flags.append(
                RedFlag(
                    code="thin_answer",
                    severity="low",
                    title="回答过短",
                    detail=f"第 {idx + 1} 轮仅 {len(text)} 字，可能存在准备不足或沟通意愿低。",
                    ts_start_ms=ts,
                )
            )
        if detect_filler_density(text) > 3.0:
            flags.append(
                RedFlag(
                    code="filler_heavy",
                    severity="low",
                    title="口头禅密集",
                    detail=f"第 {idx + 1} 轮口头禅密度 {detect_filler_density(text):.1f}/百字。",
                    ts_start_ms=ts,
                )
            )
    return flags


# --------------------------------------------------------------------------
# 评分
# --------------------------------------------------------------------------


def _evidence_quality(evidence: Evidence) -> float:
    """单条证据的信息量评分（0~1）。

    三个维度：是否有量化数字、是否有具体行为动词、长度是否足够展开。
    这三项决定了这条证据能不能支撑一个档位判断。
    """
    text = evidence.quote or ""
    score = 0.0
    if re.search(r"\d", text):
        score += 0.4
    if re.search(
        r"(实现|优化|重构|排查|设计|推动|主导|上线|解决|定位|落地|开发|引入|搭建|改进|降低|提升)",
        text,
    ):
        score += 0.35
    if len(text) >= 20:
        score += 0.25
    return min(1.0, score)


class ScoringEngine:
    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        # 阈值：总分进入下一档的分数线，以及 advance / hold 的分界
        self.advance_threshold = (thresholds or {}).get("advance", 72.0)
        self.hold_threshold = (thresholds or {}).get("hold", 60.0)

    def score_dimension(
        self, competency: Competency, evidence: list[Evidence]
    ) -> DimensionScore:
        if not evidence:
            return DimensionScore(
                dimension=competency.key,
                dimension_name=competency.name,
                level=0,
                level_label="未覆盖",
                anchor="",
                score_100=0.0,
                confidence=0.0,
                evidence=[],
                rationale="该维度未采集到有效证据，无法评分。",
            )

        # 加权：来源可信度 + 理由完整度
        def w(e: Evidence) -> float:
            base = 1.0 if e.source == "human" else 0.8
            if e.rationale:
                base += 0.1
            if e.quote:
                base += 0.1
            return base

        total_w = sum(w(e) for e in evidence)
        level = sum(e.level_hint * w(e) for e in evidence) / total_w

        # 取最接近的档位作为展示档位
        nearest = min(competency.levels, key=lambda lv: abs(lv.level - level))

        # 置信度：证据条数 + 档位一致性 + 单条证据的信息量
        n = len(evidence)
        coverage_conf = min(1.0, n / 2.0)
        spread = max(e.level_hint for e in evidence) - min(
            e.level_hint for e in evidence
        )
        consistency_conf = 1.0 if spread <= 1 else max(0.0, 1.0 - 0.35 * (spread - 1))

        # 单条证据的信息量：空洞的一句话不该和有量化细节的回答同等可信
        avg_quality = sum(_evidence_quality(e) for e in evidence) / n
        confidence = round(
            0.25 + 0.30 * coverage_conf + 0.20 * consistency_conf + 0.25 * avg_quality,
            3,
        )

        score_100 = round(nearest.level / 5.0 * 100.0, 1)
        top = max(evidence, key=lambda e: e.level_hint)
        rationale = (
            f"共 {n} 条证据，主要依据：{top.quote[:40]}"
            f"{'…' if len(top.quote) > 40 else ''}"
            f"（{top.rationale or '命中该档行为锚点'}）"
        )
        return DimensionScore(
            dimension=competency.key,
            dimension_name=competency.name,
            level=nearest.level,
            level_label=nearest.label,
            anchor=nearest.anchor,
            score_100=score_100,
            confidence=confidence,
            evidence=evidence,
            rationale=rationale,
        )

    def build_report(
        self,
        evidence_by_dimension: dict[str, list[Evidence]],
        weights: dict[str, float],
        turn_texts: list[tuple[int, str]],
    ) -> InterviewReport:
        dims: list[DimensionScore] = []
        for key, comp in COMPETENCY_BY_KEY.items():
            dims.append(
                self.score_dimension(comp, evidence_by_dimension.get(key, []))
            )

        scored = [d for d in dims if d.level > 0]
        if scored:
            wsum = sum(weights.get(d.dimension, 0.0) for d in scored)
            total = (
                sum(
                    d.score_100 * weights.get(d.dimension, 0.0) for d in scored
                )
                / wsum
                if wsum
                else 0.0
            )
            overall_conf = round(
                sum(d.confidence * weights.get(d.dimension, 0.0) for d in scored)
                / wsum
                if wsum
                else 0.0,
                3,
            )
        else:
            total, overall_conf = 0.0, 0.0

        flags: list[RedFlag] = []
        flags.extend(find_contradictions(turn_texts))
        flags.extend(detect_evidence_quality_flags(turn_texts))

        # 高危红旗直接压低总分与推荐档位
        high_flags = [f for f in flags if f.severity == "high"]
        if high_flags:
            total = round(max(0.0, total - 6.0 * len(high_flags)), 1)
            overall_conf = round(max(0.0, overall_conf - 0.15 * len(high_flags)), 3)

        if total >= self.advance_threshold and not high_flags:
            rec, reason = "advance", (
                f"综合 {total:.1f} 分，置信度 {overall_conf:.2f}，"
                "各维度均有证据支撑，建议进入下一轮。"
            )
        elif total >= self.hold_threshold:
            rec, reason = "hold", (
                f"综合 {total:.1f} 分，存在证据不足或置信度偏低的维度，"
                "建议人工复核或补充追问。"
            )
        else:
            rec, reason = "reject", (
                f"综合 {total:.1f} 分，低于 {self.hold_threshold:.0f} 分线，"
                "且未达到岗位门槛。"
            )

        return InterviewReport(
            dimensions=dims,
            total_score_100=total,
            overall_confidence=overall_conf,
            red_flags=flags,
            recommendation=rec,
            recommendation_reason=reason,
        )


def level_for_hint(comp: Competency, hint: int) -> RubricLevel:
    """把 1~5 的提示分映射到最近的档位定义。"""
    return min(comp.levels, key=lambda lv: abs(lv.level - hint))
