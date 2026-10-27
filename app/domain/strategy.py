"""追问策略引擎：决定"下一句问什么"。

**架构决策：LLM 只做信号抽取，策略决策用确定性规则。**

为什么不直接让 LLM 输出"下一个问题"？
- 不可测：没法写单测，只能人工体验，回归全靠感觉
- 不可复盘：同一场面试重跑两次，问的题不一样，没法定位问题
- 不可审计：HR 问"为什么问我这个"，答不上来

所以拆成两层：
1. **抽取层（LLM）**：从候选人回答里抽出结构化信号 —— 命中哪些维度、
   回答深度如何、是否自相矛盾、是否只是复述题干。
2. **决策层（纯函数）**：基于信号 + 覆盖度 + 剩余时间，算出下一步动作。

决策层是纯函数，可以穷举测试、可以回放、可以给审计看。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .rubric import (
    COMPETENCIES,
    COMPETENCY_BY_KEY,
    GATEWAY_COMPETENCIES,
    Competency,
    InterviewPlan,
)


class MoveKind(str, Enum):
    ASK_MAIN = "ask_main"      # 首次切入某维度
    DEEPEN = "deepen"          # 当前维度继续深挖
    PROBE_CONTRADICTION = "probe_contradiction"  # 钻矛盾点
    ADVANCE = "advance"        # 切到下一个维度
    CLOSE = "close"            # 进入收尾


@dataclass
class AnswerSignals:
    """LLM 从候选人回答中抽出的信号。"""

    hit_dimensions: list[str] = field(default_factory=list)
    depth_hint: int = 1  # 1=空洞 2=有过程 3=有细节 4=有取舍与量化 5=有系统性思考
    is_restatement: bool = False  # 只是复述题干
    is_evasive: bool = False      # 回避问题
    contradicts_previous: bool = False
    contradiction_detail: str = ""
    has_quantification: bool = False
    answer_chars: int = 0
    summary: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AnswerSignals":
        return cls(
            hit_dimensions=list(d.get("hit_dimensions") or []),
            depth_hint=int(d.get("depth_hint", 1)),
            is_restatement=bool(d.get("is_restatement")),
            is_evasive=bool(d.get("is_evasive")),
            contradicts_previous=bool(d.get("contradicts_previous")),
            contradiction_detail=str(d.get("contradiction_detail") or ""),
            has_quantification=bool(d.get("has_quantification")),
            answer_chars=int(d.get("answer_chars", 0)),
            summary=str(d.get("summary") or ""),
        )


@dataclass
class CoverageState:
    """各维度的覆盖情况。"""

    depth_reached: dict[str, int] = field(default_factory=dict)
    evidence_count: dict[str, int] = field(default_factory=dict)

    def depth_of(self, key: str) -> int:
        return self.depth_reached.get(key, 0)

    def evidence_of(self, key: str) -> int:
        return self.evidence_count.get(key, 0)

    def coverage(self, comp: Competency) -> float:
        """覆盖度 = 深度达成度与证据条数的加权。"""
        depth_ratio = min(1.0, self.depth_of(comp.key) / max(1, comp.target_coverage))
        ev_ratio = min(1.0, self.evidence_of(comp.key) / 2.0)
        return round(0.6 * depth_ratio + 0.4 * ev_ratio, 3)

    def bump(self, key: str, evidence_count: int) -> None:
        self.depth_reached[key] = self.depth_of(key) + 1
        self.evidence_count[key] = evidence_count


@dataclass
class Decision:
    kind: MoveKind
    dimension: str = ""
    depth: int = 0
    prompt_intent: str = ""   # 给 LLM 的提问意图，不是最终话术
    reason: str = ""          # 决策依据，用于审计
    urgency: float = 0.0      # 0~1，越高越优先

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "dimension": self.dimension,
            "depth": self.depth,
            "prompt_intent": self.prompt_intent,
            "reason": self.reason,
            "urgency": self.urgency,
        }


class StrategyEngine:
    """确定性决策层。"""

    def __init__(self, plan: InterviewPlan | None = None) -> None:
        self.plan = plan or InterviewPlan()
        self.coverage = CoverageState()
        self.current_dimension: str | None = None
        self._turn_index = 0

    # -- 状态更新 -------------------------------------------------------
    def observe(self, signals: AnswerSignals) -> None:
        """把一轮回答的信号吸收进覆盖状态。"""
        self._turn_index += 1
        dims = signals.hit_dimensions or ([self.current_dimension] if self.current_dimension else [])
        for key in dims:
            if key not in COMPETENCY_BY_KEY:
                continue
            if signals.depth_hint >= 3 or signals.has_quantification:
                self.coverage.bump(key, self.coverage.evidence_of(key) + 1)
            elif self.coverage.depth_of(key) == 0:
                self.coverage.bump(key, 1)

    # -- 决策 ------------------------------------------------------------
    def decide(
        self,
        signals: AnswerSignals | None = None,
        elapsed_ms: int = 0,
    ) -> Decision:
        plan = self.plan

        # 1) 时间预算 → 收尾。放最前面，这是硬约束。
        remaining = plan.total_budget_ms - elapsed_ms
        if remaining <= plan.closing_tail_ms:
            return Decision(
                kind=MoveKind.CLOSE,
                prompt_intent="收尾：请候选人用一句话总结最匹配岗位的一项能力",
                reason=f"剩余 {remaining}ms，进入收尾窗口",
                urgency=1.0,
            )

        # 2) 矛盾优先钻：矛盾是最高信息密度的追问点
        if signals and signals.contradicts_previous:
            dim = (signals.hit_dimensions or [self.current_dimension or ""])[0]
            return Decision(
                kind=MoveKind.PROBE_CONTRADICTION,
                dimension=dim,
                depth=self.coverage.depth_of(dim),
                prompt_intent=(
                    "请候选人澄清矛盾点：" + signals.contradiction_detail
                    + "。用中性语气，以澄清事实为目标，不指责。"
                ),
                reason="检测到与前文矛盾，优先澄清",
                urgency=0.95,
            )

        # 3) 空洞/回避回答 → 同维度再问一次，换角度而非加深度
        if signals and (signals.is_evasive or signals.is_restatement):
            dim = self.current_dimension or COMPETENCIES[0].key
            if self.coverage.depth_of(dim) < plan.max_probes_per_competency:
                return Decision(
                    kind=MoveKind.DEEPEN,
                    dimension=dim,
                    depth=self.coverage.depth_of(dim),
                    prompt_intent=(
                        f"候选人回答空洞或复述题干。改用更具体的切入方式重问同一维度："
                        f"要求给出一个具体的项目名、时间点和做过的一个动作。"
                    ),
                    reason="回答信号为空洞/复述，换角度追问",
                    urgency=0.8,
                )

        # 4) 当前维度还能深挖 → 深挖
        if self.current_dimension:
            comp = COMPETENCY_BY_KEY[self.current_dimension]
            depth = self.coverage.depth_of(comp.key)
            covered_enough = self.coverage.coverage(comp) >= comp.target_coverage
            if (
                not covered_enough
                and depth < plan.max_probes_per_competency
                and depth < len(comp.probes)
            ):
                return Decision(
                    kind=MoveKind.DEEPEN,
                    dimension=comp.key,
                    depth=depth,
                    prompt_intent=self._probe_intent(comp, depth),
                    reason=(
                        f"{comp.name} 覆盖度 {self.coverage.coverage(comp):.2f} "
                        f"未达 {comp.target_coverage}，深度 {depth}"
                    ),
                    urgency=0.6,
                )

        # 5) 选下一个维度：门槛维度优先，其余按"权重缺口"排序
        target = self._pick_next_dimension()
        if target is None:
            return Decision(
                kind=MoveKind.CLOSE,
                prompt_intent="收尾",
                reason="所有维度覆盖度达标",
                urgency=0.9,
            )
        self.current_dimension = target.key
        return Decision(
            kind=MoveKind.ASK_MAIN,
            dimension=target.key,
            depth=0,
            prompt_intent=self._probe_intent(target, 0),
            reason=f"切入维度 {target.name}（权重 {target.weight}）",
            urgency=0.5,
        )

    # -- 辅助 ------------------------------------------------------------
    def _pick_next_dimension(self) -> Competency | None:
        """选择下一个维度。

        排序依据 = 权重 × (1 - 覆盖度)。权重高且没覆盖到的维度优先，
        门槛维度额外加权 —— 无论表现如何都要问出来。
        """
        candidates: list[tuple[float, Competency]] = []
        for comp in COMPETENCIES:
            if self.coverage.depth_of(comp.key) >= self.plan.max_probes_per_competency:
                continue
            if self.coverage.coverage(comp) >= comp.target_coverage:
                continue
            weight = self.plan.weight_of(comp)
            gap = weight * (1.0 - self.coverage.coverage(comp))
            if comp.key in GATEWAY_COMPETENCIES:
                gap *= 1.5  # 门槛维度加权
            candidates.append((gap, comp))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _probe_intent(self, comp: Competency, depth: int) -> str:
        idx = min(depth, len(comp.probes) - 1)
        base = comp.probes[idx]
        extra = ""
        if depth == 0:
            extra = "（开场提问，简短直接，给候选人展开空间）"
        elif depth == 1:
            extra = "（聚焦个人贡献，追问具体动作）"
        elif depth == 2:
            extra = "（追问决策依据与取舍，要求具体）"
        else:
            extra = "（压力/反思类问题，考察判断与认知边界）"
        return f"考察维度【{comp.name}】。{base}{extra}"
