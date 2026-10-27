"""能力模型、评分标准与题库。

这是整套系统的"客观基准"。三个层次：

- **Competency（能力维度）**：面试要考察什么，权重多少。
- **RubricLevel（评分档位）**：1~5 档，每档有行为锚点（behavioral anchor），
  让评分从"感觉"变成"对照档位找证据"。
- **Probe（追问阶梯）**：同一维度的深挖路径，depth 越大越逼近真实水平。

题库不写死具体题目，而是写"提问意图 + 参数"，由 LLM 适配候选人的简历与回答语境生成，
避免所有候选人听到一模一样的题目（模板化面试本身就是信度杀手）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RubricLevel:
    level: int
    label: str
    descriptor: str
    anchor: str  # 行为锚点：出现什么证据才落到这一档


@dataclass(frozen=True)
class Competency:
    key: str
    name: str
    weight: float
    levels: tuple[RubricLevel, ...]
    probes: tuple[str, ...]  # 追问阶梯，按 depth 递进
    target_coverage: float = 1.0


DEFAULT_RUBRIC_LEVELS = (
    RubricLevel(
        level=1,
        label="不合格",
        descriptor="回答空泛、无法支撑岗位要求",
        anchor="仅有结论无过程，或明显与岗位能力不匹配；关键问题答非所问",
    ),
    RubricLevel(
        level=2,
        label="待改进",
        descriptor="知道概念但缺乏落地经验",
        anchor="能复述流程或术语，追问一层就露底；无具体数据、无个人动作痕迹",
    ),
    RubricLevel(
        level=3,
        label="合格",
        descriptor="能独立完成岗位常规工作",
        anchor="有清晰的个人动作与结果，能说清关键步骤；细节尚浅但经得起两轮追问",
    ),
    RubricLevel(
        level=4,
        label="优秀",
        descriptor="能处理复杂场景并做出技术判断",
        anchor="有量化结果、明确取舍逻辑、说清失败与权衡；主动暴露风险点",
    ),
    RubricLevel(
        level=5,
        label="卓越",
        descriptor="定义问题、影响他人、能带团队",
        anchor="不仅解决自己问题，还能指出体系性缺陷并给出系统性方案；有影响力证据",
    ),
)


def _build_competencies() -> tuple[Competency, ...]:
    return (
        Competency(
            key="problem_solving",
            name="问题解决",
            weight=0.28,
            levels=DEFAULT_RUBRIC_LEVELS,
            probes=(
                "请讲一个你主导解决的技术难题，背景和难点是什么？",
                "这件事里具体是你做了哪部分？其他人分别负责什么？",
                "你最终选的方案是什么？当时还有哪些备选，为什么放弃了其他方案？",
                "现在回头看，这个决策里最可能出错的地方是哪里？如果重来一次你会怎么改？",
            ),
        ),
        Competency(
            key="technical_depth",
            name="技术深度",
            weight=0.24,
            levels=DEFAULT_RUBRIC_LEVELS,
            probes=(
                "挑一个你最有把握的技术方向，讲讲你掌握到什么深度。",
                "这个方向的底层原理你能讲到什么程度？举一个你踩过的坑。",
                "线上出问题时，你的排查路径是什么？第一步做什么，依据是什么？",
                "如果让你把这套方案的性能提升一倍，你会从哪里下手？",
            ),
        ),
        Competency(
            key="impact",
            name="业务影响",
            weight=0.20,
            levels=DEFAULT_RUBRIC_LEVELS,
            probes=(
                "你做过的项目里，哪一个对业务结果影响最大？怎么衡量的？",
                "这个结果里，多少是你的改动带来的？你怎么排除其他因素？",
                "如果这个项目的指标没达成，你会认为问题出在哪？",
                "你会怎么向非技术的业务方解释这项工作的价值？",
            ),
        ),
        Competency(
            key="collaboration",
            name="协作与影响力",
            weight=0.16,
            levels=DEFAULT_RUBRIC_LEVELS,
            probes=(
                "讲一次你和同事或上级意见不一致的经历，最后怎么处理的？",
                "当时对方的理由是什么？你有没有理解它？",
                "你是怎么说服他的？用了什么具体依据？",
                "如果是对方是对的，你会怎么收场？",
            ),
        ),
        Competency(
            key="learning",
            name="学习与成长",
            weight=0.12,
            levels=DEFAULT_RUBRIC_LEVELS,
            probes=(
                "最近半年你学到的最有价值的东西是什么？是怎么学的？",
                "这个学习在后来的工作里真的用上了吗？举个例子。",
                "你是怎么判断自己已经掌握了？",
                "你给自己定过什么阶段性目标？完成情况如何？",
            ),
        ),
    )


COMPETENCIES: tuple[Competency, ...] = _build_competencies()

COMPETENCY_BY_KEY = {c.key: c for c in COMPETENCIES}

# 面试中固定要覆盖的"门槛维度"，无论表现如何都必须问到，用于识别硬伤
GATEWAY_COMPETENCIES = ("problem_solving", "technical_depth")


@dataclass
class InterviewPlan:
    """面试蓝图：总时长、覆盖策略、结束条件。"""

    total_budget_ms: int = 25 * 60_000
    closing_tail_ms: int = 90_000
    max_probes_per_competency: int = 3
    coverage_floor: float = 0.55  # 低于此覆盖度的维度不能提前放过
    weights_override: dict[str, float] = field(default_factory=dict)

    def weight_of(self, competency: Competency) -> float:
        return self.weights_override.get(competency.key, competency.weight)
