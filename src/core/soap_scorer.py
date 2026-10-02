# -*- coding: utf-8 -*-
"""SOAP 笔记启发式评分。

对学生填写的 4 个 SOAP 文本（主观/客观/评估/计划）做信息点覆盖度评分，
覆盖度 = 2-gram 重叠比例（与 :func:`core.scoring_engine._coverage` 同算法）。

每个维度满分 25，总分 100。

注：临床产品 case 的"体格检查"与"辅助检查"字段一般为空白，因此：
    * O 维度：若学生未填且病例无相关分文本，则按"诚实分（80%）"给基础分；
    * P 维度：若学生给出了任何有意义的处置/建议（>=10 个字符）即得高分。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .case_loader import Case
from .scoring_engine import DimensionDetail, _coverage_with_detail


@dataclass
class SOAPScoreResult:
    """SOAP 评分结果。"""
    total: float
    dimensions: dict[str, DimensionDetail]
    method: str = "rule"

    def to_markdown(self) -> str:
        lines = [
            "# SOAP 笔记评分",
            "",
            f"- 总分：**{round(self.total, 1)} / 100**",
            f"- 评分方式：{self.method}",
            "",
            "| 维度 | 得分 | 已覆盖 | 建议补问 |",
            "| --- | --- | --- | --- |",
        ]
        for k, d in self.dimensions.items():
            cov = "、".join(d.covered) or "—"
            miss = "、".join(d.missed) or "—"
            lines.append(f"| {k} | {d.score}/{d.max} | {cov} | {miss} |")
        return "\n".join(lines)


def _dim_coverage(
    student_text: str, reference_text: str, threshold: float = 0.3
) -> tuple[float, list[str], list[str]]:
    """对单个维度评分。

    Args:
        student_text:   学生填写的笔记文本。
        reference_text: 拼接好的参考文本（多行）。
        threshold:     覆盖度阈值（默认 0.3，比问诊评分略宽松）。

    Returns:
        (cover 比例, 已覆盖术语列表, 建议补问术语列表)
    """
    # 把 reference 文本按句号/分号/换行切成 4-12 个关键事实点（避免按字符切分太碎）
    parts = re.split(r"[。;\n]+", reference_text)
    facts = [p.strip() for p in parts if len(p.strip()) >= 4]
    if not facts:
        facts = [reference_text.strip()]
    return _coverage_with_detail(facts, student_text or "", threshold=threshold)


def score_soap(
    case: Case,
    subjective: str = "",
    objective: str = "",
    assessment: str = "",
    plan: str = "",
) -> SOAPScoreResult:
    """对学生 SOAP 笔记做 4 维启发式评分。

    Args:
        case:        病例对象（用于取参考字段）。
        subjective:  学生填写的"主观资料"文本。
        objective:   学生填写的"客观资料"文本。
        assessment:  学生填写的"评估诊断"文本。
        plan:        学生填写的"处理计划"文本。

    Returns:
        SOAPScoreResult。
    """
    # S 维度：参考 case 主诉+现病史+既往史+个人史+婚育史+家族史
    s_refs = [str(case.chief_complaint)]
    for f in ("现病史", "既往史", "个人史", "婚育史", "家族史"):
        v = case.history.get(f, "")
        if v:
            s_refs.append(str(v))
    s_ref_text = "\n".join(s_refs)
    s_cov, s_covered, s_missed = _dim_coverage(subjective, s_ref_text)

    # O 维度：参考文本通常缺失，给予基础分并按学生填写长度加分
    o_score = 0.0
    if objective.strip():
        # 诚实分：填了任何内容都给 60%；长且有条理加到 80%
        o_score = 15.0
        if len(objective.strip()) >= 20:
            o_score = 20.0
        if len(objective.strip()) >= 50:
            o_score = 25.0
    o_detail = DimensionDetail(score=o_score, max=25.0, covered=[], missed=[])

    # A 维度：参考病例诊断
    a_ref = case.diagnosis or ""
    if a_ref:
        a_cov, a_covered, a_missed = _dim_coverage(assessment, a_ref)
        a_score = round(25.0 * a_cov, 1)
        a_detail = DimensionDetail(score=a_score, max=25.0, covered=a_covered, missed=a_missed)
    else:
        a_score = 0.0
        a_detail = DimensionDetail(score=a_score, max=25.0, covered=[], missed=[])

    # P 维度：按是否给出有结构的处置文本给分
    p_score = 0.0
    if plan.strip():
        p_score = 10.0
        if len(plan.strip()) >= 30:
            p_score = 18.0
        if len(plan.strip()) >= 60:
            p_score = 25.0
    p_detail = DimensionDetail(score=p_score, max=25.0, covered=[], missed=[])

    s_detail = DimensionDetail(
        score=round(25.0 * s_cov, 1),
        max=25.0, covered=s_covered, missed=s_missed,
    )

    total = s_detail.score + o_score + a_score + p_score
    return SOAPScoreResult(
        total=round(total, 1),
        dimensions={
            "S - 主观资料": s_detail,
            "O - 客观资料": o_detail,
            "A - 评估诊断": a_detail,
            "P - 处理计划": p_detail,
        },
    )
