# -*- coding: utf-8 -*-
"""SOAP 笔记启发式评分。

对学生填写的 4 个 SOAP 文本（主观/客观/评估/计划）做信息点覆盖度评分，
覆盖度 = 2-gram 重叠比例（与 :func:`core.scoring_engine._coverage` 同算法）。

每个维度满分 25，总分 100。

注：临床病例的"体格检查"与"辅助检查"字段一般为空白，因此：
    * O 维度：按「客观检查关键词 + 文本长度」启发式给分；
    * P 维度：按「处置/随访关键词 + 文本长度」启发式给分。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .case_loader import Case
from .scoring_engine import DimensionDetail, _coverage_with_detail

# O/P 维度没有标准答案时的关键词清单：用于区分「写了检查/处置要点」与
# 纯凑字数的填充文本。关键词只作为启发式信号，不追求覆盖全部医学表达。
_OBJECTIVE_KEYWORDS = (
    "体格检查", "查体", "压痛", "反跳痛", "肠鸣音", "血压", "心率", "体温",
    "血常规", "白细胞", "中性粒", "B超", "超声", "CT", "X线", "化验",
    "检查", "叩诊", "听诊", "触诊", "视诊",
)
_PLAN_KEYWORDS = (
    "检查", "治疗", "用药", "药物", "复查", "复诊", "随访", "健康教育",
    "休息", "饮食", "转诊", "住院", "手术", "抑酸", "抗生素", "补液",
    "胃镜", "观察",
)


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    """判断文本是否包含任一关键词。"""
    return any(keyword in text for keyword in keywords)


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


def _score_objective(text: str) -> float:
    """O 维度评分：有客观检查关键词且内容较完整才给高分。"""
    text = text.strip()
    if not text:
        return 0.0
    length = len(text)
    if _has_any(text, _OBJECTIVE_KEYWORDS):
        if length >= 50:
            return 25.0
        if length >= 20:
            return 20.0
        return 15.0
    # 没有检查关键词时只给很低的「诚实分」，避免纯填充文本拿满分
    if length >= 50:
        return 10.0
    if length >= 20:
        return 8.0
    return 5.0


def _score_plan(text: str) -> float:
    """P 维度评分：有处置/随访关键词且内容较完整才给高分。"""
    text = text.strip()
    if not text:
        return 0.0
    length = len(text)
    if _has_any(text, _PLAN_KEYWORDS):
        if length >= 60:
            return 25.0
        if length >= 30:
            return 18.0
        return 12.0
    if length >= 60:
        return 10.0
    if length >= 30:
        return 8.0
    return 5.0


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

    # O 维度：病例参考文本通常缺失，按检查关键词 + 长度启发式给分
    o_score = _score_objective(objective)
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

    # P 维度：按处置/随访关键词 + 长度启发式给分
    p_score = _score_plan(plan)
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
