# -*- coding: utf-8 -*-
"""评分引擎 —— 基于问诊评分量表（Rubric）的智能评分。

提供两种评分方式：
    1. 规则评分（rule_score）：基于"关键信息点覆盖度"的启发式评分，
       无需调用大模型，可离线运行，用于演示、单测与基线对比。
    2. LLM 评分（llm_score）：调用大模型按 Rubric 结构化打分，
       输出总分、分维度得分、优点、不足与改进建议（生产环境主用）。

评分量表采用百分制：问诊完整性 80 分 + 综合表现 20 分。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .case_loader import Case
from .prompt_registry import get as get_prompt
from .role_engine import chat_completion
from .synonyms import expand_term, load_synonyms

# ---------------------------------------------------------------------------
# 评分量表（Rubric）：维度名 -> 满分值
# ---------------------------------------------------------------------------
# 问诊完整性（客观信息点，规则可覆盖）——合计 80 分
RUBRIC_COMPLETENESS: dict[str, int] = {
    "开放式提问": 5,
    "主诉": 5,
    "现病史": 15,
    "既往史": 10,
    "系统回顾 / 伴随症状": 5,
    "体格检查": 8,
    "辅助检查": 5,
    "病情演变": 4,
    "一般情况": 3,
    "个人史": 8,
    "家族史": 5,
    "婚育史": 7,
}
# 综合表现（主观维度，主要靠 LLM 判定）——合计 20 分
RUBRIC_PERFORMANCE: dict[str, int] = {
    "问诊准确性": 5,
    "问诊全面性": 5,
    "医患沟通 / 人文关怀": 5,
    "问诊效率 / 条理性": 5,
}

# 中文分句/分词用的标点，用于提取病例信息点
_SPLIT_PATTERN = re.compile(r"[，。；、；\n\r\t ,;]+")

# 匹配前归一化：去除空白与常见标点（保留中文字符与数字），用于信息点覆盖度计算
_NORM_RE = re.compile(r"[\s，。；、：,.!?？:;()（）\[\]【】]+")


@dataclass
class DimensionDetail:
    """单个评分维度的明细。

    Attributes:
        score:    该维度实际得分。
        max:      该维度满分（与量表配置保持一致）。
        covered:  已覆盖的关键术语列表（学生问出的关键短语）。
        missed:   未覆盖的关键术语列表（学生尚未提及，按教学建议补问）。
    """
    score: float
    max: float
    covered: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "得分": round(self.score, 1),
            "满分": self.max,
            "已覆盖": list(self.covered),
            "建议补问": list(self.missed),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "DimensionDetail":
        if isinstance(data, dict):
            score = _safe_float(data.get("得分") or data.get("score"), 0.0)
            max_score = _safe_float(data.get("满分") or data.get("max"), 0.0)
            covered = [str(x) for x in (data.get("已覆盖") or data.get("covered") or []) if x]
            missed = [str(x) for x in (data.get("建议补问") or data.get("missed") or []) if x]
            return cls(score=score, max=max_score, covered=covered, missed=missed)
        # 兼容旧格式：纯数字
        return cls(score=_safe_float(data, 0), max=0.0)


@dataclass
class ScoringResult:
    """评分结果数据结构。

    Attributes:
        total:        总分（0-100）。
        dimensions:   分维度得分，{维度名: 得分}。
        strengths:    优点列表。
        weaknesses:   不足列表。
        suggestions:  改进建议（字符串）。
        method:       评分方式（rule / llm）。
    """
    total: float
    # 维度细节：{维度名: DimensionDetail 或 float}；兼容旧数据。
    dimensions: dict[str, Any] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    suggestions: str = ""
    method: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        """导出为可序列化的 dict。"""
        dims_out: dict[str, Any] = {}
        for k, v in self.dimensions.items():
            if isinstance(v, DimensionDetail):
                dims_out[k] = v.to_dict()
            else:
                dims_out[k] = round(float(v), 1)
        return {
            "总分": round(self.total, 1),
            "分维度得分": dims_out,
            "优点": list(self.strengths),
            "不足": list(self.weaknesses),
            "改进建议": self.suggestions,
            "评分方式": self.method,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringResult":
        """从 to_dict() 的输出反序列化（用于会话恢复等场景）。"""
        raw_dims = data.get("分维度得分") or {}
        dims: dict[str, Any] = {}
        if isinstance(raw_dims, dict):
            for k, v in raw_dims.items():
                if isinstance(v, dict):
                    dims[str(k)] = DimensionDetail.from_dict(v)
                else:
                    # 旧格式：纯数字
                    try:
                        dims[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
        return cls(
            total=_safe_float(data.get("总分"), 0.0),
            dimensions=dims,
            strengths=[str(s) for s in (data.get("优点") or [])],
            weaknesses=[str(s) for s in (data.get("不足") or [])],
            suggestions=str(data.get("改进建议", "")),
            method=str(data.get("评分方式") or "rule"),
        )

    def to_markdown(self) -> str:
        """把评分结果渲染为 Markdown 报告文本（用于导出）。"""
        lines = [
            "# 问诊评分报告",
            "",
            f"- 总分：**{round(self.total, 1)} / 100**",
            f"- 评分方式：{self.method}",
            "",
            "## 分维度得分",
            "",
        ]
        if self.dimensions:
            lines.append("| 维度 | 得分 | 已覆盖 | 建议补问 |")
            lines.append("| --- | --- | --- | --- |")
            for k, v in self.dimensions.items():
                if isinstance(v, DimensionDetail):
                    cov = "、".join(v.covered) or "—"
                    miss = "、".join(v.missed) or "—"
                    lines.append(f"| {k} | {v.score}/{v.max} | {cov} | {miss} |")
                else:
                    lines.append(f"| {k} | {v} | — | — |")
        else:
            lines.append("（无分维度得分）")
        lines.append("")
        if self.strengths:
            lines.append("## 优点")
            for s in self.strengths:
                lines.append(f"- {s}")
            lines.append("")
        if self.weaknesses:
            lines.append("## 不足")
            for w in self.weaknesses:
                lines.append(f"- {w}")
            lines.append("")
        if self.suggestions:
            lines.append("## 改进建议")
            lines.append(self.suggestions)
        return "\n".join(lines)


def _extract_info_points(text: str) -> list[str]:
    """把一段病史文本切分为若干"信息点"（按标点切分后去空）。

    用于规则评分：每个信息点代表学生应当问出的一个关键事实。
    """
    if not text:
        return []
    return [p.strip() for p in _SPLIT_PATTERN.split(text) if p.strip()]


def _normalize_text(text: str) -> str:
    """去除空白与常见标点并转小写，作为信息点匹配的归一化形式。"""
    return _NORM_RE.sub("", text or "").lower()


def _bigrams(text: str) -> set[str]:
    """提取字符二元组（2-gram），用于中文文本的模糊覆盖度匹配。

    中医文本无天然词边界，2-gram 在保留语序的同时容忍部分提及，
    避免"整段子串完全一致才算命中"导致的覆盖度过低问题。
    """
    t = _normalize_text(text)
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _coverage_with_detail(
    points: list[str], inquiry_text: str, threshold: float = 0.5
) -> tuple[float, list[str], list[str]]:
    """带命中术语明细的查询函数。

    对每个信息点，取其在学生问诊文本中命中的 2-gram 比例作为该点覆盖度；
    覆盖度 ≥ ``threshold`` 的点视为「已覆盖」，否则视为「建议补问」。
    同时为了应对口语化表达，先用同义词词典做扩展（对每个 point 在
    ``synonyms`` 字典中查同义形态，命中任一即算命中）。

    Args:
        points:       病例事实点（来自现病史/既往史/...）。
        inquiry_text: 学生问诊文本。
        threshold:    覆盖度阈值；默认 0.5。

    Returns:
        (整体覆盖度, covered 列表, missed 列表)
    """
    if not points:
        return 1.0, [], []
    inquiry_norm = _normalize_text(inquiry_text)
    inquiry_bg = _bigrams(inquiry_text)
    synonyms = load_synonyms()

    total = 0.0
    valid = 0
    covered: list[str] = []
    missed: list[str] = []

    for point in points:
        point_clean = point.strip()
        if not point_clean:
            continue
        valid += 1

        # 同义词扩展：把 point 在词典里的同义形态都算一遍
        variants = expand_term(point_clean, synonyms)
        # 用任一变体在 inquiry 中找 bigram 重叠，取最大者作为覆盖度
        best_cov = 0.0
        for variant in variants:
            v_bg = _bigrams(variant)
            if not v_bg:
                continue
            cov = len(v_bg & inquiry_bg) / len(v_bg)
            if cov > best_cov:
                best_cov = cov

        # 另外做一次「子串包含」检查：若 inquiry 中直接出现任一 variant 原词（如 "DM"），
        # 视为命中（处理纯缩写场景）
        if best_cov < threshold:
            for variant in variants:
                v_norm = _normalize_text(variant)
                if v_norm and v_norm in inquiry_norm:
                    best_cov = max(best_cov, 0.8)
                    break

        total += best_cov
        if best_cov >= threshold:
            covered.append(point_clean)
        else:
            missed.append(point_clean)

    return (total / valid if valid else 0.0), covered, missed


def _coverage(points: list[str], inquiry_text: str) -> float:
    """计算信息点覆盖度（基于字符 2-gram 的重叠比例）。

    对每个信息点，取其在学生问诊文本中命中的 2-gram 比例作为该点覆盖；
    总覆盖度 = 各信息点覆盖度的均值。该算法对口语化、部分提及的场景更
    宽容，避免"关键词需整段完全匹配"导致的低估。
    """
    cov, _, _ = _coverage_with_detail(points, inquiry_text)
    return cov


def rule_score(case: Case, inquiry_text: str) -> ScoringResult:
    """规则评分：基于病史信息点覆盖度的启发式打分。

    对每个客观病史维度（现病史/既往史/个人史/婚育史/家族史），
    用信息点覆盖度乘以该维度满分得到得分；其余维度按客观维度覆盖度同比例
    折算（避免"空问诊也白拿满分"）。同时每个维度返回「已覆盖 / 建议补问」
    的术语列表，便于学生看到扣分原因与改进方向。

    Args:
        case:         病例对象。
        inquiry_text: 学生问诊记录的拼接文本。

    Returns:
        ScoringResult（dimensions 为 ``DimensionDetail``）。
    """
    dimensions: dict[str, Any] = {}
    # 客观可判定维度 -> 对应病例文本来源（主诉 + 各病史字段）
    objective_sources = {
        "主诉": case.chief_complaint,
        "现病史": case.history.get("现病史", ""),
        "既往史": case.history.get("既往史", ""),
        "个人史": case.history.get("个人史", ""),
        "婚育史": case.history.get("婚育史", ""),
        "家族史": case.history.get("家族史", ""),
    }
    objective_full = 0.0
    objective_scored = 0.0
    for dim, full_score in RUBRIC_COMPLETENESS.items():
        if dim in objective_sources:
            points = _extract_info_points(objective_sources[dim])
            cov, covered, missed = _coverage_with_detail(points, inquiry_text)
            score = round(full_score * cov, 1)
            dimensions[dim] = DimensionDetail(
                score=score, max=full_score,
                covered=covered, missed=missed,
            )
            objective_full += full_score
            objective_scored += score

    # 过程/结构维度（开放式提问、系统回顾、体格检查、辅助检查、病情演变、
    # 一般情况等）无法靠规则可靠判定，按客观维度覆盖度同比例折算，
    # 避免"空问诊也白拿满分"的评分失真；这些维度的细化交由 LLM 评分完成。
    overall_coverage = (objective_scored / objective_full) if objective_full > 0 else 0.0
    for dim, full_score in RUBRIC_COMPLETENESS.items():
        if dim not in objective_sources:
            dimensions[dim] = DimensionDetail(
                score=round(full_score * overall_coverage, 1),
                max=full_score, covered=[], missed=[],
            )

    completeness = sum(
        d.score if isinstance(d, DimensionDetail) else float(d)
        for d in dimensions.values()
    )

    # 综合表现：规则评分按完整度折算给一个基础分
    completeness_ratio = completeness / sum(RUBRIC_COMPLETENESS.values())
    for dim, full_score in RUBRIC_PERFORMANCE.items():
        dimensions[dim] = DimensionDetail(
            score=round(full_score * completeness_ratio, 1),
            max=full_score, covered=[], missed=[],
        )

    total = completeness + sum(
        d.score if isinstance(d, DimensionDetail) else float(d)
        for d in (dimensions[d] for d in RUBRIC_PERFORMANCE)
    )
    if not inquiry_text.strip():
        suggestions = "未采集到有效问诊内容，请先对患者进行系统问诊。"
    else:
        suggestions = "请结合问诊记录，逐维度核对是否遗漏关键病史信息。"
    return ScoringResult(
        total=round(total, 1),
        dimensions=dimensions,
        strengths=["已完成问诊信息点覆盖度评估（规则基线）"],
        weaknesses=["规则评分为启发式基线，建议采用 LLM 评分获得更细致的反馈"],
        suggestions=suggestions,
        method="rule",
    )


# LLM 评分 Prompt：要求模型输出结构化 JSON
SCORING_PROMPT_TEMPLATE = """你是医学教育问诊技能考官。请依据下面的评分量表，对医学生与标准化病人（SP）的整段问诊对话进行评分。

【评分量表】
问诊完整性（80 分）：{completeness}
综合表现（20 分）：{performance}

【病例关键信息（用于判断是否问全）】
{case_facts}

【问诊对话】
{dialogue}

请仅输出一个 JSON 对象（不要输出任何解释性文字），格式如下：
{{
  "总分": 0-100 的数字,
  "分维度得分": {{"维度名": 数字, ...}},
  "优点": ["...", "..."],
  "不足": ["...", "..."],
  "改进建议": "一段 50-100 字的总结建议"
}}
"""


def build_scoring_prompt(case: Case, history: list[dict[str, str]] | None = None) -> str:
    """构建 LLM 评分的 Prompt。

    Args:
        case:    病例对象。
        history: 问诊对话历史。

    Returns:
        Prompt 字符串。
    """
    from .case_loader import case_to_facts

    dialogue = "\n".join(
        f"{'学生' if t['role'] == 'user' else 'SP'}：{t['content']}"
        for t in (history or [])
        if t.get("role") in ("user", "assistant")
    )
    template = get_prompt("scoring_system").strip() or SCORING_PROMPT_TEMPLATE
    return template.format(
        completeness=json.dumps(RUBRIC_COMPLETENESS, ensure_ascii=False),
        performance=json.dumps(RUBRIC_PERFORMANCE, ensure_ascii=False),
        case_facts=case_to_facts(case),
        dialogue=dialogue or "（无对话）",
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把任意值安全转换为 float，失败时返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_llm_score(text: str | None) -> ScoringResult:
    """解析 LLM 返回的 JSON 评分结果（容忍畸形输出，绝不抛出异常）。

    容错策略：
        1. 去除 ```json ... ``` 包裹，截取第一个 { 到最后一个 } 之间的 JSON；
        2. 非法 JSON / 非对象输出时，返回 total=0 的降级结果；
        3. 总分钳制到 [0, 100]，分维度得分逐项解析并跳过非数值项。

    Args:
        text: 模型原始输出文本。

    Returns:
        ScoringResult（method 恒为 "llm"）。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ScoringResult(
            total=0.0,
            weaknesses=["LLM 未返回评分内容"],
            suggestions="请重试或改用规则评分。",
            method="llm",
        )
    # 去除可能的代码块包裹
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
    # 截取第一个 { 到最后一个 } 之间的内容，提高容错
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end >= start:
        cleaned = cleaned[start:end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return ScoringResult(
            total=0.0,
            weaknesses=["LLM 返回非标准 JSON，无法解析"],
            suggestions="请重试或改用规则评分。",
            method="llm",
        )
    if not isinstance(data, dict):
        return ScoringResult(
            total=0.0,
            weaknesses=["LLM 返回结构异常，无法解析"],
            suggestions="请重试或改用规则评分。",
            method="llm",
        )

    total = _safe_float(data.get("总分"), 0.0)
    total = max(0.0, min(100.0, total))  # 钳制到合法区间

    dims: dict[str, Any] = {}
    raw_dims = data.get("分维度得分") or {}
    if isinstance(raw_dims, dict):
        for k, v in raw_dims.items():
            if isinstance(v, dict):
                dims[str(k)] = DimensionDetail.from_dict(v)
            else:
                # 跳过非数值项
                try:
                    dims[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue

    # 已知维度钳制到量表满分数，避免模型输出超满分或负分的异常值
    for name, max_score in {**RUBRIC_COMPLETENESS, **RUBRIC_PERFORMANCE}.items():
        if name in dims:
            dims[name] = round(max(0.0, min(max_score, dims[name])), 1)

    def _as_str_list(value: Any) -> list[str]:
        return [str(x) for x in value] if isinstance(value, list) else []

    return ScoringResult(
        total=round(total, 1),
        dimensions=dims,
        strengths=_as_str_list(data.get("优点")),
        weaknesses=_as_str_list(data.get("不足")),
        suggestions=str(data.get("改进建议", "")),
        method="llm",
    )


# LLM 评分可选的"分维度详情"字段名（兼容多版本 prompt 输出）
_LLM_DETAIL_FIELDS = ("分维度详情", "dimension_details", "details")


def llm_score(
    client: Any,
    case: Case,
    history: list[dict[str, str]] | None = None,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    json_mode: bool = True,
) -> ScoringResult:
    """调用大模型进行 Rubric 结构化评分。

    Args:
        client:      OpenAI 兼容客户端。
        case:        病例对象。
        history:     问诊对话历史。
        model:       模型标识。
        temperature: 采样温度，评分用更低的 0.1 保证可复现。
        json_mode:   是否强制 JSON 输出；部分提供商不支持 response_format，
                     失败时自动降级为不带该参数重试。

    Returns:
        ScoringResult。
    """
    from .config import DEFAULT_MODEL

    prompt = build_scoring_prompt(case, history)
    kwargs: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严谨的医学问诊技能评分助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}  # 强制 JSON（模型需支持）

    try:
        resp = chat_completion(client, **kwargs)
    except Exception:
        # 某些模型/提供商不支持 response_format，降级为普通文本请求重试一次
        if not json_mode:
            raise
        kwargs.pop("response_format", None)
        resp = chat_completion(client, **kwargs)

    content = resp.choices[0].message.content or "{}"
    return parse_llm_score(content)
