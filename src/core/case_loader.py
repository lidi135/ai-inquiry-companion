# -*- coding: utf-8 -*-
"""病例库加载与校验。

病例以结构化 JSON 文件存储，一个文件一个病例。每个病例包含：
    case_id      病例编号（如 "P01"）
    主诉          患者主诉
    评分属性       评分量表所属类别（如 "腹部问诊评分"）
    患者画像       性别、年龄、职业、文化、就诊状态、心理状态
    现病史 / 既往史 / 个人史 / 婚育史 / 家族史   病史字段
    诊断          真实诊断（仅用于评分，不注入给患者角色，防止剧透）
    评分维度       评分时需要考查的关键点（供评分引擎使用）
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CaseLoadError(Exception):
    """病例加载或校验失败时抛出。"""


@dataclass
class Case:
    """一个结构化病例对象。

    使用 dataclass 便于类型安全地访问字段，同时保留原始 JSON 以灵活扩展。
    """
    case_id: str
    chief_complaint: str                    # 主诉
    scoring_category: str                   # 评分属性
    department: str = ""                    # 科室（用于按科室筛选）
    difficulty: str = ""                    # 难度（易/中/难，用于按难度筛选）
    patient_profile: dict[str, Any] = field(default_factory=dict)  # 患者画像
    history: dict[str, str] = field(default_factory=dict)          # 病史字段
    diagnosis: str = ""                     # 真实诊断（评分用）
    scoring_points: list[str] = field(default_factory=list)        # 评分维度
    raw: dict[str, Any] = field(default_factory=dict)              # 原始数据

    @property
    def display_name(self) -> str:
        """用于侧边栏展示的病例名，例如 "P01 · 腹痛"."""
        return f"{self.case_id} · {self.chief_complaint}"


def _as_list(value: Any) -> list[str]:
    """把字段规整为字符串列表，兼容 JSON 中字符串或列表两种写法。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


# 合法难度取值（用于病例校验与筛选）
VALID_DIFFICULTIES = frozenset({"易", "中", "难"})


def validate_case(case: Case) -> list[str]:
    """校验病例数据完整性，返回问题列表（空列表表示通过）。

    检查：病例编号、主诉、诊断、科室、难度取值、现病史、评分维度。
    用于加载病例库后做数据质量断言（不致加载失败，仅报告问题）。
    """
    problems: list[str] = []
    cid = case.case_id or "?"
    if not case.chief_complaint:
        problems.append(f"{cid}：缺少主诉")
    if not case.diagnosis:
        problems.append(f"{cid}：缺少诊断（评分与剧透检测均依赖）")
    if not case.department:
        problems.append(f"{cid}：缺少科室")
    if case.difficulty not in VALID_DIFFICULTIES:
        problems.append(f"{cid}：难度取值非法（{case.difficulty!r}，应为 易/中/难）")
    if not case.history.get("现病史", "").strip():
        problems.append(f"{cid}：缺少现病史")
    if not case.scoring_points:
        problems.append(f"{cid}：缺少评分维度")
    return problems


def load_case(path: Path | str) -> Case:
    """从单个 JSON 文件加载病例。

    Args:
        path: 病例 JSON 文件路径。

    Returns:
        解析并校验后的 Case 对象。

    Raises:
        CaseLoadError: 文件不存在、JSON 非法或缺少关键字段时抛出。
    """
    path = Path(path)
    if not path.exists():
        raise CaseLoadError(f"病例文件不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CaseLoadError(f"病例 JSON 解析失败：{path}（{e}）")

    # 兼容顶层 case_id 缺失时回退到文件名
    case_id = data.get("case_id") or path.stem
    chief = data.get("主诉") or data.get("chief_complaint")
    if not chief:
        raise CaseLoadError(f"病例缺少主诉字段：{path}")

    difficulty = str(data.get("难度") or data.get("difficulty") or "")
    if difficulty and difficulty not in VALID_DIFFICULTIES:
        raise CaseLoadError(f"病例难度取值非法：{path}（{difficulty!r}，应为 易/中/难）")

    # 患者画像：兼容中文字段与英文别名
    profile = data.get("患者画像") or data.get("patient_profile") or {}
    if not isinstance(profile, dict):
        profile = {"描述": str(profile)}

    history = {
        "现病史": data.get("现病史", "") or data.get("present_illness", ""),
        "既往史": data.get("既往史", "") or data.get("past_history", ""),
        "个人史": data.get("个人史", "") or data.get("personal_history", ""),
        "婚育史": data.get("婚育史", "") or data.get("marital_history", ""),
        "家族史": data.get("家族史", "") or data.get("family_history", ""),
    }

    return Case(
        case_id=str(case_id),
        chief_complaint=str(chief),
        scoring_category=str(data.get("评分属性") or data.get("scoring_category") or "问诊评分"),
        department=str(data.get("科室") or data.get("department") or ""),
        difficulty=difficulty,
        patient_profile=profile,
        history=history,
        diagnosis=str(data.get("诊断") or data.get("diagnosis") or ""),
        scoring_points=_as_list(data.get("评分维度") or data.get("scoring_points")),
        raw=data,
    )


def load_all_cases(cases_dir: Path | str) -> dict[str, Case]:
    """加载目录下所有病例 JSON。

    Args:
        cases_dir: 病例库目录。

    Returns:
        以 case_id 为键的病例字典。
    """
    cases: dict[str, Case] = {}
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        return cases
    for path in sorted(cases_dir.glob("*.json")):
        case = load_case(path)
        if case.case_id in cases:
            raise CaseLoadError(f"发现重复的病例编号：{case.case_id}（{path}）")
        cases[case.case_id] = case
    return cases


def case_to_facts(case: Case) -> str:
    """把病例转成注入给角色引擎的"事实描述"文本。

    注意：这里刻意【不包含诊断】字段，避免患者角色主动泄露诊断，
    从而保证问诊训练的真实性（学生需自行问出关键信息）。

    Args:
        case: 病例对象。

    Returns:
        患者可感知的事实描述文本。
    """
    parts = [
        f"患者主诉：{case.chief_complaint}",
        f"患者基本信息：{case.patient_profile}",
    ]
    for key, val in case.history.items():
        if val:
            parts.append(f"{key}：{val}")
    return "\n".join(parts)
