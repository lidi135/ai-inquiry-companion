# -*- coding: utf-8 -*-
"""训练 / 考试记录持久化与个人成长档案数据。

每次问诊评分完成后，将评分结果以 JSON 形式落盘到 RECORDS_DIR，
并按时间顺序读取历史记录，用于个人成长档案（历史表、总分趋势、能力雷达图）。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .case_loader import Case
from .config import RECORDS_DIR
from .scoring_engine import ScoringResult

# 记录文件名时间戳格式（含微秒，避免同一秒多次保存相互覆盖）
_TIMESTAMP_FMT = "%Y%m%d_%H%M%S_%f"
_FRIENDLY_FMT = "%Y-%m-%d %H:%M:%S"


def save_record(
    result: ScoringResult,
    case: Case,
    *,
    mode: str = "技能训练",
    student_id: str = "self",
    class_id: str = "default",
) -> Path:
    """把一次评分结果保存为单独的 JSON 记录文件。

    Args:
        result:     评分结果。
        case:       对应的病例对象。
        mode:       训练模式标签（"技能训练" / "考试模拟"）。
        student_id: 学生标识；本地默认 "self"，接入班级时可由 UI 改为学号。
        class_id:   班级标识；默认 "default"，用于教师视图按班级聚合。

    Returns:
        已写入的记录文件路径。
    """
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    path = RECORDS_DIR / f"{now.strftime(_TIMESTAMP_FMT)}_{case.case_id}.json"
    record = {
        "时间": now.strftime(_FRIENDLY_FMT),
        "病例": case.display_name,
        "case_id": case.case_id,
        "科室": case.department,
        "难度": case.difficulty,
        "模式": mode,
        "student_id": student_id,
        "class_id": class_id,
        **result.to_dict(),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_history() -> list[dict[str, Any]]:
    """按时间顺序读取全部历史评分记录（文件名时间戳排序）。

    Returns:
        记录字典列表；无记录或读取失败时返回空列表。
    """
    if not RECORDS_DIR.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(RECORDS_DIR.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            # 跳过损坏或不完整的记录文件，不影响整体读取
            continue
    return records


# 会话状态文件：放在独立子目录，避免被 load_history 当作评分记录读取
def _session_file() -> Path:
    return RECORDS_DIR / "_session" / "current.json"


def save_session(state: dict[str, Any]) -> None:
    """把当前会话状态（对话、报告、计时）落盘，支持刷新/重进恢复。

    持久化失败不影响主流程（静默忽略）。
    """
    try:
        path = _session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def load_session() -> dict[str, Any] | None:
    """读取上次保存的会话状态；无记录或解析失败时返回 None。"""
    path = _session_file()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_session() -> None:
    """清除已保存的会话状态。"""
    try:
        path = _session_file()
        if path.exists():
            path.unlink()
    except OSError:
        pass
