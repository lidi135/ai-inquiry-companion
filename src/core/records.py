# -*- coding: utf-8 -*-
"""训练 / 考试记录持久化与个人成长档案数据。

每次问诊评分完成后，将评分结果以 JSON 形式落盘到 RECORDS_DIR，
并按时间顺序读取历史记录，用于个人成长档案（历史表、总分趋势、能力雷达图）。
"""
from __future__ import annotations

import json
import re
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
    session_id: str = "",
) -> Path:
    """把一次评分结果保存为单独的 JSON 记录文件。

    Args:
        result:     评分结果。
        case:       对应的病例对象。
        mode:       训练模式标签（"技能训练" / "考试模拟"）。
        student_id: 学生标识；本地默认 "self"，接入班级时可由 UI 改为学号。
        class_id:   班级标识；默认 "default"，用于教师视图按班级聚合。
        session_id: 浏览器会话标识；用于未填写学号时隔离个人成长档案。

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
        "session_id": session_id,
        **result.to_dict(),
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_history(
    *,
    student_id: str | None = None,
    class_id: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """按时间顺序读取历史评分记录（文件名时间戳排序）。

    Args:
        student_id: 仅保留该学生的记录；None 表示不过滤。
        class_id:   仅保留该班级的记录；None 表示不过滤。
        session_id: 仅保留该浏览器会话的记录；None 表示不过滤。

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

    if student_id:
        records = [r for r in records if r.get("student_id", "self") == student_id]
    if class_id:
        records = [r for r in records if r.get("class_id", "default") == class_id]
    if session_id:
        records = [r for r in records if r.get("session_id", "") == session_id]
    return records


# 会话状态文件：放在独立子目录，按浏览器会话隔离，避免多用户串号
def _session_file(session_id: str = "default") -> Path:
    """返回某会话的落盘路径；对会话 ID 做白名单清洗，防止路径穿越。"""
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id or "")[:64] or "default"
    return RECORDS_DIR / "_sessions" / f"{safe}.json"


def migrate_legacy_session(
    session_id: str = "default",
    *,
    expected_case_id: str | None = None,
) -> dict[str, Any] | None:
    """把旧版固定会话文件迁移到按浏览器会话隔离的新文件。

    旧版本使用 ``records/_session/current.json`` 保存单份会话。迁移仅在
    目标会话文件不存在或只含空状态时执行一次，成功后把旧文件重命名为
    ``current_migrated_<时间戳>.json``，避免其他会话重复迁移。

    Args:
        session_id:       目标浏览器会话标识。
        expected_case_id: 当前选中的病例 ID；提供时仅在旧会话属于该病例
                          才迁移，避免把旧会话错误归到其他病例。

    Returns:
        迁移后的会话状态；无旧文件、目标已有有效对话、病例不匹配或
        解析失败时返回 None。
    """
    target = _session_file(session_id)
    existing = load_session(session_id) if target.exists() else None
    if isinstance(existing, dict) and (existing.get("messages") or existing.get("report")):
        return existing

    legacy = RECORDS_DIR / "_session" / "current.json"
    if not legacy.exists():
        return existing if isinstance(existing, dict) else None
    try:
        state = json.loads(legacy.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(state, dict):
        return None
    if expected_case_id and str(state.get("case_id", "")) != expected_case_id:
        # 旧会话不属于当前病例：保留旧文件，等用户切到对应病例后再迁移
        return None

    save_session(state, session_id)
    if not _session_file(session_id).exists():
        return None

    try:
        archive = legacy.with_name(
            f"current_migrated_{datetime.now().strftime(_TIMESTAMP_FMT)}.json"
        )
        legacy.replace(archive)
    except OSError:
        pass
    return state


def save_session(state: dict[str, Any], session_id: str = "default") -> None:
    """把当前会话状态（对话、报告、计时）落盘，支持刷新/重进恢复。

    不同浏览器会话使用不同文件，避免公网部署时互相覆盖。
    持久化失败不影响主流程（静默忽略）。
    """
    try:
        path = _session_file(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def load_session(session_id: str = "default") -> dict[str, Any] | None:
    """读取指定会话上次保存的状态；无记录或解析失败时返回 None。"""
    path = _session_file(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_session(session_id: str = "default") -> None:
    """清除指定会话已保存的状态。"""
    try:
        path = _session_file(session_id)
        if path.exists():
            path.unlink()
    except OSError:
        pass
