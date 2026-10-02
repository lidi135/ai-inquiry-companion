# -*- coding: utf-8 -*-
"""教师视图数据聚合 —— 从训练记录中按班级/学生聚合统计指标。

主要 API：
    * :func:`load_class_overview` ：汇总「总分分布 / 维度失分 / 趋势」
    * :func:`dim_loss_ranking`   ：按维度计算全班平均失分排名（找弱项）

数据源：``records/*.json``，每条记录由 :func:`core.records.save_record` 写入。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _iter_records(class_id: str | None = None) -> list[dict[str, Any]]:
    """读取符合班级的所有记录，按学生与时间排序。"""
    from .records import load_history

    records = load_history()
    if class_id:
        records = [r for r in records if r.get("class_id", "default") == class_id]
    return records


def _extract_dim_score(rec: dict[str, Any], dim: str) -> float | None:
    """从记录中提取某维度的得分（兼容 DimensionDetail 与旧格式）。"""
    dims = rec.get("分维度得分") or {}
    v = dims.get(dim)
    if isinstance(v, dict):
        return float(v.get("得分", 0) or 0)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_class_overview(class_id: str = "default") -> dict[str, Any]:
    """汇总指定班级（默认 ``default``）的统计数据。

    Returns:
        dict，包含：
            * ``n_records``  记录条数
            * ``students``   学生 ID 列表
            * ``avg_total``  全班平均总分
            * ``total_trend`` 最近 10 次记录的总分趋势（按时间排序）
            * ``dim_avg``    各维度平均得分（已折算为 100 制，便于横比）
            * ``records``    原始记录列表（供渲染表格）
    """
    records = _iter_records(class_id)
    if not records:
        return {
            "n_records": 0, "students": [],
            "avg_total": 0.0, "total_trend": [],
            "dim_avg": {}, "records": [],
        }

    totals = [float(r.get("总分", 0) or 0) for r in records]
    students = sorted({r.get("student_id", "self") for r in records})

    # 维度平均（折算成 100 制）
    dim_scores: dict[str, list[float]] = defaultdict(list)
    # 收集所有出现过的维度
    all_dims: set[str] = set()
    for r in records:
        for k in (r.get("分维度得分") or {}).keys():
            all_dims.add(str(k))
    for r in records:
        for d in all_dims:
            s = _extract_dim_score(r, d)
            if s is not None:
                # 把 "得分/满分" 折算成 100 制
                dims = r.get("分维度得分") or {}
                raw = dims.get(d)
                if isinstance(raw, dict):
                    max_v = float(raw.get("满分", 1) or 1)
                else:
                    max_v = 100.0
                dim_scores[d].append((s / max_v) * 100.0 if max_v else s)

    dim_avg = {
        d: round(sum(v) / len(v), 1)
        for d, v in dim_scores.items() if v
    }

    # 趋势图：按时间排序，取最后 10 条
    sorted_records = sorted(records, key=lambda r: str(r.get("时间", "")))
    trend = [
        {"时间": r.get("时间", ""), "总分": float(r.get("总分", 0) or 0)}
        for r in sorted_records[-10:]
    ]

    return {
        "n_records": len(records),
        "students": students,
        "avg_total": round(sum(totals) / len(totals), 1),
        "total_trend": trend,
        "dim_avg": dim_avg,
        "records": records,
    }


def dim_loss_ranking(class_id: str = "default", top_n: int = 5) -> list[tuple[str, float]]:
    """按维度计算全班平均失分（100 - 均分），返回 top_n 弱项。

    Returns:
        ``[(维度名, 平均失分), ...]``，按失分降序。
    """
    overview = load_class_overview(class_id)
    dim_avg = overview.get("dim_avg", {})
    if not dim_avg:
        return []
    losses = [(d, round(100.0 - avg, 1)) for d, avg in dim_avg.items()]
    losses.sort(key=lambda x: -x[1])
    return losses[:top_n]
