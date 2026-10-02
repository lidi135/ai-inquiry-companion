"""教师视图数据聚合单元测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from core import records as records_mod
from core.case_loader import Case
from core.records import save_record
from core.scoring_engine import DimensionDetail, ScoringResult
from core.teacher import dim_loss_ranking, load_class_overview


@pytest.fixture()
def tmp_records(monkeypatch, tmp_path: Path):
    """把 RECORDS_DIR 切到临时目录，写入若干测试记录。"""
    monkeypatch.setattr(records_mod, "RECORDS_DIR", tmp_path)

    def _make(case_id: str, total: float, dims: dict[str, DimensionDetail],
              student_id: str = "stu_1", class_id: str = "C1") -> None:
        case = Case(case_id=case_id, chief_complaint="腹痛", scoring_category="1")
        result = ScoringResult(total=total, dimensions=dims, method="rule")
        save_record(result, case, student_id=student_id, class_id=class_id)

    return _make


def test_overview_empty_when_no_records(tmp_records):
    overview = load_class_overview("C1")
    assert overview["n_records"] == 0
    assert overview["avg_total"] == 0.0
    assert overview["students"] == []


def test_overview_filters_by_class(tmp_records):
    tmp_records("P01", 80.0, {}, student_id="alice", class_id="C1")
    tmp_records("P02", 60.0, {}, student_id="bob", class_id="C2")
    overview_c1 = load_class_overview("C1")
    overview_c2 = load_class_overview("C2")
    assert overview_c1["n_records"] == 1
    assert overview_c2["n_records"] == 1
    assert overview_c1["avg_total"] == 80.0
    assert overview_c2["avg_total"] == 60.0


def test_overview_aggregates_dimension_scores(tmp_records):
    tmp_records("P01", 80.0, {
        "现病史": DimensionDetail(score=12.0, max=15.0),
        "既往史": DimensionDetail(score=8.0, max=10.0),
    })
    tmp_records("P02", 90.0, {
        "现病史": DimensionDetail(score=15.0, max=15.0),
        "既往史": DimensionDetail(score=10.0, max=10.0),
    })
    overview = load_class_overview("C1")
    # 现病史均分 = (12/15 + 15/15) / 2 * 100 = 90.0
    assert overview["dim_avg"]["现病史"] == 90.0
    # 既往史均分 = (8/10 + 10/10) / 2 * 100 = 90.0
    assert overview["dim_avg"]["既往史"] == 90.0


def test_overview_total_trend_returns_latest_10(tmp_records):
    for i in range(15):
        tmp_records(f"P{i:02d}", 50.0 + i, {})
    overview = load_class_overview("C1")
    assert len(overview["total_trend"]) == 10


def test_dim_loss_ranking_returns_weakest_dimensions(tmp_records):
    # 现病史 95/100, 既往史 60/100 —— 既往史失分更多，应排第一
    tmp_records("P01", 80.0, {
        "现病史": DimensionDetail(score=14.25, max=15.0),
        "既往史": DimensionDetail(score=6.0, max=10.0),
    })
    ranking = dim_loss_ranking("C1")
    assert ranking
    # 第一项应是既往史（失分更多）
    assert ranking[0][0] == "既往史"
    assert ranking[0][1] > 30.0  # 失分 > 30%
