# -*- coding: utf-8 -*-
"""训练记录持久化模块单元测试。"""
import json

from core.case_loader import Case
from core.records import (
    clear_session,
    load_history,
    load_session,
    save_record,
    save_session,
)
from core.scoring_engine import ScoringResult


def _case():
    return Case(
        case_id="P01",
        chief_complaint="腹痛 3 天",
        scoring_category="问诊评分",
        department="消化内科",
        difficulty="中",
    )


def _result():
    return ScoringResult(
        total=86.5,
        dimensions={"现病史": 12.0, "既往史": 8.0},
        strengths=["问得不错"],
        weaknesses=["遗漏家族史"],
        suggestions="继续努力",
        method="rule",
    )


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    path = save_record(_result(), _case(), mode="考试模拟")
    assert path.exists()
    assert path.parent == tmp_path

    history = load_history()
    assert len(history) == 1
    rec = history[0]
    assert rec["病例"] == "P01 · 腹痛 3 天"
    assert rec["模式"] == "考试模拟"
    assert rec["总分"] == 86.5
    assert rec["分维度得分"]["现病史"] == 12.0


def test_load_history_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    assert load_history() == []


def test_load_history_skips_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    (tmp_path / "good.json").write_text(
        json.dumps({"总分": 90}, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
    history = load_history()
    assert len(history) == 1
    assert history[0]["总分"] == 90


def test_load_history_sorted_by_time(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    # 通过文件名时间戳区分顺序
    (tmp_path / "20260101_000000_000001_P01.json").write_text(
        json.dumps({"总分": 70}, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "20260102_000000_000001_P01.json").write_text(
        json.dumps({"总分": 90}, ensure_ascii=False), encoding="utf-8"
    )
    history = load_history()
    assert history[0]["总分"] == 70
    assert history[1]["总分"] == 90


def test_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    assert load_session() is None

    state = {"case_id": "P01", "messages": [{"role": "user", "content": "你好"}]}
    save_session(state)
    assert load_session() == state

    clear_session()
    assert load_session() is None


def test_session_file_not_in_history(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    save_session({"case_id": "P01"})
    # 会话文件放在独立子目录，不应被当作评分记录读取
    assert load_history() == []
