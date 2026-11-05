# -*- coding: utf-8 -*-
"""训练记录持久化模块单元测试。"""
import json

from core.case_loader import Case
from core.records import (
    clear_session,
    load_history,
    load_session,
    migrate_legacy_session,
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


def test_sessions_are_isolated_by_session_id(tmp_path, monkeypatch):
    """不同浏览器会话必须写入不同文件，不能互相覆盖。"""
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    save_session({"case_id": "P01", "messages": [{"role": "user", "content": "A"}]}, "sid-a")
    save_session({"case_id": "P02", "messages": [{"role": "user", "content": "B"}]}, "sid-b")

    assert load_session("sid-a")["case_id"] == "P01"
    assert load_session("sid-b")["case_id"] == "P02"
    assert load_session("sid-c") is None

    clear_session("sid-a")
    assert load_session("sid-a") is None
    assert load_session("sid-b")["case_id"] == "P02"


def test_migrate_legacy_session_only_once(tmp_path, monkeypatch):
    """旧版 _session/current.json 应迁移到当前会话，且只迁移一次。"""
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    legacy_dir = tmp_path / "_session"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_dir / "current.json"
    state = {"case_id": "P01", "messages": [{"role": "user", "content": "旧会话"}]}
    legacy.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    migrated = migrate_legacy_session("sid-a")
    assert migrated == state
    assert load_session("sid-a") == state
    assert not legacy.exists()
    assert list(legacy_dir.glob("current_migrated_*.json"))

    # 旧文件已归档，其他会话不会重复拿到同一份旧数据
    assert migrate_legacy_session("sid-b") is None


def test_migrate_legacy_session_without_legacy_file(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    assert migrate_legacy_session("sid-a") is None


def test_migrate_legacy_session_respects_case_id(tmp_path, monkeypatch):
    """旧会话不属于当前病例时不应迁移，避免旧数据被错误覆盖。"""
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    legacy_dir = tmp_path / "_session"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_dir / "current.json"
    state = {"case_id": "P01", "messages": [{"role": "user", "content": "旧会话"}]}
    legacy.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    assert migrate_legacy_session("sid-a", expected_case_id="P02") is None
    assert legacy.exists()
    assert migrate_legacy_session("sid-a", expected_case_id="P01") == state
    assert not legacy.exists()


def test_migrate_legacy_session_overwrites_empty_target(tmp_path, monkeypatch):
    """目标会话只写了空状态时，仍应允许旧会话迁入，避免错过迁移。"""
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    save_session({"case_id": "P01", "messages": []}, "sid-a")
    legacy_dir = tmp_path / "_session"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_dir / "current.json"
    state = {"case_id": "P01", "messages": [{"role": "user", "content": "旧会话"}]}
    legacy.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    migrated = migrate_legacy_session("sid-a")
    assert migrated == state
    assert load_session("sid-a") == state
    assert not legacy.exists()


def test_save_record_persists_identity_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    save_record(
        _result(), _case(), student_id="alice", class_id="C1", session_id="sid-a"
    )
    record = load_history()[0]
    assert record["student_id"] == "alice"
    assert record["class_id"] == "C1"
    assert record["session_id"] == "sid-a"


def test_load_history_filters_by_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("core.records.RECORDS_DIR", tmp_path)
    save_record(_result(), _case(), student_id="alice", class_id="C1", session_id="sid-a")
    save_record(_result(), _case(), student_id="bob", class_id="C1", session_id="sid-b")
    save_record(_result(), _case(), student_id="bob", class_id="C2", session_id="sid-c")

    assert len(load_history(student_id="alice")) == 1
    assert len(load_history(class_id="C1")) == 2
    assert len(load_history(session_id="sid-b")) == 1
    assert len(load_history(student_id="bob", class_id="C2")) == 1
    assert load_history(student_id="nobody") == []
