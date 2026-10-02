# -*- coding: utf-8 -*-
"""病例加载模块单元测试。"""
import json

import pytest

from core.case_loader import (
    Case,
    CaseLoadError,
    case_to_facts,
    load_all_cases,
    load_case,
    validate_case,
)


def _write_case(tmp_path, name="P01.json", data=None):
    data = data or {
        "case_id": "P01",
        "主诉": "腹痛 3 天",
        "评分属性": "腹部问诊评分",
        "科室": "消化内科",
        "难度": "中",
        "患者画像": {"性别": "女", "年龄": 55},
        "现病史": "3 天前腹痛，餐后加重",
        "诊断": "慢性胃炎",
        "评分维度": ["主诉采集", "部位", "诱因"],
    }
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_case_basic(tmp_path):
    case = load_case(_write_case(tmp_path))
    assert case.case_id == "P01"
    assert case.chief_complaint == "腹痛 3 天"
    assert case.department == "消化内科"
    assert case.difficulty == "中"
    assert case.diagnosis == "慢性胃炎"
    assert case.scoring_points == ["主诉采集", "部位", "诱因"]


def test_load_case_missing_file():
    with pytest.raises(CaseLoadError):
        load_case("__no_such_file__.json")


def test_load_case_missing_chief_complaint(tmp_path):
    path = tmp_path / "P02.json"
    path.write_text(json.dumps({"case_id": "P02"}), encoding="utf-8")
    with pytest.raises(CaseLoadError):
        load_case(path)


def test_load_case_english_aliases(tmp_path):
    path = tmp_path / "P03.json"
    path.write_text(json.dumps({
        "case_id": "P03",
        "chief_complaint": "cough",
        "department": "呼吸内科",
        "difficulty": "易",
        "present_illness": "咳嗽",
    }, ensure_ascii=False), encoding="utf-8")
    case = load_case(path)
    assert case.chief_complaint == "cough"
    assert case.department == "呼吸内科"
    assert case.history["现病史"] == "咳嗽"


def test_load_all_cases(tmp_path):
    _write_case(tmp_path, "P01.json")
    _write_case(tmp_path, "P04.json",
                {"case_id": "P04", "主诉": "胸闷", "诊断": "冠心病"})
    cases = load_all_cases(tmp_path)
    assert set(cases) == {"P01", "P04"}


def test_case_to_facts_excludes_diagnosis(tmp_path):
    case = load_case(_write_case(tmp_path))
    facts = case_to_facts(case)
    assert "慢性胃炎" not in facts  # 诊断不得泄露给患者角色
    assert "腹痛 3 天" in facts


def test_display_name():
    case = Case(case_id="P01", chief_complaint="腹痛", scoring_category="问诊评分")
    assert case.display_name == "P01 · 腹痛"


def test_load_case_invalid_difficulty_raises(tmp_path):
    path = tmp_path / "P05.json"
    path.write_text(json.dumps({
        "case_id": "P05",
        "主诉": "头痛",
        "难度": "很难",
    }, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CaseLoadError):
        load_case(path)


def test_load_all_cases_duplicate_id_raises(tmp_path):
    _write_case(tmp_path, "P01.json")
    _write_case(tmp_path, "P01_copy.json")  # 不同文件名但 case_id 相同
    with pytest.raises(CaseLoadError):
        load_all_cases(tmp_path)


def test_validate_case_valid(tmp_path):
    case = load_case(_write_case(tmp_path))
    assert validate_case(case) == []


def test_validate_case_reports_problems():
    case = Case(
        case_id="P99",
        chief_complaint="",
        scoring_category="问诊评分",
        department="",
        difficulty="不明",
        history={},
        diagnosis="",
        scoring_points=[],
    )
    problems = validate_case(case)
    assert problems
    assert any("主诉" in p for p in problems)
    assert any("诊断" in p for p in problems)
    assert any("难度" in p for p in problems)
    assert any("现病史" in p for p in problems)
    assert any("评分维度" in p for p in problems)
