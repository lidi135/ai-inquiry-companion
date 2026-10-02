"""同义词词典模块单测。"""
from __future__ import annotations

from core.synonyms import contains_any, expand_term, load_synonyms


def test_load_synonyms_returns_dict():
    d = load_synonyms()
    assert isinstance(d, dict)
    # 关键术语都在
    assert "糖尿病" in d
    assert "高血压" in d
    # 同义词都是列表
    for v in d.values():
        assert isinstance(v, list)


def test_expand_term_includes_original_and_extras():
    extras = expand_term("糖尿病")
    assert "糖尿病" in extras
    assert "DM" in extras
    assert "血糖高" in extras


def test_expand_term_unknown_returns_self():
    extras = expand_term("未知术语XYZ")
    assert extras == ["未知术语XYZ"]


def test_contains_any_matches_synonym():
    # 文本里写"血糖高"，能识别出"糖尿病"族的同义变体「血糖高」
    assert contains_any("病人有血糖高", ["血糖高"])


def test_contains_any_returns_false():
    assert not contains_any("头痛完全", [])  # 空候选
    assert not contains_any("", ["糖尿病"])  # 空文本


def test_contains_any_case_insensitive():
    assert contains_any("病人有 DM", ["dm"])
    assert contains_any("病人有dm", ["DM"])


def test_contains_any_partial_match():
    # 子串也算命中（口语中常出现"血糖高了很久"）
    assert contains_any("血糖高了很久", ["血糖高"])


def test_contains_any_no_match():
    assert not contains_any("头痛发热", ["糖尿病"])
