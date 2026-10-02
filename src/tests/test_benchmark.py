# -*- coding: utf-8 -*-
"""多模型评测与迁移模块单元测试（不涉及网络调用）。"""
import json

import pytest

from eval.benchmark import (
    load_mcq_dataset,
    migration_decay,
    parse_mcq_answer,
)


def test_parse_mcq_answer_explicit():
    assert parse_mcq_answer("答案：A") == "A"
    assert parse_mcq_answer("正确选项是 C") == "C"
    assert parse_mcq_answer("Answer: B") == "B"


def test_parse_mcq_answer_leading():
    assert parse_mcq_answer("(B) 高血压") == "B"
    assert parse_mcq_answer("D. 冠心病") == "D"


def test_parse_mcq_answer_fallback():
    assert parse_mcq_answer("我觉得选 A，不太确定") == "A"


def test_parse_mcq_answer_empty():
    assert parse_mcq_answer("") == ""
    assert parse_mcq_answer(None) == ""


def test_load_mcq_dataset_list(tmp_path):
    data = [
        {"question": "q1", "options": {"A": "a", "B": "b"}, "answer": "A", "category": "内科"},
        {"question": "q2", "options": {"A": "a", "B": "b"}, "answer": "B", "category": "外科"},
    ]
    path = tmp_path / "mcq.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    samples = load_mcq_dataset(path)
    assert len(samples) == 2
    assert samples[0].answer == "A"
    assert samples[1].category == "外科"


def test_load_mcq_dataset_with_data_key(tmp_path):
    path = tmp_path / "mcq.json"
    path.write_text(json.dumps({"data": [
        {"question": "q1", "options": {}, "answer": "C"}
    ]}, ensure_ascii=False), encoding="utf-8")
    samples = load_mcq_dataset(path)
    assert len(samples) == 1
    assert samples[0].answer == "C"


def test_migration_decay():
    assert migration_decay(80, 60) == pytest.approx(0.25)
    assert migration_decay(0, 60) == 0.0


def test_migration_decay_negative_target_better():
    # 目标域更优 -> 负衰减
    assert migration_decay(60, 70) < 0
