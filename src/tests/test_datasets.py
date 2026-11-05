# -*- coding: utf-8 -*-
"""MCQ 评测数据集完整性测试。

重点防止「正确答案位置偏置」导致评测失真：若某一份数据集里绝大多数
答案都是 A，模型只需固定输出 A 就能取得虚高准确率，多模型对比与
macro-F1 都会失去区分度。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LETTERS = {"A", "B", "C", "D"}


def _iter_datasets():
    """遍历 data/ 下的全部 JSON 数据集，产出 (文件名, 题目列表)。"""
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
        yield path.name, list(items or [])


def test_all_datasets_are_non_empty_and_complete():
    names = [name for name, _ in _iter_datasets()]
    assert len(names) == 6, f"应有 6 套数据集，实际 {len(names)} 套：{names}"

    for name, items in _iter_datasets():
        assert len(items) == 40, f"{name} 应有 40 题，实际 {len(items)} 题"
        for index, item in enumerate(items):
            assert str(item.get("question", "")).strip(), f"{name} 第 {index + 1} 题缺少题干"
            options = item.get("options") or {}
            assert set(options) == LETTERS, f"{name} 第 {index + 1} 题选项不是完整的 A/B/C/D"
            assert all(str(v).strip() for v in options.values()), (
                f"{name} 第 {index + 1} 题存在空选项"
            )
            answer = str(item.get("answer", "")).strip().upper()
            assert answer in LETTERS, f"{name} 第 {index + 1} 题答案非法：{answer!r}"
            assert str(options[answer]).strip(), (
                f"{name} 第 {index + 1} 题的答案指向空选项"
            )
            assert str(item.get("category", "")).strip(), (
                f"{name} 第 {index + 1} 题缺少 category"
            )


def test_answer_distribution_is_balanced():
    """每套数据集的答案分布应大致均衡，不能出现答案位置偏置。"""
    for name, items in _iter_datasets():
        dist = Counter(str(item.get("answer", "")).strip().upper() for item in items)
        total = len(items)
        # 四个选项都应出现过，且任一选项占比不超过 40%
        assert set(dist) == LETTERS, f"{name} 答案未覆盖 A/B/C/D：{dict(dist)}"
        assert max(dist.values()) <= total * 0.4, (
            f"{name} 答案分布偏置：{dict(dist)}（共 {total} 题）"
        )


def test_questions_are_not_duplicated():
    """同一套及跨数据集的题干都不应重复，避免评测样本泄漏。"""
    seen: dict[str, str] = {}
    for name, items in _iter_datasets():
        for item in items:
            question = str(item.get("question", "")).strip()
            assert question not in seen, (
                f"题目重复：{name} 与 {seen.get(question)} 均出现 {question!r}"
            )
            seen[question] = name
