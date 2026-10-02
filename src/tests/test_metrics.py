# -*- coding: utf-8 -*-
"""评估指标模块单元测试。"""
import pytest

from eval.metrics import compute_metrics


def test_perfect_prediction():
    m = compute_metrics(["A", "B", "A", "B"], ["A", "B", "A", "B"], labels=["A", "B"])
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.micro_f1 == 1.0
    assert m.weighted_f1 == 1.0
    assert m.per_class["A"]["f1"] == 1.0


def test_all_wrong_multi_class():
    m = compute_metrics(["A", "B", "C"], ["B", "C", "A"], labels=["A", "B", "C"])
    assert m.accuracy == 0.0
    assert m.per_class["A"]["recall"] == 0.0


def test_partial_prediction():
    # 3 个 A 中预测对 2 个，2 个 B 中预测对 1 个
    m = compute_metrics(["A", "A", "A", "B", "B"], ["A", "A", "B", "B", "A"], labels=["A", "B"])
    assert m.per_class["A"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m.per_class["A"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m.per_class["B"]["precision"] == pytest.approx(1 / 2)
    assert m.per_class["B"]["recall"] == pytest.approx(1 / 2)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(["A", "B"], ["A"])


def test_macro_vs_micro_weighted():
    # 多数类主导场景：模型全部预测为多数类 A，少数类 B 漏检
    y_true = ["A"] * 9 + ["B"]
    y_pred = ["A"] * 10
    m = compute_metrics(y_true, y_pred, labels=["A", "B"])
    # micro-F1 等于准确率，被多数类拉高到 0.9；macro 反映少数类漏检而显著偏低
    assert m.micro_f1 == 0.9
    assert m.per_class["B"]["recall"] == 0.0
    assert m.macro_f1 < m.micro_f1
    assert m.confusion == [[9, 0], [1, 0]]


def test_confusion_matrix_shape():
    m = compute_metrics(["A", "B"], ["A", "B"], labels=["A", "B"])
    assert m.confusion == [[1, 0], [0, 1]]


def test_empty_input_returns_zeros():
    m = compute_metrics([], [])
    assert m.accuracy == 0.0
    assert m.macro_f1 == 0.0
    assert m.micro_f1 == 0.0
    assert m.weighted_f1 == 0.0
    assert m.per_class == {}
    assert m.confusion == []
