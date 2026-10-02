# -*- coding: utf-8 -*-
"""不平衡样本处理模块单元测试。"""
from collections import Counter

import pytest

from eval.imbalance import (
    KNNClassifier,
    analyze_balance,
    make_imbalanced_dataset,
    random_oversample,
    random_undersample,
    run_imbalance_experiment,
)


def test_analyze_balance():
    dist = analyze_balance(["A", "A", "A", "B"])
    assert dist["A"]["count"] == 3
    assert dist["B"]["count"] == 1
    # 多数/少数 = 3:1
    assert dist["_imbalance_ratio"] == 3.0


def test_analyze_balance_empty():
    dist = analyze_balance([])
    assert dist == {}


def test_random_oversample_balances():
    X = [[0, 0], [0, 0], [0, 0], [1, 1]]
    y = ["A", "A", "A", "B"]
    X_new, y_new = random_oversample(X, y, seed=0)
    counts = Counter(y_new)
    assert counts["A"] == counts["B"] == 3


def test_random_undersample_balances():
    X = [[0, 0], [0, 0], [0, 0], [1, 1]]
    y = ["A", "A", "A", "B"]
    X_new, y_new = random_undersample(X, y, seed=0)
    counts = Counter(y_new)
    assert counts["A"] == counts["B"] == 1
    assert len(X_new) == len(y_new)


def test_knn_predict():
    X = [[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]]
    y = ["A", "A", "B", "B"]
    clf = KNNClassifier(k=3).fit(X, y)
    assert clf.predict_one([0.0, 0.0]) == "A"
    assert clf.predict_one([5.0, 5.0]) == "B"


def test_knn_predict_unfitted_raises():
    clf = KNNClassifier(k=3)
    with pytest.raises(ValueError, match="未训练|为空"):
        clf.predict_one([0.0, 0.0])


def test_make_imbalanced_dataset():
    tX, ty, eX, ey = make_imbalanced_dataset(n_major=180, n_minor=20, seed=42)
    train_counts = Counter(ty)
    assert train_counts["A"] == 180
    assert train_counts["B"] == 20
    # 测试集类别平衡
    test_counts = Counter(ey)
    assert test_counts["A"] == test_counts["B"] == 50


def test_run_imbalance_experiment():
    out = run_imbalance_experiment(seed=42)
    assert set(out["results"]) == {"baseline", "oversample", "undersample", "class_weight"}
    for r in out["results"].values():
        assert 0 <= r["accuracy"] <= 1
        assert 0 <= r["macro_f1"] <= 1
        assert 0 <= r["minority_recall"] <= 1
