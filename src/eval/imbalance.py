# -*- coding: utf-8 -*-
"""不平衡样本处理方案与实验验证。

医学诊断数据普遍存在类别不平衡（常见病样本多、罕见病样本少），
直接训练会导致模型偏向多数类、少数类召回率极低。

本模块实现并对比以下处理方案：
    1. 基线（不处理）
    2. 随机过采样（Random Oversampling）：复制少数类样本
    3. 随机欠采样（Random Undersampling）：删减多数类样本
    4. 类别加权（Class Weighting）：在分类器中给少数类更高权重

评估重点采用 macro-F1 与少数类 recall（二者对不平衡更敏感），
而非会被多数类主导的 accuracy。

运行方式：
    python -m eval.imbalance
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Sequence

from .metrics import compute_metrics


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """欧氏距离。"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def analyze_balance(labels: Sequence[str]) -> dict[str, Any]:
    """统计类别分布，返回各类别样本数、占比与不平衡比。

    Args:
        labels: 标签列表。

    Returns:
        {类别: {"count": n, "ratio": r}}，以及最大/最小类别的样本数之比。
    """
    counter = Counter(labels)
    total = len(labels)
    dist: dict[str, Any] = {
        k: {"count": v, "ratio": round(v / total, 4) if total else 0.0}
        for k, v in counter.items()
    }
    if counter:
        majority = max(counter.values())
        minority = min(counter.values())
        dist["_imbalance_ratio"] = round(majority / minority, 2)
    return dist


def random_oversample(X: Sequence[Sequence[float]], y: Sequence[str], seed: int = 0):
    """随机过采样：把每个类别复制到与多数类相同数量。

    Args:
        X: 特征矩阵。
        y: 标签。
        seed: 随机种子（保证可复现）。

    Returns:
        (X_new, y_new)。
    """
    rng = random.Random(seed)
    counter = Counter(y)
    max_count = max(counter.values())
    X_new, y_new = list(X), list(y)
    for label, count in counter.items():
        if count < max_count:
            indices = [i for i, y_label in enumerate(y) if y_label == label]
            for _ in range(max_count - count):
                idx = rng.choice(indices)
                X_new.append(list(X[idx]))
                y_new.append(label)
    return X_new, y_new


def random_undersample(X: Sequence[Sequence[float]], y: Sequence[str], seed: int = 0):
    """随机欠采样：把多数类随机删减到与少数类相同数量。

    Args:
        X: 特征矩阵。
        y: 标签。
        seed: 随机种子。

    Returns:
        (X_new, y_new)。
    """
    rng = random.Random(seed)
    counter = Counter(y)
    min_count = min(counter.values())
    keep_indices: list[int] = []
    for label in counter:
        indices = [i for i, y_label in enumerate(y) if y_label == label]
        if len(indices) > min_count:
            indices = rng.sample(indices, min_count)
        keep_indices.extend(indices)
    keep_indices.sort()
    return [list(X[i]) for i in keep_indices], [y[i] for i in keep_indices]


class KNNClassifier:
    """简单 K 近邻分类器（纯 Python 实现，无需第三方依赖）。

    Attributes:
        k:             邻居数量。
        class_weights: 类别权重字典 {类别: 权重}，用于加权投票（类别加权方案）。
    """

    def __init__(self, k: int = 5, class_weights: dict[str, float] | None = None):
        self.k = k
        self.class_weights = class_weights or {}
        self._X: list[Sequence[float]] = []
        self._y: list[str] = []

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[str]) -> "KNNClassifier":
        """记忆训练样本（KNN 为惰性学习）。"""
        self._X = list(X)
        self._y = list(y)
        return self

    def predict_one(self, x: Sequence[float]) -> str:
        """预测单个样本的类别（距离加权 + 类别加权投票）。"""
        if not self._X:
            raise ValueError("KNN 分类器未训练或训练集为空，无法预测")
        # 计算所有训练样本的距离并排序
        dists = sorted(
            ((euclidean(x, tx), ty) for tx, ty in zip(self._X, self._y)),
            key=lambda t: t[0],
        )
        neighbors = dists[: self.k]
        scores: dict[str, float] = {}
        for d, label in neighbors:
            # 距离加权：距离越近权重越大（加 eps 防止除零）
            w = 1.0 / (d + 1e-6)
            # 类别加权：少数类给更高权重，缓解多数类主导
            w *= self.class_weights.get(label, 1.0)
            scores[label] = scores.get(label, 0.0) + w
        return max(scores, key=lambda k: scores[k])

    def predict(self, X: Sequence[Sequence[float]]) -> list[str]:
        return [self.predict_one(x) for x in X]


# ---------------------------------------------------------------------------
# 模拟不平衡数据集（保证实验可离线复现）
# ---------------------------------------------------------------------------
def make_imbalanced_dataset(
    n_major: int = 180,
    n_minor: int = 20,
    seed: int = 42,
):
    """生成二分类不平衡模拟数据。

    多数类（A）与少数类（B）为两个高斯簇，特征维度 2。
    返回训练集与一个类别平衡的测试集，用于评估泛化性能。

    Args:
        n_major: 多数类样本数。
        n_minor: 少数类样本数。
        seed:    随机种子。

    Returns:
        (train_X, train_y, test_X, test_y)。
    """
    rng = random.Random(seed)

    def cluster(n, center, spread):
        return [[rng.gauss(center[i], spread) for i in range(2)] for _ in range(n)]

    # 多数类 A：中心 (0, 0)；少数类 B：中心 (2, 2)
    train_X = cluster(n_major, (0.0, 0.0), 1.0) + cluster(n_minor, (2.0, 2.0), 1.0)
    train_y = ["A"] * n_major + ["B"] * n_minor

    # 平衡测试集：每类 50 个
    test_X = cluster(50, (0.0, 0.0), 1.0) + cluster(50, (2.0, 2.0), 1.0)
    test_y = ["A"] * 50 + ["B"] * 50
    return train_X, train_y, test_X, test_y


# ---------------------------------------------------------------------------
# 不平衡实验主流程
# ---------------------------------------------------------------------------
def run_imbalance_experiment(seed: int = 42) -> dict:
    """运行"处理前 vs 处理后"的对比实验。

    依次评估四种策略在平衡测试集上的 accuracy / macro-F1 / 少数类 recall，
    展示不平衡处理对少数类识别能力的提升。

    Args:
        seed: 随机种子。

    Returns:
        各策略的指标字典，便于绘制对比图。
    """
    train_X, train_y, test_X, test_y = make_imbalanced_dataset(seed=seed)
    labels = ["A", "B"]

    def eval_strategy(name: str, clf: KNNClassifier) -> dict:
        pred = clf.predict(test_X)
        m = compute_metrics(test_y, pred, labels=labels)
        return {
            "strategy": name,
            "accuracy": round(m.accuracy, 4),
            "macro_f1": round(m.macro_f1, 4),
            "minority_recall": round(m.per_class["B"]["recall"], 4),
        }

    results: dict = {}

    # 1. 基线（不处理）
    results["baseline"] = eval_strategy("基线（不处理）", KNNClassifier(k=5).fit(train_X, train_y))

    # 2. 随机过采样
    ov_X, ov_y = random_oversample(train_X, train_y, seed=seed)
    results["oversample"] = eval_strategy("随机过采样", KNNClassifier(k=5).fit(ov_X, ov_y))

    # 3. 随机欠采样
    un_X, un_y = random_undersample(train_X, train_y, seed=seed)
    results["undersample"] = eval_strategy("随机欠采样", KNNClassifier(k=5).fit(un_X, un_y))

    # 4. 类别加权（少数类权重 = 多数类样本数 / 少数类样本数）
    counter = Counter(train_y)
    weight_minor = counter["A"] / counter["B"]
    weights = {"A": 1.0, "B": weight_minor}
    results["class_weight"] = eval_strategy(
        "类别加权", KNNClassifier(k=5, class_weights=weights).fit(train_X, train_y)
    )

    return {
        "distribution": analyze_balance(train_y),
        "results": results,
    }


if __name__ == "__main__":
    import json

    out = run_imbalance_experiment()
    print(json.dumps(out, ensure_ascii=False, indent=2))
