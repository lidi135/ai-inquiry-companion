# -*- coding: utf-8 -*-
"""评估指标计算。

提供多分类场景下的常用指标，用于：
    - 多模型性能对比
    - 零样本迁移能力评估
    - 不平衡样本处理效果验证

关键指标：准确率、精确率、召回率、F1（macro/micro/weighted）、
以及混淆矩阵。macro-F1 对少数类敏感，是评估类别不平衡任务的核心指标。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ClassificationMetrics:
    """多分类评估指标结果。

    Attributes:
        accuracy:   准确率。
        macro_f1:  宏平均 F1（各类 F1 的算术平均，对少数类敏感）。
        micro_f1:  微平均 F1（等价于准确率，受多数类主导）。
        weighted_f1: 按类别样本数加权的 F1。
        per_class:  各类别的 {precision, recall, f1} 字典。
        confusion:  混淆矩阵（二维列表）。
    """
    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_f1": round(self.micro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": self.per_class,
            "confusion": self.confusion,
        }


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """计算多分类指标（不依赖第三方库，纯 Python 实现）。

    Args:
        y_true: 真实标签列表。
        y_pred: 预测标签列表。
        labels: 类别顺序（用于混淆矩阵；缺省按出现顺序自动收集）。

    Returns:
        ClassificationMetrics。
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 长度不一致")

    # 空输入：无样本可评估，返回全 0 指标，避免除零
    if not y_true:
        return ClassificationMetrics(
            accuracy=0.0, macro_f1=0.0, micro_f1=0.0, weighted_f1=0.0,
        )

    # 确定类别顺序
    if labels is None:
        labels = list(dict.fromkeys(list(y_true) + list(y_pred)))
    label_index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)

    # 构建混淆矩阵：confusion[真实][预测]
    confusion = [[0] * n for _ in range(n)]
    correct = 0
    for t, p in zip(y_true, y_pred):
        ti, pi = label_index[t], label_index[p]
        confusion[ti][pi] += 1
        if t == p:
            correct += 1
    accuracy = correct / len(y_true)

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    weighted_f1_sum = 0.0
    total = len(y_true)

    for i, lab in enumerate(labels):
        tp = confusion[i][i]
        # 真实为该类的样本数（行和）
        actual = sum(confusion[i])
        # 预测为该类的样本数（列和）
        predicted = sum(confusion[j][i] for j in range(n))

        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[lab] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1s.append(f1)
        weighted_f1_sum += f1 * actual

    macro_f1 = sum(f1s) / n if n else 0.0
    micro_f1 = accuracy  # 多分类中 micro-F1 与准确率相等
    weighted_f1 = weighted_f1_sum / total if total else 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion=confusion,
    )
