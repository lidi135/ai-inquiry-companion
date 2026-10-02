# -*- coding: utf-8 -*-
"""评测可视化模块单元测试（仅覆盖纯数据准备逻辑）。"""
from eval.visualize import metrics_to_chart_data


def test_metrics_to_chart_data_extracts_keys():
    metrics = {
        "accuracy": 0.9,
        "macro_f1": 0.85,
        "micro_f1": 0.9,
        "weighted_f1": 0.88,
        "per_class": {},
        "confusion": [],
    }
    data = metrics_to_chart_data(metrics)
    assert data == {
        "准确率": 0.9,
        "宏观 F1": 0.85,
        "微观 F1": 0.9,
        "加权 F1": 0.88,
    }


def test_metrics_to_chart_data_partial():
    data = metrics_to_chart_data({"accuracy": 0.8})
    assert data == {"准确率": 0.8}
