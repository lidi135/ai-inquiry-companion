# -*- coding: utf-8 -*-
"""评测结果可视化。

提供把 ClassificationMetrics 汇总结果渲染为图表的工具。
matplotlib 为懒加载依赖：仅在实际绘图时导入，便于离线环境（如仅跑数据准备逻辑）
或未安装 matplotlib 的 32 位 Python 环境仍可导入本模块进行纯数据准备。
"""
from __future__ import annotations

from typing import Any

# 用于条形图展示的汇总指标及其中文标签
_CHART_KEYS = {
    "accuracy": "准确率",
    "macro_f1": "宏观 F1",
    "micro_f1": "微观 F1",
    "weighted_f1": "加权 F1",
}


def metrics_to_chart_data(metrics: dict[str, Any]) -> dict[str, float]:
    """提取用于绘图的汇总指标（纯数据准备，无第三方依赖，可离线测试）。

    Args:
        metrics: compute_metrics(...).to_dict() 的结果字典。

    Returns:
        {中文标签: 数值} 的有序字典。
    """
    result: dict[str, float] = {}
    for key, label in _CHART_KEYS.items():
        if key in metrics:
            result[label] = float(metrics[key])
    return result


def plot_metrics_bar(
    metrics: dict[str, Any],
    title: str = "模型评测指标",
    save_path: str | None = None,
):
    """绘制指标条形图（懒加载 matplotlib）。

    Args:
        metrics:   汇总指标字典。
        title:     图表标题。
        save_path: 可选保存路径（.png）。

    Returns:
        matplotlib Figure。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = metrics_to_chart_data(metrics)
    labels = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#2e5485")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
