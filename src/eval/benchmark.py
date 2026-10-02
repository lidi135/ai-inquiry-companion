# -*- coding: utf-8 -*-
"""多模型评测与零样本迁移能力评估。

本模块提供一个与厂商无关的评测框架：
    1. 对多个模型（DeepSeek / Qwen / GLM / GPT）在相同数据集上跑分；
    2. 支持零样本迁移：模型不经过任何微调，直接在目标域数据集上评测，
       以"源域 -> 目标域"的表现差异量化迁移能力。

样本格式（选择题 MCQ）：
    {"question": "...", "options": {"A": "..", "B": "..", "C": "..", "D": ".."},
     "answer": "A", "category": "内科"}

运行方式：
    python -m eval.benchmark --models deepseek qwen glm --dataset cases.json
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from core.config import MODEL_REGISTRY, get_model_config
from core.role_engine import chat_completion, create_client

from .metrics import compute_metrics


@dataclass
class EvalSample:
    """一条评测样本。"""
    question: str
    options: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    category: str = ""


def load_mcq_dataset(path: Path | str) -> list[EvalSample]:
    """从 JSON 文件加载 MCQ 数据集。

    支持两种结构：
        - 顶层为列表：[{...}, {...}]
        - 顶层为字典且含 "data" 键：{"data": [...]}

    Args:
        path: 数据集 JSON 路径。

    Returns:
        EvalSample 列表。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("data") if isinstance(data, dict) and "data" in data else data
    samples: list[EvalSample] = []
    for it in items or []:
        samples.append(EvalSample(
            question=str(it.get("question", "")),
            options={str(k): str(v) for k, v in (it.get("options") or {}).items()},
            answer=str(it.get("answer", "")),
            category=str(it.get("category", "")),
        ))
    return samples


def parse_mcq_answer(response: str) -> str:
    """从模型回复中解析选择题答案（A/B/C/D）。

    解析策略（按优先级）：
        1. 匹配形如 "答案：A" / "Answer: A" / "选 A" 的模式；
        2. 匹配句首独立出现的 "(A)" / "A." 等；
        3. 匹配回复中出现的单个大写字母 A-D。

    Args:
        response: 模型原始回复文本。

    Returns:
        解析出的选项字母（大写），失败返回空字符串。
    """
    if not response:
        return ""
    text = response.strip()
    # 策略 1：显式答案声明
    m = re.search(r"(?:答案|正确选项|Answer|answer)[^\w]*([A-D])", text)
    if m:
        return m.group(1)
    # 策略 2：句首 (A) 或 A.
    m = re.match(r"^\s*[\(（]?([A-D])[\)）\.．、]", text)
    if m:
        return m.group(1)
    # 策略 3：第一个出现的孤立大写字母 A-D
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    return ""


def _format_question(sample: EvalSample) -> str:
    """把样本格式化为 Prompt。"""
    opts = "\n".join(f"{k}. {v}" for k, v in sample.options.items())
    return f"请从下列选项中选择唯一正确答案，只输出选项字母（A/B/C/D），不要解释。\n问题：{sample.question}\n{opts}\n答案："


def _evaluate_one(client: Any, model_name: str, temperature: float, sample: EvalSample):
    """对单条样本发起一次推理，返回 (真实答案, 预测答案)。"""
    resp = chat_completion(
        client,
        model=model_name,
        messages=[{"role": "user", "content": _format_question(sample)}],
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    return sample.answer, parse_mcq_answer(content)


def _evaluate_one_safe(client: Any, model_name: str, temperature: float, sample: EvalSample):
    """带容错的单次推理，返回 (结果, 错误信息)。

    捕获网络、额度、限流等异常：失败时预测答案记为空串（计为答错），
    错误信息为非空字符串，保证单题失败不会中断整轮评测。
    """
    try:
        return _evaluate_one(client, model_name, temperature, sample), ""
    except Exception as e:  # noqa: BLE001 —— 单题异常不中断评测
        return (sample.answer, ""), f"{type(e).__name__}: {e}"


def evaluate_mcq(
    model_key: str,
    samples: Sequence[EvalSample],
    *,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_workers: int = 4,
) -> tuple[dict, list[str], list[str], list[dict]]:
    """对单个模型在 MCQ 数据集上评测。

    Args:
        model_key:   模型注册名。
        samples:     评测样本。
        api_key:     API Key（缺省从环境变量读取）。
        model:       覆盖默认模型标识。
        temperature: 采样温度（评测用 0 保证确定性）。
        max_workers: 并发线程数（>=2 时并行发起推理，显著缩短多样本评测耗时）。

    Returns:
        (metrics_dict, y_true, y_pred, failures) 四元组，
        其中 failures 为失败样本列表 [{index, error}]，失败样本预测记为答错。
    """
    cfg = get_model_config(model_key)
    client = create_client(model_key, api_key=api_key)
    model_name = model or cfg.model

    if len(samples) <= 1 or max_workers <= 1:
        results = [_evaluate_one_safe(client, model_name, temperature, s) for s in samples]
    else:
        # 并发调用，保持样本顺序（ThreadPoolExecutor.map 保证顺序）
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(
                lambda s: _evaluate_one_safe(client, model_name, temperature, s),
                samples,
            ))

    y_true = [r[0][0] for r in results]
    y_pred = [r[0][1] for r in results]
    failures = [
        {"index": i, "error": err} for i, (_, err) in enumerate(results) if err
    ]
    m = compute_metrics(y_true, y_pred)
    return m.to_dict(), y_true, y_pred, failures


def migration_decay(source_score: float, target_score: float) -> float:
    """计算零样本迁移衰减率。

    衰减率 = (源域得分 - 目标域得分) / 源域得分。
    值越小说明迁移越稳定；为负表示目标域反而更优。

    Args:
        source_score: 源域指标（如 accuracy）。
        target_score: 目标域指标。

    Returns:
        迁移衰减率（比例）。
    """
    if source_score == 0:
        return 0.0
    return (source_score - target_score) / source_score


def run_benchmark(
    model_keys: Sequence[str],
    dataset_paths: dict[str, Path | str],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """运行多模型 × 多数据集的评测，得到对比与迁移矩阵。

    Args:
        model_keys:    模型注册名列表。
        dataset_paths: {数据集名: 数据文件路径}。
        api_key:       API Key。

    Returns:
        嵌套结果：{模型名: {数据集名: metrics_dict}}。
    """
    results: dict[str, Any] = {}
    # 数据集只需加载一次，供多个模型复用，避免重复 IO/解析
    datasets = {name: load_mcq_dataset(path) for name, path in dataset_paths.items()}
    for mk in model_keys:
        results[mk] = {}
        for ds_name, samples in datasets.items():
            metrics_dict, _, _, _ = evaluate_mcq(mk, samples, api_key=api_key)
            results[mk][ds_name] = metrics_dict
    return results


def save_results(results: dict[str, Any], out_path: Path | str) -> None:
    """把评测结果写入 JSON 文件。"""
    Path(out_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":  # 命令行入口
    import argparse

    parser = argparse.ArgumentParser(description="多模型医疗 MCQ 评测")
    parser.add_argument("--models", nargs="+", default=["deepseek", "qwen", "glm"],
                        choices=list(MODEL_REGISTRY), help="参与对比的模型")
    parser.add_argument("--dataset", required=True, help="数据集 JSON 路径")
    parser.add_argument("--out", default="benchmark_result.json", help="结果输出路径")
    args = parser.parse_args()

    ds_name = Path(args.dataset).stem
    results = run_benchmark(args.models, {ds_name: args.dataset})
    save_results(results, args.out)
    print(json.dumps(results, ensure_ascii=False, indent=2))
