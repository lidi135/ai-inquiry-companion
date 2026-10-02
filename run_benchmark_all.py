# -*- coding: utf-8 -*-
"""一键跑完 data/ 目录下所有 MCQ 数据集的评测脚本。

自动发现数据文件、逐个数据集跑分，并输出两部分结果：
    1. 各模型 × 各数据集的准确率（accuracy）对比表；
    2. 零样本迁移衰减率矩阵（源域 → 目标域的表现差异）。

用法（在项目根目录执行）：
    python run_benchmark_all.py                          # 默认 deepseek 跑全部数据集
    python run_benchmark_all.py --models deepseek qwen   # 多模型对比
    python run_benchmark_all.py --data-dir data --out benchmark_result.json --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import MODEL_REGISTRY  # noqa: E402
from eval.benchmark import evaluate_mcq, load_mcq_dataset, migration_decay  # noqa: E402


def discover_datasets(data_dir: Path) -> dict[str, Path]:
    """发现数据目录下所有 MCQ JSON，返回 {数据集名: 文件路径}。

    文件名（去扩展名）作为数据集名，按名称排序保证输出稳定。
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"数据目录下没有 JSON 数据文件：{data_dir}")
    return {f.stem: f for f in files}


def run_all(model_keys: list[str], dataset_paths: dict[str, Path], workers: int) -> dict:
    """逐个模型、逐个数据集跑分，返回 {模型: {数据集: metrics_dict}}。

    每个数据集的 metrics_dict 额外写入 "failed" 字段（失败题数）。
    """
    results: dict[str, dict[str, dict]] = {}
    # 数据集只加载一次，供多个模型复用，避免重复 IO/解析
    datasets = {name: load_mcq_dataset(path) for name, path in dataset_paths.items()}
    for mk in model_keys:
        results[mk] = {}
        for ds_name, samples in datasets.items():
            metrics_dict, _, _, failures = evaluate_mcq(mk, samples, max_workers=workers)
            metrics_dict["failed"] = len(failures)
            results[mk][ds_name] = metrics_dict
            status = f"失败 {len(failures)} 题" if failures else "全部成功"
            print(f"  [{mk}] {ds_name}（{len(samples)} 题）完成，{status}", flush=True)
    return results


def _accuracy(metrics: dict) -> float:
    return float(metrics.get("accuracy", 0.0))


def _fmt_cols(header: list[str], rows: list[list[str]]) -> None:
    """以固定列宽、左对齐打印表格。"""
    widths = [max(len(str(cell)) for cell in col) for col in zip(*([header] + rows)) if col]
    # 上面推导的宽度基于列，逐列对齐输出
    def render(line: list[str]) -> str:
        return "  ".join(f"{str(cell):<{w}}" for cell, w in zip(line, widths))
    print(render(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(render(r))


def print_accuracy_table(results: dict, ds_names: list[str]) -> None:
    """打印 模型 × 数据集 准确率对比表。"""
    print("\n一、准确率对比（%）")
    header = ["模型"] + ds_names
    rows: list[list[str]] = []
    for mk, per_ds in results.items():
        label = MODEL_REGISTRY.get(mk).name if MODEL_REGISTRY.get(mk) else mk
        cells = [f"{_accuracy(per_ds[ds]) * 100:.1f}" for ds in ds_names]
        rows.append([label] + cells)
    _fmt_cols(header, rows)


def migration_matrix(results: dict, ds_names: list[str]) -> dict[str, dict[str, float]]:
    """基于各模型在各数据集上的平均准确率，计算 源域→目标域 迁移衰减矩阵。

    衰减率 = (源域准确率 - 目标域准确率) / 源域准确率。
    值越小迁移越稳定；为负表示目标域表现反而更优。
    """
    n_models = len(results)
    avg_acc = {
        ds: sum(_accuracy(results[mk][ds]) for mk in results) / n_models
        for ds in ds_names
    }
    matrix: dict[str, dict[str, float]] = {}
    for src in ds_names:
        matrix[src] = {
            dst: migration_decay(avg_acc[src], avg_acc[dst]) for dst in ds_names
        }
    return matrix


def print_migration(matrix: dict[str, dict[str, float]], ds_names: list[str]) -> None:
    """打印迁移衰减率矩阵（百分比）。"""
    print("\n二、零样本迁移衰减率矩阵（%，行=源域，列=目标域）")
    header = ["源域 ↘ 目标域"] + ds_names
    rows: list[list[str]] = []
    for src in ds_names:
        cells = [f"{matrix[src][dst] * 100:+.1f}" for dst in ds_names]
        rows.append([src] + cells)
    _fmt_cols(header, rows)
    print("说明：对角线为 0；负值表示目标域表现优于源域（迁移无衰减甚至提升）。")


def _print_failures(results: dict, ds_names: list[str]) -> None:
    """若有失败题，打印失败统计；无失败则不输出。"""
    total_failed = 0
    lines: list[str] = []
    for mk, per_ds in results.items():
        for ds in ds_names:
            failed = int(per_ds[ds].get("failed", 0))
            if failed:
                total_failed += failed
                lines.append(f"    - {mk}/{ds}: {failed} 题失败")
    if not total_failed:
        return
    print("\n三、失败统计（失败题按“答错”计入指标）")
    print(f"共 {total_failed} 题失败：")
    for line in lines:
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键跑完 data/ 下全部 MCQ 数据集")
    parser.add_argument("--models", nargs="+", default=["deepseek"],
                        choices=list(MODEL_REGISTRY), help="参与对比的模型（可多个）")
    parser.add_argument("--data-dir", default=str(ROOT / "data"), help="数据集目录")
    parser.add_argument("--out", default="benchmark_result.json", help="结果 JSON 输出路径")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    args = parser.parse_args()

    dataset_paths = discover_datasets(Path(args.data_dir))
    ds_names = list(dataset_paths.keys())
    print(f"数据集：{', '.join(ds_names)}")
    print(f"模型：{', '.join(args.models)}\n")

    results = run_all(args.models, dataset_paths, args.workers)

    print_accuracy_table(results, ds_names)
    print_migration(migration_matrix(results, ds_names), ds_names)

    _print_failures(results, ds_names)

    out = Path(args.out)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存至：{out.resolve()}")


if __name__ == "__main__":
    main()
