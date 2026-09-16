# -*- coding: utf-8 -*-
"""analyze.py — 消融实验结果分析

读取 8 组实验结果，生成对比表 + 图表。

用法：python analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SCENARIOS = {
    "S1": "S1_balanced_N10_M10",
    "S2": "S2_srp_N10_M20",
}
CONFIGS = {
    "A0": "A0_dmde",
    "A1": "A1_cr_control",
    "A2": "A2_pop_init",
    "A3": "A3_full",
}


def load_results(scenario_dir: str, config_dir: str) -> list[dict]:
    """加载单组实验结果。"""
    path = SCRIPT_DIR / scenario_dir / config_dir / "results" / "ablation_results.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def compute_stats(results: list[dict]) -> dict:
    """计算统计指标。"""
    if not results:
        return {}
    fitness = np.array([r["best_fitness"] for r in results])
    times = np.array([r["total_time"] for r in results])
    llm_times = np.array([r.get("llm_time", 0) for r in results])
    llm_calls = np.array([r.get("llm_call_count", 0) for r in results])

    return {
        "n_runs": len(results),
        "best": round(float(fitness.min()), 2),
        "mean": round(float(fitness.mean()), 2),
        "std": round(float(fitness.std()), 2),
        "median": round(float(np.median(fitness)), 2),
        "total_time_mean": round(float(times.mean()), 2),
        "llm_time_mean": round(float(llm_times.mean()), 2),
        "llm_calls_mean": round(float(llm_calls.mean()), 1),
    }


def generate_summary_table(all_stats: dict) -> str:
    """生成 Markdown 格式的汇总表。"""
    lines = [
        "| 场景 | 配置 | Best | Mean ± Std | Median | 总耗时(s) | LLM耗时(s) | LLM调用次数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s_key, s_dir in SCENARIOS.items():
        for c_key, c_dir in CONFIGS.items():
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats:
                lines.append(f"| {s_key} | {c_key} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {s_key} | {c_key} "
                f"| {stats['best']:.2f} "
                f"| {stats['mean']:.2f} ± {stats['std']:.2f} "
                f"| {stats['median']:.2f} "
                f"| {stats['total_time_mean']:.1f} "
                f"| {stats['llm_time_mean']:.1f} "
                f"| {stats['llm_calls_mean']:.0f} |"
            )
    return "\n".join(lines)


def main():
    all_stats = {}

    # 加载所有结果
    for s_key, s_dir in SCENARIOS.items():
        for c_key, c_dir in CONFIGS.items():
            results = load_results(s_dir, c_dir)
            stats = compute_stats(results)
            all_stats[f"{s_key}_{c_key}"] = stats

    # 生成汇总表
    table = generate_summary_table(all_stats)
    print("\n## 消融实验结果\n")
    print(table)

    # 保存
    out = FIGURES_DIR / "summary_table.md"
    with open(out, "w") as f:
        f.write("## 消融实验结果\n\n")
        f.write(table)
        f.write("\n")
    print(f"\n汇总表已保存: {out}")

    # 保存 CSV
    csv_out = FIGURES_DIR / "summary_table.csv"
    with open(csv_out, "w") as f:
        f.write("scenario,config,best,mean,std_median,total_time,llm_time,llm_calls\n")
        for s_key, _ in SCENARIOS.items():
            for c_key, _ in CONFIGS.items():
                stats = all_stats.get(f"{s_key}_{c_key}", {})
                if not stats:
                    continue
                f.write(f"{s_key},{c_key},{stats['best']},{stats['mean']},"
                        f"{stats['std']},{stats['total_time_mean']},"
                        f"{stats['llm_time_mean']},{stats['llm_calls_mean']}\n")
    print(f"CSV 已保存: {csv_out}")


if __name__ == "__main__":
    main()