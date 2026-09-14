# -*- coding: utf-8 -*-
"""result_table.py — 从实验结果 JSON 生成 Markdown 汇总表（DMDE 基线版）

读取本实验的结果 JSON（``results/*_data.json``），生成
``results/result_summary.md``：

    表 1：实验配置
    表 2：各轮运行结果
    表 3：汇总统计

与 LLM 增强版（exp_llm_dmde_0x/result_table.py）的区别：
    DMDE 基线没有 LLM 决策日志，因此不生成「LLM 决策明细」表与
    ``llm_log_run_*.md``。

用法：
    # 默认：自动查找本目录 results/ 下最新的 *_data.json
    python result_table.py

    # 指定数据 / 输出文件
    python result_table.py --data results/exp_dmde_02_data.json \
                           --out  results/result_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def find_default_data() -> Path | None:
    """自动查找本目录 results/ 下最新的 *_data.json。"""
    if not RESULTS_DIR.exists():
        return None
    jsons = sorted(RESULTS_DIR.glob("*_data.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return jsons[0] if jsons else None


def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# 表 1：实验配置
# ──────────────────────────────────────────────────────────────

def gen_config_table(meta: dict, scenario: dict) -> str:
    sp = meta.get("solver_params", {})
    lines = [
        "## 表 1：实验配置\n",
        "| 参数 | 值 |",
        "|------|-----|",
        f"| 场景 | {scenario['name']} ({scenario['n_uavs']}U/{scenario['n_targets']}T) |",
        "| 算法 | DMDE |",
        f"| pop_size | {sp.get('pop_size', '—')} |",
        f"| max_generations | {sp.get('max_generations', '—')} |",
        f"| zeta | {sp.get('zeta', '—')} |",
        f"| delta | {sp.get('delta', '—')} |",
        f"| 约束配置 | {meta.get('constraint_desc', '—')} |",
        f"| N_RUNS | {meta.get('n_runs', '—')} |",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 表 2：各轮运行结果
# ──────────────────────────────────────────────────────────────

def gen_runs_table(scenario: dict) -> str:
    runs = scenario["runs"]
    lines = [
        "## 表 2：各轮运行结果\n",
        "| Run | Best Fitness | 可行性 | 违反量 | 耗时(s) |",
        "|-----|-------------|--------|--------|---------|",
    ]
    for i, r in enumerate(runs):
        feasible = "✅" if r["extra"].get("is_feasible", False) else "❌"
        violation = r["extra"].get("total_violation", 0)
        elapsed = r.get("elapsed_seconds", 0)
        lines.append(
            f"| {i + 1} | {r['best_fitness']:.1f} | {feasible} | "
            f"{violation:.1f} | {elapsed:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 表 3：汇总统计
# ──────────────────────────────────────────────────────────────

def gen_summary_table(scenario: dict) -> str:
    m = scenario["metrics"]
    runs = scenario["runs"]
    n_runs = max(len(runs), 1)

    feasible_count = sum(1 for r in runs if r["extra"].get("is_feasible", False))
    mean_time = sum(r.get("elapsed_seconds", 0) for r in runs) / n_runs

    lines = [
        "## 表 3：汇总统计\n",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| Best fitness | {m['best_fitness']:.2f} |",
        f"| Mean fitness | {m['mean_fitness']:.2f} ± {m['std_fitness']:.2f} |",
        f"| Worst fitness | {m['worst_fitness']:.2f} |",
        f"| 可行解率 | {100 * feasible_count / n_runs:.0f}% "
        f"({feasible_count}/{len(runs)}) |",
        f"| 平均耗时 | {mean_time:.1f}s |",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="从实验结果 JSON 生成 Markdown 汇总表（DMDE 基线版）"
    )
    ap.add_argument("--data", default=None,
                    help="实验结果 JSON 路径（默认自动查找 results/*_data.json）")
    ap.add_argument("--out", default=None,
                    help="输出 Markdown 路径（默认 results/result_summary.md）")
    args = ap.parse_args()

    data_path = Path(args.data) if args.data else find_default_data()
    if not data_path or not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行实验: python run.py")
        sys.exit(1)

    data = load_data(data_path)
    meta = data.get("meta", {})
    scenario = data["scenarios"][0]

    parts = [
        f"# 实验结果汇总：DMDE 基线 ({data_path.stem})\n",
        f"*生成时间: {meta.get('timestamp', '—')}*\n",
        gen_config_table(meta, scenario),
        gen_runs_table(scenario),
        gen_summary_table(scenario),
    ]

    out_path = Path(args.out) if args.out else RESULTS_DIR / "result_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"✅ 结果汇总表: {out_path}")


if __name__ == "__main__":
    main()
