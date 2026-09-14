# -*- coding: utf-8 -*-
"""result_table.py — 从实验结果 JSON 生成 Markdown 表格和 LLM 日志

读取 results/exp_llm_dmde_01_data.json，生成：
1. results/result_summary.md  — 实验配置 + 各轮结果 + 汇总统计 + LLM 决策明细
2. results/llm_log_run_{idx}.md — 每次运行的 LLM 完整输入输出日志

用法：
    cd experiments/exp_llm_enhanced_dmde/exp_llm_dmde_01
    python result_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DATA_FILE = RESULTS_DIR / "exp_llm_dmde_01_data.json"


def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# 表 1：实验配置
# ──────────────────────────────────────────────────────────────

def gen_config_table(meta: dict, scenario: dict) -> str:
    sp = meta.get("solver_params", {})
    llm = meta.get("llm_config", {})
    lines = [
        "## 表 1：实验配置\n",
        "| 参数 | 值 |",
        "|------|-----|",
        f"| 场景 | {scenario['name']} ({scenario['n_uavs']}U/{scenario['n_targets']}T) |",
        f"| pop_size | {sp.get('pop_size', '—')} |",
        f"| max_generations | {sp.get('max_generations', '—')} |",
        f"| zeta | {sp.get('zeta', '—')} |",
        f"| delta | {sp.get('delta', '—')} |",
        f"| LLM interval | 100 |",
        f"| LLM model | {llm.get('model', '—')} |",
        f"| reasoning_effort | {llm.get('reasoning_effort', 'none')} |",
        f"| N_RUNS | {meta.get('n_runs', '—')} |",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 表 2：各轮运行结果
# ──────────────────────────────────────────────────────────────

def gen_runs_table(scenario: dict, llm_decisions: dict | None) -> str:
    runs = scenario["runs"]
    lines = [
        "## 表 2：各轮运行结果\n",
        "| Run | Best Fitness | 可行性 | 违反量 | LLM调用次数 | LLM成功次数 | 耗时(s) |",
        "|-----|-------------|--------|--------|------------|------------|---------|",
    ]
    for i, r in enumerate(runs):
        feasible = "✅" if r["extra"].get("is_feasible", False) else "❌"
        violation = r["extra"].get("total_violation", 0)
        elapsed = r.get("elapsed_seconds", 0)

        # LLM stats
        llm_calls = 0
        llm_success = 0
        if llm_decisions and str(i) in llm_decisions:
            decisions = llm_decisions[str(i)]
            llm_calls = len(decisions)
            llm_success = sum(1 for d in decisions if d.get("decision", {}).get("cr") is not None)

        lines.append(
            f"| {i+1} | {r['best_fitness']:.1f} | {feasible} | {violation:.1f} "
            f"| {llm_calls} | {llm_success} | {elapsed:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 表 3：汇总统计
# ──────────────────────────────────────────────────────────────

def gen_summary_table(meta: dict, scenario: dict, llm_decisions: dict | None) -> str:
    m = scenario["metrics"]
    runs = scenario["runs"]
    n_runs = len(runs)

    feasible_count = sum(1 for r in runs if r["extra"].get("is_feasible", False))
    mean_time = sum(r.get("elapsed_seconds", 0) for r in runs) / max(n_runs, 1)

    # LLM stats
    total_llm_calls = 0
    total_llm_success = 0
    if llm_decisions:
        for run_decisions in llm_decisions.values():
            total_llm_calls += len(run_decisions)
            total_llm_success += sum(
                1 for d in run_decisions if d.get("decision", {}).get("cr") is not None
            )
    llm_success_rate = (total_llm_success / total_llm_calls * 100) if total_llm_calls > 0 else 0

    lines = [
        "## 表 3：汇总统计\n",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| Best fitness | {m['best_fitness']:.2f} |",
        f"| Mean fitness | {m['mean_fitness']:.2f} ± {m['std_fitness']:.2f} |",
        f"| 可行解率 | {100 * feasible_count / n_runs:.0f}% ({feasible_count}/{n_runs}) |",
        f"| 平均耗时 | {mean_time:.1f}s |",
        f"| LLM 总调用 | {total_llm_calls} 次 |",
        f"| LLM 成功率 | {llm_success_rate:.0f}% |",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 表 4：LLM 决策明细
# ──────────────────────────────────────────────────────────────

def gen_llm_decisions_table(llm_decisions: dict | None) -> str:
    if not llm_decisions:
        return "## 表 4：LLM 决策明细\n\n*无 LLM 决策记录。*\n"

    lines = [
        "## 表 4：LLM 决策明细\n",
        "| Run | Gen | CR | 耗时(s) | 状态 | 理由 |",
        "|-----|-----|-----|---------|------|------|",
    ]
    for run_idx_str in sorted(llm_decisions.keys(), key=int):
        run_idx = int(run_idx_str)
        for d in llm_decisions[run_idx_str]:
            gen = d.get("generation", "?")
            decision = d.get("decision", {})
            cr = decision.get("cr")
            duration = d.get("duration", 0)
            reasoning = decision.get("reasoning", "") or d.get("reasoning", "")

            if cr is not None:
                status = "✅"
                cr_str = str(cr)
            else:
                status = "❌"
                cr_str = "—"

            # Truncate reasoning for table
            reason_short = " ".join(reasoning.split())
            if len(reason_short) > 60:
                reason_short = reason_short[:57] + "..."

            lines.append(
                f"| {run_idx+1} | {gen} | {cr_str} | {duration:.1f} | {status} | {reason_short} |"
            )
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# LLM 日志文件（每次运行一个 .md）
# ──────────────────────────────────────────────────────────────

def gen_llm_log(run_idx: int, decisions: list[dict], meta: dict) -> str:
    llm_cfg = meta.get("llm_config", {})
    model = llm_cfg.get("model", "unknown")
    reasoning_effort = llm_cfg.get("reasoning_effort", "none")

    lines = [
        f"# LLM Decision Log — Run {run_idx + 1}\n",
    ]

    for d in decisions:
        gen = d.get("generation", "?")
        decision = d.get("decision", {})
        duration = d.get("duration", 0)
        reasoning = decision.get("reasoning", "") or d.get("reasoning", "")
        cr = decision.get("cr")
        status = "✅ success" if cr is not None else "❌ failed"

        llm_input = d.get("llm_input", {})
        messages = llm_input.get("messages", [])

        lines.append(f"## Gen {gen}\n")

        # Input messages
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                lines.append("### Input (System Prompt)")
            elif role == "user":
                lines.append("### Input (User Prompt)")
            else:
                lines.append(f"### Input ({role})")
            lines.append(f"```\n{content}\n```\n")

        # Output
        lines.append("### Output")
        raw_output = d.get("llm_raw_output", "")
        if raw_output:
            # Try to format as JSON
            try:
                parsed = json.loads(raw_output)
                lines.append(f"```json\n{json.dumps(parsed, ensure_ascii=False, indent=2)}\n```\n")
            except (json.JSONDecodeError, TypeError):
                lines.append(f"```\n{raw_output}\n```\n")
        else:
            lines.append("*(empty)*\n")

        # Metadata
        lines.append("### Metadata")
        lines.append(f"- Model: {model}")
        lines.append(f"- reasoning_effort: {reasoning_effort}")
        lines.append(f"- Duration: {duration:.1f}s")
        lines.append(f"- Status: {status}")
        if cr is not None:
            lines.append(f"- CR chosen: {cr}")
        if reasoning:
            lines.append(f"- Reasoning: {reasoning}")
        lines.append("\n---\n")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    if not DATA_FILE.exists():
        print(f"❌ 数据文件不存在: {DATA_FILE}")
        print("请先运行实验: python run.py")
        sys.exit(1)

    data = load_data(DATA_FILE)
    meta = data.get("meta", {})
    scenario = data["scenarios"][0]
    llm_decisions = data.get("llm_decisions")

    # 1. 生成 result_summary.md
    parts = [
        "# 实验结果汇总：LLM 增强 DMDE (exp_llm_dmde_01)\n",
        f"*生成时间: {meta.get('timestamp', '—')}*\n",
        gen_config_table(meta, scenario),
        gen_runs_table(scenario, llm_decisions),
        gen_summary_table(meta, scenario, llm_decisions),
        gen_llm_decisions_table(llm_decisions),
    ]
    summary_path = RESULTS_DIR / "result_summary.md"
    summary_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"✅ 结果汇总表: {summary_path}")

    # 2. 生成 LLM 日志文件
    if llm_decisions:
        for run_idx_str, decisions in llm_decisions.items():
            run_idx = int(run_idx_str)
            log_content = gen_llm_log(run_idx, decisions, meta)
            log_path = RESULTS_DIR / f"llm_log_run_{run_idx + 1}.md"
            log_path.write_text(log_content, encoding="utf-8")
            print(f"✅ LLM 日志: {log_path}")
    else:
        print("ℹ️  无 LLM 决策记录，跳过日志生成。")


if __name__ == "__main__":
    main()
