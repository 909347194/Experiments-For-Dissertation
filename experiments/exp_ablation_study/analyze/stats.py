# -*- coding: utf-8 -*-
"""统计计算：描述性统计 + 假设检验 + 收敛分析。"""

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_stats(results: list[dict]) -> dict:
    """计算单组实验的描述性统计。"""
    if not results:
        return {}
    fitness = np.array([r["best_fitness"] for r in results])
    times = np.array([r["total_time"] for r in results])
    llm_times = np.array([r.get("llm_time", 0) for r in results])
    llm_init_times = np.array([r.get("llm_init_time", 0) for r in results])
    llm_cr_times = np.array([r.get("llm_cr_time", 0) for r in results])
    llm_calls = np.array([r.get("llm_call_count", 0) for r in results])
    dmde_times = np.array([r.get("dmde_time", 0) for r in results])
    std_val = float(fitness.std(ddof=1)) if len(fitness) > 1 else 0.0
    return {
        "n_runs": len(results),
        "best": round(float(fitness.min()), 2),
        "mean": round(float(fitness.mean()), 2),
        "std": round(std_val, 2),
        "median": round(float(np.median(fitness)), 2),
        "total_time_mean": round(float(times.mean()), 2),
        "dmde_time_mean": round(float(dmde_times.mean()), 2),
        "llm_time_mean": round(float(llm_times.mean()), 2),
        "llm_init_time_mean": round(float(llm_init_times.mean()), 2),
        "llm_cr_time_mean": round(float(llm_cr_times.mean()), 2),
        "llm_calls_mean": round(float(llm_calls.mean()), 1),
        "fitness_array": fitness,
        "_raw": results,
    }


def mannwhitney_test(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U 秩和检验（非配对双侧）。"""
    if not HAS_SCIPY:
        return -1.0
    if len(a) < 5 or len(b) < 5:
        return -1.0
    try:
        _, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        return round(float(p), 4)
    except Exception:
        return -1.0


def p_mark(p: float) -> str:
    """根据 p 值返回显著性标记。"""
    if p < 0:
        return "---"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ── 收敛曲线分析 ──────────────────────────────────────────

def extract_convergence_curves(results: list[dict]) -> list[list[float]]:
    curves = []
    for r in results:
        curve = r.get("convergence_curve", [])
        if curve:
            curves.append(curve)
    return curves


def extract_cr_histories(results: list[dict]) -> list[list[float]]:
    histories = []
    for r in results:
        cr = r.get("cr_history", [])
        if cr:
            histories.append(cr)
    return histories


def compute_convergence_stats(curves: list[list[float]]) -> dict:
    if not curves:
        return {}
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "n_runs": len(curves),
        "length": min_len,
    }


def find_convergence_gen(curve: list[float], threshold: float = 0.95) -> int:
    if not curve:
        return -1
    final_best = curve[-1]
    target = final_best * threshold
    for i, v in enumerate(curve):
        if v <= target:
            return i
    return len(curve) - 1


# ── LLM 决策分析 ──────────────────────────────────────────

def extract_llm_decisions(results: list[dict]) -> list[dict]:
    all_decisions = []
    for r in results:
        decisions = r.get("llm_decisions", [])
        for d in decisions:
            d["_run_seed"] = r.get("seed")
            all_decisions.append(d)
    return all_decisions


def summarize_llm_decisions(results: list[dict]) -> dict:
    decisions = extract_llm_decisions(results)
    if not decisions:
        return {}
    modules = {}
    for d in decisions:
        mod = d.get("module", "unknown")
        if mod not in modules:
            modules[mod] = {"count": 0, "durations": [], "cr_values": []}
        modules[mod]["count"] += 1
        modules[mod]["durations"].append(d.get("duration", 0))
        if mod == "search_controller":
            cr = d.get("parsed_decision", {}).get("cr", 0)
            if cr:
                modules[mod]["cr_values"].append(cr)
    summary = {}
    for mod, data in modules.items():
        durs = np.array(data["durations"])
        summary[mod] = {
            "total_calls": data["count"],
            "avg_duration": round(float(durs.mean()), 3) if len(durs) else 0,
            "total_duration": round(float(durs.sum()), 2),
        }
        if data["cr_values"]:
            crs = np.array(data["cr_values"])
            summary[mod]["cr_mean"] = round(float(crs.mean()), 4)
            summary[mod]["cr_std"] = round(float(crs.std()), 4)
            summary[mod]["cr_min"] = round(float(crs.min()), 4)
            summary[mod]["cr_max"] = round(float(crs.max()), 4)
    return summary