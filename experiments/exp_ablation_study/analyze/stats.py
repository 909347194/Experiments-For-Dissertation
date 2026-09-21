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

def curve_gens(run: dict) -> list[int]:
    """返回某 run 收敛曲线的**真实代数轴**（与 ``convergence_curve`` 等长）。

    ``convergence_curve[i]`` 与 ``generation_records[i]['gen']`` 一一对应。
    记录粒度因配置而异：A0（DMDE）逐代记录 → 1001 点；LLM 配置每
    ``max(1, max_generations // 100)`` 代记录一次 → 约 101 点。
    因此**绝不能用采样下标充当横轴**（否则 LLM 曲线会被横向压缩/拉伸）。
    """
    curve = run.get("convergence_curve", [])
    recs = run.get("generation_records", [])
    if recs and len(recs) == len(curve):
        return [int(x.get("gen", i)) for i, x in enumerate(recs)]
    return list(range(len(curve)))


def extract_convergence_curves(results: list[dict]) -> list[tuple[list[int], list[float]]]:
    """返回 ``[(gens, curve), ...]``，``gens`` 为该 run 的真实代数轴。"""
    curves = []
    for r in results:
        curve = r.get("convergence_curve", [])
        if curve:
            curves.append((curve_gens(r), list(curve)))
    return curves


def extract_cr_histories(results: list[dict]) -> list[list[float]]:
    histories = []
    for r in results:
        cr = r.get("cr_history", [])
        if cr:
            histories.append(cr)
    return histories


def compute_convergence_stats(curves: list[tuple[list[int], list[float]]]) -> dict:
    """对同一配置的多 run 曲线求逐代均值/标准差。

    Args:
        curves: ``extract_convergence_curves`` 的输出，即 ``[(gens, curve), ...]``。
            同一配置内各 run 的采样粒度一致（同为逐代或同为每 10 代）。

    Returns:
        含 ``gens``（真实代数轴）的统计字典；``gens`` 与 ``mean`` 等长。
    """
    if not curves:
        return {}
    min_len = min(len(c) for _, c in curves)
    arr = np.array([c[:min_len] for _, c in curves])
    gens = [int(g) for g in curves[0][0][:min_len]]
    return {
        "gens": gens,
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "n_runs": len(curves),
        "length": min_len,
    }


def compute_convergence_gens(all_stats: dict, thresholds: list[float] = None) -> dict:
    """计算各配置达到 X% 改进所需代数（跨 run 取中位数）。

    语义：
        improvement = initial_best - final_best  (cost 场景下为正)
        target = final_best + improvement * (1 - t)
        t = 0.90 → 需要达到 90% 改进（fitness 已下降到 initial - 0.90*(initial-final)）
        t = 0.95 → 95% 改进
        t = 0.99 → 99% 改进
    注意：cost 场景下 t 越大越难（需要收敛到离 final 更近），但**目标值变大**。

    Args:
        all_stats: compute_stats 输出
        thresholds: 目标阈值列表，如 [0.90, 0.95, 0.99]

    Returns:
        {"S1_A0": {0.90: gen, 0.95: gen, 0.99: gen}, ...}
    """
    if thresholds is None:
        thresholds = [0.90, 0.95, 0.99]
    from .constants import SCENARIOS, CONFIGS
    result = {}
    for s_key in SCENARIOS:
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            if not raw:
                continue
            # 收集每个 run 的收敛代数
            gen_at = {t: [] for t in thresholds}
            for r in raw:
                curve = r.get("convergence_curve", [])
                if len(curve) < 2:
                    continue
                gens = curve_gens(r)
                initial_best = curve[0]
                final_best = curve[-1]
                improvement = initial_best - final_best
                # 无改进（卡点、震荡、或初始种群全部不可行，fitness=inf）：
                # 无法定义 X% 改进 → 记 -1
                if not np.isfinite(improvement) or improvement <= 0:
                    for t in thresholds:
                        gen_at[t].append(-1)
                    continue
                for t in thresholds:
                    # 目标：fitness 已下降到 initial - t*improvement
                    target = final_best + improvement * (1 - t)
                    gen = gens[-1]  # 默认最后一代
                    for i, v in enumerate(curve):
                        if v <= target:
                            gen = gens[i]  # 用**真实代数**，不是采样下标
                            break
                    gen_at[t].append(gen)
            # 取中位数（忽略 -1 卡点）
            result[f"{s_key}_{c_key}"] = {
                t: int(np.median([g for g in gens if g >= 0])) if any(g >= 0 for g in gens) else -1
                for t, gens in gen_at.items()
            }
    return result


# ── LLM 决策分析 ──────────────────────────────────────────

def extract_llm_decisions(results: list[dict]) -> list[dict]:
    all_decisions = []
    for r in results:
        decisions = r.get("llm_decisions", [])
        for d in decisions:
            # 浅拷贝避免污染原始数据（原始 dict 可能被多次读取）
            all_decisions.append({**d, "_run_seed": r.get("seed")})
    return all_decisions


def summarize_llm_decisions(results: list[dict]) -> dict:
    decisions = extract_llm_decisions(results)
    if not decisions:
        return {}
    modules = {}
    for d in decisions:
        mod = d.get("module", "unknown")
        if mod not in modules:
            modules[mod] = {"count": 0, "durations": [], "cr_values": [],
                           "f_values": [], "gmr_modes": []}
        modules[mod]["count"] += 1
        modules[mod]["durations"].append(d.get("duration", 0))
        if mod == "search_controller":
            parsed = d.get("parsed_decision", {})
            cr = parsed.get("cr")
            if cr is not None:
                modules[mod]["cr_values"].append(cr)
            f_val = parsed.get("f")
            if f_val is not None:
                modules[mod]["f_values"].append(f_val)
            gmr = parsed.get("gmr_mode", "auto")
            modules[mod]["gmr_modes"].append(gmr)
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
        if data["f_values"]:
            fvs = np.array(data["f_values"])
            summary[mod]["f_mean"] = round(float(fvs.mean()), 4)
            summary[mod]["f_std"] = round(float(fvs.std()), 4)
            summary[mod]["f_min"] = round(float(fvs.min()), 4)
            summary[mod]["f_max"] = round(float(fvs.max()), 4)
        if data["gmr_modes"]:
            modes = data["gmr_modes"]
            total = len(modes)
            summary[mod]["gmr_mode_counts"] = {
                m: modes.count(m) for m in set(modes)
            }
            summary[mod]["gmr_mode_pcts"] = {
                m: round(modes.count(m) / total * 100, 1) for m in set(modes)
            }
    return summary


# ── F 轨迹分析 ────────────────────────────────────────────

def extract_f_histories(results: list[dict]) -> list[tuple[list[int], list[float]]]:
    """提取各 run 的 F 值轨迹。

    优先使用 f_override（LLM 直接指定），回退到 f_scale（公式推导）。
    Returns:
        [(gens, f_values), ...]
    """
    histories = []
    for r in results:
        recs = r.get("generation_records", [])
        if not recs:
            continue
        gens = []
        f_vals = []
        for rec in recs:
            gens.append(rec.get("gen", 0))
            # f_override 为 LLM 直接指定值，f_scale 为公式推导值
            f_val = rec.get("f_override")
            if f_val is None:
                f_val = rec.get("f_scale", 0.5)
            f_vals.append(f_val)
        if gens:
            histories.append((gens, f_vals))
    return histories


def extract_f_override_histories(results: list[dict]) -> list[tuple[list[int], list[float]]]:
    """提取各 run 中 LLM 实际覆写 F 的代数和值（仅 f_override 非 None 的代）。

    Returns:
        [(gens, f_values), ...]  gens/f_values 等长，只含 LLM 覆写代。
    """
    histories = []
    for r in results:
        recs = r.get("generation_records", [])
        if not recs:
            continue
        gens = []
        f_vals = []
        for rec in recs:
            f_val = rec.get("f_override")
            if f_val is not None:
                gens.append(rec.get("gen", 0))
                f_vals.append(f_val)
        if gens:
            histories.append((gens, f_vals))
    return histories


def compute_f_stats(results: list[dict]) -> dict:
    """计算 F 值的统计信息。"""
    histories = extract_f_histories(results)
    if not histories:
        return {}
    # 收集所有 F 值
    all_f = []
    for _, f_vals in histories:
        all_f.extend(f_vals)
    if not all_f:
        return {}
    arr = np.array(all_f)
    # LLM 覆写统计
    override_histories = extract_f_override_histories(results)
    all_override_f = []
    for _, f_vals in override_histories:
        all_override_f.extend(f_vals)
    result = {
        "f_mean": round(float(arr.mean()), 4),
        "f_std": round(float(arr.std()), 4),
        "f_min": round(float(arr.min()), 4),
        "f_max": round(float(arr.max()), 4),
        "f_median": round(float(np.median(arr)), 4),
        "n_runs": len(histories),
    }
    if all_override_f:
        oarr = np.array(all_override_f)
        result["f_override_count"] = len(all_override_f)
        result["f_override_mean"] = round(float(oarr.mean()), 4)
        result["f_override_std"] = round(float(oarr.std()), 4)
    return result


# ── GMR 模式分析 ──────────────────────────────────────────

def extract_gmr_histories(results: list[dict]) -> list[list[str]]:
    """提取各 run 的 GMR 模式轨迹。

    Returns:
        [[mode_gen0, mode_gen1, ...], ...]
    """
    histories = []
    for r in results:
        recs = r.get("generation_records", [])
        if not recs:
            continue
        modes = [rec.get("gmr_mode", "auto") for rec in recs]
        histories.append(modes)
    return histories


def compute_gmr_stats(results: list[dict]) -> dict:
    """计算 GMR 模式的统计信息。"""
    decisions = extract_llm_decisions(results)
    sc_decisions = [d for d in decisions if d.get("module") == "search_controller"]
    if not sc_decisions:
        return {}
    all_modes = []
    for d in sc_decisions:
        parsed = d.get("parsed_decision", {})
        gmr = parsed.get("gmr_mode", "auto")
        all_modes.append(gmr)
    total = len(all_modes)
    mode_counts = {m: all_modes.count(m) for m in set(all_modes)}
    mode_pcts = {m: round(c / total * 100, 1) for m, c in mode_counts.items()}
    # 模式切换次数
    switches = sum(1 for i in range(1, len(all_modes)) if all_modes[i] != all_modes[i-1])
    return {
        "total_decisions": total,
        "mode_counts": mode_counts,
        "mode_pcts": mode_pcts,
        "mode_switches": switches,
    }


# ── 解耦参数控制综合分析 ──────────────────────────────────

def compute_parameter_coupling(results: list[dict]) -> dict:
    """分析 CR、F、GMR 三个参数之间的独立性和相关性。

    Returns:
        含相关系数、独立性指标的字典。
    """
    if not results:
        return {}
    # 收集所有 run 的 CR 和 F 序列
    all_cr = []
    all_f = []
    for r in results:
        recs = r.get("generation_records", [])
        for rec in recs:
            cr = rec.get("cr", 0.5)
            f_val = rec.get("f_override")
            if f_val is None:
                f_val = rec.get("f_scale", 0.5)
            all_cr.append(cr)
            all_f.append(f_val)
    if len(all_cr) < 10:
        return {}
    cr_arr = np.array(all_cr)
    f_arr = np.array(all_f)
    # Pearson 相关系数
    if np.std(cr_arr) > 1e-10 and np.std(f_arr) > 1e-10:
        corr = float(np.corrcoef(cr_arr, f_arr)[0, 1])
    else:
        corr = 0.0
    # GMR 模式统计
    gmr_modes = []
    for r in results:
        recs = r.get("generation_records", [])
        for rec in recs:
            gmr_modes.append(rec.get("gmr_mode", "auto"))
    mode_counts = {m: gmr_modes.count(m) for m in set(gmr_modes)} if gmr_modes else {}
    return {
        "cr_f_correlation": round(corr, 4),
        "cr_mean": round(float(cr_arr.mean()), 4),
        "cr_std": round(float(cr_arr.std()), 4),
        "f_mean": round(float(f_arr.mean()), 4),
        "f_std": round(float(f_arr.std()), 4),
        "gmr_mode_counts": mode_counts,
        "n_samples": len(all_cr),
    }
