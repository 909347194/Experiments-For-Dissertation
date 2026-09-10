# -*- coding: utf-8 -*-
"""data_store.py — LLM 增强实验全量数据的保存与加载

基于 exp_dmde 的 data_store.py 扩展，额外支持 LLM 决策日志的持久化。

新增内容：
    - llm_decisions : 每次运行中 LLM 的所有决策记录
        - generation    : 触发 LLM 的代数
        - features      : 上报给 LLM 的特征值
        - llm_response  : LLM 返回的算子选择与推理过程
        - chosen_operator: 最终选定的 DE 算子
        - timestamp     : 决策时间戳

使用方式：
    # 保存（实验 run.py 内）
    from data_store import save_experiment, DEFAULT_DATA_FILE
    save_experiment(scenarios, uavs_dict, targets_dict,
                    meta={...}, llm_decisions={...},
                    path=RESULTS_DIR / DEFAULT_DATA_FILE)

    # 加载（plot_from_saved.py 内）
    from data_store import load_experiment
    payload = load_experiment("results/exp_llm_dmde_01_data.json")
    scenarios     = payload["scenarios"]
    uavs_dict     = payload["uavs_dict"]
    targets_dict  = payload["targets_dict"]
    llm_decisions = payload["llm_decisions"]   # 新增
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np

# 全量数据文件默认名
DEFAULT_DATA_FILE = "exp_llm_dmde_01_data.json"


# ──────────────────────────────────────────────────────────────
# 序列化辅助
# ──────────────────────────────────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    """递归地把对象转换为可 JSON 序列化的纯 Python 类型。

    覆盖 numpy 标量 / 布尔 / 数组、tuple、嵌套 list/dict；
    遇到未知类型直接抛错，避免静默丢数据。
    """
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    raise TypeError(f"无法序列化的类型: {type(obj)!r}")


def _pair_to_list(pair) -> list:
    """把 (uav_id, target_id) 归一为等长列表（并做类型归一）。"""
    return _to_jsonable(list(pair))


# ──────────────────────────────────────────────────────────────
# 场景 → dict 序列化
# ──────────────────────────────────────────────────────────────

def _result_to_dict(r: Any) -> dict:
    """单个 SolverResult → dict。"""
    return {
        "best_assignment": [_pair_to_list(p) for p in r.best_assignment],
        "best_fitness": _to_jsonable(r.best_fitness),
        "cost_history": _to_jsonable(r.cost_history),
        "total_generations": _to_jsonable(r.total_generations),
        "elapsed_seconds": _to_jsonable(r.elapsed_seconds),
        "solver_name": _to_jsonable(r.solver_name),
        "extra": _to_jsonable(r.extra),
    }


def _metrics_to_dict(m: Any) -> dict:
    """ExperimentMetrics → dict，并附带按次数统计的派生指标。"""
    d = asdict(m)
    d["n_feasible"] = m.n_feasible
    d["n_infeasible"] = m.n_infeasible
    d["infeasible_rate"] = m.infeasible_rate
    return d


def _scenario_to_dict(sc: dict) -> dict:
    """把可视化器使用的场景 dict 序列化为纯 JSON 结构。"""
    return {
        "name": sc["name"],
        "model_type": sc["model_type"],
        "n_uavs": sc["n_uavs"],
        "n_targets": sc["n_targets"],
        "cost_matrix": _to_jsonable(sc["cost_matrix"]),
        "metrics": _to_jsonable(_metrics_to_dict(sc["metrics"])),
        "runs": [_result_to_dict(r) for r in sc["results"]],
    }


def _uav_to_dict(u: Any) -> dict:
    return {
        "id": _to_jsonable(u.id),
        "start_pos": _to_jsonable(u.start_pos),
        "speed_range": _to_jsonable(u.speed_range),
        "max_range": _to_jsonable(u.max_range),
        "max_time": _to_jsonable(u.max_time),
    }


def _target_to_dict(t: Any) -> dict:
    return {
        "id": _to_jsonable(t.id),
        "position": _to_jsonable(t.position),
        "weight": _to_jsonable(t.weight),
        "time_window": _to_jsonable(t.time_window),
        "sequence_group": _to_jsonable(t.sequence_group),
    }


# ──────────────────────────────────────────────────────────────
# LLM 决策日志序列化
# ──────────────────────────────────────────────────────────────

def _llm_decisions_to_jsonable(decisions: Any) -> Any:
    """将 LLM 决策日志转换为可 JSON 序列化的格式。

    decisions 可以是：
    - list[dict]: 每个 dict 包含一次 LLM 调用的记录
    - dict[int, list[dict]]: 按 run_idx 分组的决策记录
    - None: 无 LLM 决策（兼容原始 DMDE）
    """
    if decisions is None:
        return None
    return _to_jsonable(decisions)


# ──────────────────────────────────────────────────────────────
# 保存
# ──────────────────────────────────────────────────────────────

def save_experiment(
    scenarios: list[dict],
    uavs_dict: dict[str, list[Any]],
    targets_dict: dict[str, list[Any]],
    meta: dict[str, Any] | None = None,
    path: str | Path | None = None,
    llm_decisions: Any = None,
) -> Path:
    """保存全量实验数据到单个 JSON 文件。

    Args:
        scenarios:     run_scenario 返回的场景 dict 列表。
        uavs_dict:     场景名 → UAV 列表。
        targets_dict:  场景名 → Target 列表。
        meta:          实验元信息（参数、时间戳等），可为 None。
        path:          输出文件路径；None 时用当前目录下 DEFAULT_DATA_FILE。
        llm_decisions: LLM 决策日志，按 run_idx 分组或扁平列表。

    Returns:
        实际写入的文件路径。
    """
    out_path = Path(path) if path is not None else Path(DEFAULT_DATA_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entities = {}
    for sc in scenarios:
        name = sc["name"]
        entities[name] = {
            "uavs": [_uav_to_dict(u) for u in uavs_dict.get(name, [])],
            "targets": [_target_to_dict(t) for t in targets_dict.get(name, [])],
        }

    payload = {
        "meta": _to_jsonable(meta or {}),
        "scenarios": [_scenario_to_dict(sc) for sc in scenarios],
        "entities": entities,
        "llm_decisions": _llm_decisions_to_jsonable(llm_decisions),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


# ──────────────────────────────────────────────────────────────
# 加载 / 还原
# ──────────────────────────────────────────────────────────────

def _result_from_dict(d: dict):
    """dict → SolverResult（与可视化器使用的对象一致）。"""
    from algorithms.algorithm_llm_enhanced_dmde.base.base_optimizer import SolverResult

    return SolverResult(
        best_assignment=[tuple(p) for p in d["best_assignment"]],
        best_fitness=d["best_fitness"],
        cost_history=list(d["cost_history"]),
        total_generations=d["total_generations"],
        elapsed_seconds=d["elapsed_seconds"],
        solver_name=d.get("solver_name", "unknown"),
        extra=dict(d.get("extra", {})),
    )


def _scenario_from_dict(d: dict) -> dict:
    """还原出与可视化器 plot_all 所需完全一致的场景 dict。"""
    from utils.utils_dmde.metrics import ExperimentMetrics

    known = {f.name for f in fields(ExperimentMetrics)}
    metrics = ExperimentMetrics(**{k: v for k, v in d["metrics"].items() if k in known})
    runs = [_result_from_dict(r) for r in d["runs"]]

    return {
        "name": d["name"],
        "model_type": d["model_type"],
        "n_uavs": d["n_uavs"],
        "n_targets": d["n_targets"],
        "metrics": metrics,
        "results": runs,
        "cost_matrix": np.array(d["cost_matrix"]),
    }


def _uav_from_dict(d: dict):
    from models.model_dmde.entities import UAV

    return UAV(
        id=d["id"],
        start_pos=tuple(d["start_pos"]),
        speed_range=tuple(d["speed_range"]),
        max_range=d["max_range"],
        max_time=d["max_time"],
    )


def _target_from_dict(d: dict):
    from models.model_dmde.entities import Target

    return Target(
        id=d["id"],
        position=tuple(d["position"]),
        weight=d["weight"],
        time_window=tuple(d["time_window"]) if d["time_window"] is not None else None,
        sequence_group=d["sequence_group"],
    )


def load_experiment(path: str | Path) -> dict:
    """从 JSON 加载并完整还原实验数据。

    Returns:
        dict，包含键：
        - "meta":          元信息 dict。
        - "scenarios":     可视化器可用的场景 dict 列表。
        - "uavs_dict":     场景名 → UAV 对象列表。
        - "targets_dict":  场景名 → Target 对象列表。
        - "llm_decisions": LLM 决策日志（可能为 None）。
    """
    src = Path(path)
    with open(src, "r", encoding="utf-8") as f:
        payload = json.load(f)

    scenarios = [_scenario_from_dict(d) for d in payload["scenarios"]]

    uavs_dict = {}
    targets_dict = {}
    for name, ent in payload.get("entities", {}).items():
        uavs_dict[name] = [_uav_from_dict(d) for d in ent.get("uavs", [])]
        targets_dict[name] = [_target_from_dict(d) for d in ent.get("targets", [])]

    return {
        "meta": payload.get("meta", {}),
        "scenarios": scenarios,
        "uavs_dict": uavs_dict,
        "targets_dict": targets_dict,
        "llm_decisions": payload.get("llm_decisions"),
    }
