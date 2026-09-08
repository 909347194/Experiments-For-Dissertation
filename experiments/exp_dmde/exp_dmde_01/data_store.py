# -*- coding: utf-8 -*-
"""data_store.py — 实验全量数据的保存与加载

职责：
    将一次实验运行产生的“绘图所需的全部数据”持久化为单个 JSON 文件，
    并支持从该文件完整还原出可视化器所需的对象结构。
    这样调整绘图（修改 visualizer.py）时，只需重新运行
    ``plot_from_saved.py`` 读取本文件即可，无需重跑实验求解。

保存的内容（相对旧版汇总 JSON 的关键差异）：
    - cost_matrix   : 每个场景的代价矩阵（旧版未存，绘图热力图/连线需要）
    - runs          : 每次运行的 SolverResult 全部字段，含 cost_history
                      （收敛曲线）、best_fitness、elapsed_seconds、extra
                      （total_violation / is_feasible 等）——旧版只存了
                      best_assignment
    - metrics       : ExperimentMetrics 全部 10 个字段（含 best_run_idx）
    - entities      : 每个场景的 UAV / Target 列表（分配图、3D 图需要坐标）

说明（序列化坑）：
    旧版用 ``json.dump(default=str)`` 会把 numpy 整数转成 ``"2"`` 这类字符串，
    破坏类型。本模块通过 :func:`_to_jsonable` 递归地把 numpy 标量 / 数组 / 元组
    归一为纯 Python 类型（np.integer → int 等），保证往返一致。

使用方式：
    # 保存（实验 run.py 内）
    from data_store import save_experiment, DEFAULT_DATA_FILE
    save_experiment(scenarios, uavs_dict, targets_dict,
                    meta={...}, path=RESULTS_DIR / DEFAULT_DATA_FILE)

    # 加载（plot_from_saved.py 内）
    from data_store import load_experiment
    payload = load_experiment("results/exp_dmde_01_data.json")
    scenarios   = payload["scenarios"]     # 与可视化器所需结构一致
    uavs_dict   = payload["uavs_dict"]
    targets_dict = payload["targets_dict"]
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# 全量数据文件默认名（保存在 experiments/exp_dmde/exp_dmde_01/results/ 下）
DEFAULT_DATA_FILE = "exp_dmde_01_data.json"


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


def _scenario_to_dict(sc: dict) -> dict:
    """把可视化器使用的场景 dict 序列化为纯 JSON 结构。"""
    return {
        "name": sc["name"],
        "model_type": sc["model_type"],
        "n_uavs": sc["n_uavs"],
        "n_targets": sc["n_targets"],
        "cost_matrix": _to_jsonable(sc["cost_matrix"]),
        "metrics": _to_jsonable(asdict(sc["metrics"])),
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
# 保存
# ──────────────────────────────────────────────────────────────

def save_experiment(
    scenarios: list[dict],
    uavs_dict: dict[str, list[Any]],
    targets_dict: dict[str, list[Any]],
    meta: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path:
    """保存全量实验数据到单个 JSON 文件。

    Args:
        scenarios:    run_scenario 返回的场景 dict 列表（含 metrics/results/cost_matrix）。
        uavs_dict:    场景名 → UAV 列表。
        targets_dict: 场景名 → Target 列表。
        meta:         实验元信息（参数、时间戳等），可为 None。
        path:         输出文件路径；None 时用当前目录下 DEFAULT_DATA_FILE。

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
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


# ──────────────────────────────────────────────────────────────
# 加载 / 还原
# ──────────────────────────────────────────────────────────────

def _result_from_dict(d: dict):
    """dict → SolverResult（与可视化器使用的对象一致）。"""
    from algorithms.algorithm_dmde.base.base_optimizer import SolverResult

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

    metrics = ExperimentMetrics(**d["metrics"])
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
        - "meta":        元信息 dict。
        - "scenarios":   可视化器可用的场景 dict 列表
                         （metrics 为 ExperimentMetrics、results 为 SolverResult、
                         cost_matrix 为 np.ndarray）。
        - "uavs_dict":   场景名 → UAV 对象列表。
        - "targets_dict": 场景名 → Target 对象列表。
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
    }
