# -*- coding: utf-8 -*-
"""registry.py — 实验结果索引

每个实验 run.py 完成后调用 register()，将结果路径写入统一索引。
compare_experiments.py 和 run_batch.py 读取索引来发现所有实验结果。

索引文件：experiments/index.json

用法：
    from registry import register, get_results, list_experiments

    # 注册（run.py 结束时调用）
    register(
        experiment_id="dmde_01",
        label="DMDE 基线 N=M=10",
        result_path="exp_dmde/exp_dmde_01/results/exp_dmde_01_data.json",
        tags=["baseline", "balanced", "10u10t"],
    )

    # 查询
    results = get_results(tags=["baseline"])       # 按标签查
    results = get_results(experiment_id="dmde_01")  # 按 ID 查
    all_exps = list_experiments()                    # 列出全部
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

INDEX_FILE = Path(__file__).resolve().parent / "index.json"


def _load_index() -> dict:
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"experiments": {}}


def _save_index(data: dict) -> None:
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def register(
    experiment_id: str,
    label: str,
    result_path: str | Path,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """注册一个实验结果到索引。

    Args:
        experiment_id: 唯一标识，如 "dmde_01"、"llm_dmde_01"。
        label:         人类可读标签，如 "DMDE 基线 N=M=10"。
        result_path:   结果 JSON 路径（相对于 experiments/ 目录）。
        tags:          标签列表，用于筛选。
        meta:          额外元信息。
    """
    data = _load_index()
    data["experiments"][experiment_id] = {
        "label": label,
        "result_path": str(result_path),
        "tags": tags or [],
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta or {},
    }
    _save_index(data)


def get_results(
    experiment_id: str | None = None,
    tags: list[str] | None = None,
) -> list[dict]:
    """查询实验结果。

    Args:
        experiment_id: 精确匹配 ID。
        tags:          按标签过滤（AND 逻辑）。

    Returns:
        匹配的实验记录列表。
    """
    data = _load_index()
    results = []
    for eid, entry in data["experiments"].items():
        if experiment_id and eid != experiment_id:
            continue
        if tags and not all(t in entry.get("tags", []) for t in tags):
            continue
        results.append({"id": eid, **entry})
    return results


def list_experiments() -> list[dict]:
    """列出索引中的所有实验。"""
    return get_results()


def get_result_path(experiment_id: str) -> Path | None:
    """获取实验结果文件的绝对路径。"""
    data = _load_index()
    entry = data["experiments"].get(experiment_id)
    if not entry:
        return None
    p = INDEX_FILE.parent / entry["result_path"]
    return p if p.exists() else None


def remove(experiment_id: str) -> bool:
    """从索引中移除一个实验。"""
    data = _load_index()
    if experiment_id in data["experiments"]:
        del data["experiments"][experiment_id]
        _save_index(data)
        return True
    return False
