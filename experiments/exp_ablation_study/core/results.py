# -*- coding: utf-8 -*-
"""results.py — 结果保存/加载 + 配置读取"""

import json
from pathlib import Path


def save_results(results: list[dict], output_dir: Path) -> Path:
    """保存结果列表到 JSON 文件。

    Args:
        results: 运行结果字典列表
        output_dir: 输出目录（如 A0_dmde/）

    Returns:
        保存的文件路径
    """
    out = output_dir / "results" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return out


def load_results(scenario_dir: Path, config_dir: str) -> list[dict]:
    """从 results/ablation_results.json 加载结果。

    Args:
        scenario_dir: 场景目录（如 S1_balanced_N10_M10/）
        config_dir: 配置目录名（如 A0_dmde）

    Returns:
        结果字典列表，文件不存在时返回空列表
    """
    path = scenario_dir / config_dir / "results" / "ablation_results.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_modules_config(llm_config_path: str) -> dict:
    """从 llm_config.yaml 读取 modules 配置。

    Args:
        llm_config_path: YAML 配置文件路径

    Returns:
        modules 配置字典
    """
    import yaml
    with open(llm_config_path, encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data.get("modules", {})