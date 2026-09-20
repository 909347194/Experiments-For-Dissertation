# -*- coding: utf-8 -*-
"""recorder.py — 实验过程记录器

统一记录每次运行的完整过程数据：
- 每代 fitness / CR / F / diversity
- LLM 决策详情（输入、输出、推理、耗时）
- 时间分口径

职责单一：只负责收集和序列化，不负责求解逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GenerationRecord:
    """单代记录。"""
    gen: int
    fitness_best: float
    fitness_mean: float
    cr: float
    f_scale: float
    diversity: float = 0.0
    feasible_ratio: float = 0.0
    # LLM 独立覆写的参数（None = 使用公式推导值）
    f_override: float | None = None     # LLM 直接指定的 F 值
    gmr_mode: str = "auto"              # LLM 指定的 GMR 模式: "auto" | "on" | "off"
    # LLM 决策（仅在 LLM 触发代有值）
    llm_module: str = ""
    llm_decision: dict = field(default_factory=dict)
    llm_reasoning: str = ""
    llm_call_duration: float = 0.0


@dataclass
class LLMDecisionRecord:
    """一次 LLM 决策的完整记录。"""
    generation: int
    module: str
    # 输入
    prompt_messages: list[dict] = field(default_factory=list)
    model: str = ""
    temperature: float = 0.0
    # 输出
    raw_output: str = ""
    parsed_decision: dict = field(default_factory=dict)
    reasoning: str = ""
    # 应用效果
    cr_before: float = 0.0
    cr_after: float = 0.0
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    # 耗时
    duration: float = 0.0


class RunRecorder:
    """单次运行的完整过程记录器。

    用法::

        recorder = RunRecorder(seed=42)
        recorder.record_generation(gen=1, fitness_best=100, cr=0.5, ...)
        recorder.record_llm_decision(gen=50, module="search_controller", ...)
        result = recorder.finalize(total_time=36.5)
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._generations: list[GenerationRecord] = []
        self._llm_decisions: list[LLMDecisionRecord] = []
        self._convergence_curve: list[float] = []

        # 时间分口径
        self._total_time: float = 0.0
        self._dmde_time: float = 0.0
        self._llm_time: float = 0.0
        self._llm_init_time: float = 0.0
        self._llm_cr_time: float = 0.0
        self._llm_call_count: int = 0

    # ── 记录方法 ──────────────────────────────────────────

    def record_generation(self, gen: int, fitness_best: float,
                          fitness_mean: float, cr: float, f_scale: float,
                          diversity: float = 0.0, feasible_ratio: float = 0.0,
                          f_override: float | None = None,
                          gmr_mode: str = "auto",
                          llm_module: str = "", llm_decision: dict = None,
                          llm_reasoning: str = "", llm_call_duration: float = 0.0):
        """记录一代进化数据。"""
        self._generations.append(GenerationRecord(
            gen=gen, fitness_best=fitness_best, fitness_mean=fitness_mean,
            cr=cr, f_scale=f_scale, diversity=diversity,
            feasible_ratio=feasible_ratio,
            f_override=f_override, gmr_mode=gmr_mode,
            llm_module=llm_module, llm_decision=llm_decision or {},
            llm_reasoning=llm_reasoning, llm_call_duration=llm_call_duration,
        ))
        self._convergence_curve.append(fitness_best)

    def record_llm_decision(self, generation: int, module: str,
                            prompt_messages: list[dict], model: str,
                            raw_output: str, parsed_decision: dict,
                            reasoning: str, duration: float,
                            cr_before: float = 0.0, cr_after: float = 0.0,
                            fitness_before: float = 0.0, fitness_after: float = 0.0,
                            temperature: float = 0.0):
        """记录一次 LLM 决策的完整详情。"""
        self._llm_decisions.append(LLMDecisionRecord(
            generation=generation, module=module,
            prompt_messages=prompt_messages, model=model,
            temperature=temperature,
            raw_output=raw_output, parsed_decision=parsed_decision,
            reasoning=reasoning, duration=duration,
            cr_before=cr_before, cr_after=cr_after,
            fitness_before=fitness_before, fitness_after=fitness_after,
        ))
        self._llm_call_count += 1

    # ── 时间设置 ──────────────────────────────────────────

    def set_times(self, total: float, dmde: float = 0.0,
                  llm: float = 0.0, llm_init: float = 0.0,
                  llm_cr: float = 0.0):
        """设置时间分口径。"""
        self._total_time = total
        self._dmde_time = dmde
        self._llm_time = llm
        self._llm_init_time = llm_init
        self._llm_cr_time = llm_cr

    def set_llm_call_count(self, count: int):
        """设置 LLM 调用次数（用于从 solver 获取准确值）。"""
        self._llm_call_count = count

    # ── 输出 ──────────────────────────────────────────────

    def finalize(self) -> dict:
        """生成最终结果字典（保存到 JSON）。"""
        fitness_arr = np.array(self._convergence_curve) if self._convergence_curve else np.array([float("inf")])
        cr_history = [g.cr for g in self._generations]

        return {
            "seed": self.seed,
            "best_fitness": round(float(fitness_arr.min()), 2),
            # 收敛曲线
            "convergence_curve": [round(float(f), 2) for f in self._convergence_curve],
            # CR 变化轨迹
            "cr_history": [round(float(c), 4) for c in cr_history],
            # 每代详细记录（精简版，用于分析）
            "generation_records": [
                {
                    "gen": g.gen,
                    "fitness_best": round(float(g.fitness_best), 2),
                    "fitness_mean": round(float(g.fitness_mean), 2),
                    "cr": round(float(g.cr), 4),
                    "f_scale": round(float(g.f_scale), 4),
                    "f_override": round(float(g.f_override), 4) if g.f_override is not None else None,
                    "gmr_mode": g.gmr_mode,
                    "diversity": round(float(g.diversity), 4),
                    "llm_module": g.llm_module,
                }
                for g in self._generations
            ],
            # LLM 决策完整记录
            "llm_decisions": [
                {
                    "generation": d.generation,
                    "module": d.module,
                    "model": d.model,
                    "temperature": d.temperature,
                    "prompt_messages": d.prompt_messages,
                    "raw_output": d.raw_output,
                    "parsed_decision": d.parsed_decision,
                    "reasoning": d.reasoning,
                    "cr_before": round(float(d.cr_before), 4),
                    "cr_after": round(float(d.cr_after), 4),
                    "fitness_before": round(float(d.fitness_before), 2),
                    "fitness_after": round(float(d.fitness_after), 2),
                    "duration": round(float(d.duration), 3),
                }
                for d in self._llm_decisions
            ],
            # 时间分口径
            "total_time": round(self._total_time, 2),
            "dmde_time": round(self._dmde_time, 2),
            "llm_time": round(self._llm_time, 2),
            "llm_init_time": round(self._llm_init_time, 2),
            "llm_cr_time": round(self._llm_cr_time, 2),
            "llm_call_count": self._llm_call_count,
        }

    @staticmethod
    def save_results(results: list[dict], output_dir: Path) -> Path:
        """保存结果列表到 JSON。"""
        out = output_dir / "results" / "ablation_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)
        return out


def _json_default(obj):
    """JSON 序列化兜底。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)