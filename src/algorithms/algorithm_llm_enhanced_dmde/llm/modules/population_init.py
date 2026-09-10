# -*- coding: utf-8 -*-
"""population_init.py — LLM 种群初始化模块

职责：
    在种群初始化阶段，由 LLM 基于代价矩阵特征建议初始种群策略，
    或对随机生成的初始种群进行智能优化。

注入点：after_init（种群初始化后，优化初始种群）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from ..base_module import BaseLLMModule, ModuleState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in evolutionary optimization initialization.

Your task: Given the cost matrix structure and problem characteristics, \
suggest an initialization strategy to improve the initial population quality.

## Decision Format
Respond with a JSON object only (no markdown):
{{
    "init_strategy": "<greedy|diverse|hybrid|random>",
    "temperature": <float in [0.0, 1.0], or null to keep default>,
    "reasoning": "<brief explanation>"
}}

## Strategies
- "greedy": Bias toward low-cost assignments (exploitative start)
- "diverse": Maximize initial diversity (exploratory start)
- "hybrid": Mix of greedy and diverse individuals
- "random": Standard random initialization (no change)

## Guidelines
- Small problem (N<=10) → greedy often works well
- Large problem (N>20) → diverse or hybrid preferred
- Many constraints → hybrid to ensure feasible individuals
"""


class LLMPopulationInitModule(BaseLLMModule):
    """LLM 种群初始化模块。"""

    @property
    def name(self) -> str:
        return "population_init"

    @property
    def hook_point(self) -> str:
        return "after_init"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # 代价矩阵统计
        cm = state.cost_matrix
        cm_stats = {}
        if cm is not None:
            finite_vals = cm[np.isfinite(cm)]
            cm_stats = {
                "shape": list(cm.shape),
                "min": float(finite_vals.min()) if len(finite_vals) else 0,
                "max": float(finite_vals.max()) if len(finite_vals) else 0,
                "mean": float(finite_vals.mean()) if len(finite_vals) else 0,
                "n_infinite": int(np.sum(~np.isfinite(cm))),
            }

        features = {
            "n_uavs": state.n_uavs,
            "n_targets": state.n_targets,
            "model_type": state.model_type,
            "pop_size": state.extra.get("pop_size", 50),
            "cost_matrix_stats": cm_stats,
            "initial_diversity": state.diversity,
            "initial_best_fitness": state.best_fitness,
            "initial_feasible_ratio": state.feasible_ratio,
        }

        user = (
            f"## Problem Characteristics\n{json.dumps(features, indent=2)}\n\n"
            f"## Task\nSuggest the best initialization strategy. "
            f"Respond with JSON only."
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return {"init_strategy": "random", "reasoning": "Parse failed"}

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {"init_strategy": "random", "reasoning": "Invalid JSON"}

        strategy = data.get("init_strategy", "random")
        valid = {"greedy", "diverse", "hybrid", "random"}
        if strategy not in valid:
            strategy = "random"

        temp = data.get("temperature")
        if temp is not None:
            try:
                temp = max(0.0, min(1.0, float(temp)))
            except (ValueError, TypeError):
                temp = None

        return {
            "init_strategy": strategy,
            "temperature": temp,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """Apply LLM decision by modifying part of the population.

        Strategies:
            "greedy"  — replace individuals with greedy-constructed ones
            "diverse" — perturb existing individuals to increase diversity
            "hybrid"  — 50/50 mix of greedy and diverse
            "random"  — no change
        """
        strategy = decision.get("init_strategy", "random")
        temperature = decision.get("temperature")

        if state.population is None or state.cost_matrix is None:
            return state

        if strategy == "random":
            state.extra["llm_init_applied"] = "random"
            state.extra["llm_init_n_modified"] = 0
            return state

        rng = np.random.default_rng()
        cm = state.cost_matrix
        pop = state.population
        n_uavs = state.n_uavs
        n_targets = state.n_targets

        # Determine how many individuals to modify (30% of population)
        n_modify = max(1, int(len(pop) * 0.3))
        indices_to_modify = rng.choice(len(pop), size=n_modify, replace=False)

        for idx in indices_to_modify:
            if strategy == "greedy" or (strategy == "hybrid" and rng.random() < 0.5):
                ind = self._build_greedy_individual(cm, n_uavs, n_targets, state.model_type, rng)
            else:
                ind = self._perturb_individual(pop[idx], cm, n_uavs, n_targets, state.model_type, rng)

            ind.fitness = float("inf")  # Force re-evaluation
            pop[idx] = ind

        state.extra["llm_init_applied"] = strategy
        state.extra["llm_init_n_modified"] = n_modify
        state.extra["llm_init_temperature"] = temperature

        return state

    # ---- Helper: greedy individual construction ----

    def _build_greedy_individual(
        self,
        cm: np.ndarray,
        n_uavs: int,
        n_targets: int,
        model_type: str,
        rng: np.random.Generator,
    ) -> "Individual":
        """Build one individual using a greedy nearest-cost assignment.

        Follows the same encoding rules as encoder.py:
            - balanced  (N==M): Rule 3.1 — one-to-one, no repeats
            - overloaded (N>M): Rule 3.2 — each target at least once
            - srp       (N<M): Rule 3.3 — UAVs visit multiple targets
        """
        from ...representation.encoder import Gene, Individual

        if model_type == "balanced":
            return self._greedy_balanced(cm, n_uavs, rng)
        elif model_type == "overloaded":
            return self._greedy_overloaded(cm, n_uavs, n_targets, rng)
        else:
            return self._greedy_srp(cm, n_uavs, n_targets, rng)

    def _greedy_balanced(self, cm, n, rng):
        """Rule 3.1: greedy one-to-one assignment (hungarian-lite)."""
        from ...representation.encoder import Gene, Individual

        available_targets = set(range(n))
        genes = []
        for uav_id in range(n):
            # Pick lowest-cost available target for this UAV
            best_tgt = min(available_targets, key=lambda t: cm[uav_id, t])
            genes.append(Gene(uav_id=uav_id, target_id=best_tgt, cost=float(cm[uav_id, best_tgt])))
            available_targets.remove(best_tgt)
        return Individual(genes=genes, model_type="balanced")

    def _greedy_overloaded(self, cm, n_uavs, n_targets, rng):
        """Rule 3.2: each target at least once, extra UAVs go to cheapest target."""
        from ...representation.encoder import Gene, Individual

        # Shuffle UAV order to introduce some randomness
        uav_order = list(range(n_uavs))
        rng.shuffle(uav_order)

        genes = []
        assigned_targets = set()

        # First pass: assign each target to its cheapest available UAV
        for tgt_id in range(n_targets):
            # Find unassigned UAV with lowest cost to this target
            candidates = [u for u in uav_order if u not in {g.uav_id for g in genes}]
            if candidates:
                best_uav = min(candidates, key=lambda u: cm[u, tgt_id])
                genes.append(Gene(uav_id=best_uav, target_id=tgt_id, cost=float(cm[best_uav, tgt_id])))
                assigned_targets.add(tgt_id)

        # Second pass: remaining UAVs → cheapest target
        used_uavs = {g.uav_id for g in genes}
        for uav_id in uav_order:
            if uav_id not in used_uavs:
                tgt_id = int(rng.integers(0, n_targets))
                genes.append(Gene(uav_id=uav_id, target_id=tgt_id, cost=float(cm[uav_id, tgt_id])))

        return Individual(genes=genes, model_type="overloaded")

    def _greedy_srp(self, cm, n_uavs, n_targets, rng):
        """Rule 3.3: greedy nearest-neighbor tour construction."""
        from ...representation.encoder import Gene, Individual

        # Distribute targets to UAVs: each UAV gets at least one
        targets = list(range(n_targets))
        rng.shuffle(targets)

        uav_groups: dict[int, list[int]] = {}
        for i in range(n_uavs):
            uav_groups[i] = [targets[i]]
        for i in range(n_uavs, n_targets):
            uav_id = int(rng.integers(0, n_uavs))
            uav_groups[uav_id].append(targets[i])

        # Build genes with greedy tour ordering
        genes = []
        for uav_id, tgt_list in uav_groups.items():
            ordered = self._order_targets_greedy(cm, n_uavs, uav_id, tgt_list)
            for seq, tgt_id in enumerate(ordered):
                if seq == 0:
                    cost = float(cm[uav_id, tgt_id])
                    genes.append(Gene(uav_id=uav_id, target_id=tgt_id, cost=cost))
                else:
                    prev_tgt = ordered[seq - 1]
                    cost = float(cm[n_uavs + prev_tgt, tgt_id])
                    genes.append(Gene(uav_id=-1, target_id=tgt_id, cost=cost))

        return Individual(genes=genes, model_type="srp")

    @staticmethod
    def _order_targets_greedy(cm, n_uavs, uav_id, targets):
        """Nearest-neighbor ordering for SRP tours."""
        if len(targets) <= 1:
            return targets
        remaining = set(targets)
        first = min(remaining, key=lambda t: cm[uav_id, t])
        ordered = [first]
        remaining.remove(first)
        while remaining:
            last = ordered[-1]
            next_tgt = min(remaining, key=lambda t: cm[n_uavs + last, t])
            ordered.append(next_tgt)
            remaining.remove(next_tgt)
        return ordered

    # ---- Helper: perturb individual for diversity ----

    def _perturb_individual(
        self,
        individual: "Individual",
        cm: np.ndarray,
        n_uavs: int,
        n_targets: int,
        model_type: str,
        rng: np.random.Generator,
    ) -> "Individual":
        """Create a perturbed copy: swap some target assignments randomly."""
        from ...representation.encoder import Gene, Individual

        new_ind = individual.copy()
        genes = list(new_ind.genes)
        n_swap = max(1, len(genes) // 3)

        for _ in range(n_swap):
            i, j = rng.choice(len(genes), size=2, replace=False)
            gi, gj = genes[i], genes[j]

            if model_type == "balanced":
                # Swap targets between two UAVs
                genes[i] = Gene(uav_id=gi.uav_id, target_id=gj.target_id, cost=float(cm[gi.uav_id, gj.target_id]))
                genes[j] = Gene(uav_id=gj.uav_id, target_id=gi.target_id, cost=float(cm[gj.uav_id, gi.target_id]))
            elif model_type == "overloaded":
                # Reassign one gene to a random target
                new_tgt = int(rng.integers(0, n_targets))
                genes[i] = Gene(uav_id=gi.uav_id, target_id=new_tgt, cost=float(cm[gi.uav_id, new_tgt]))
            else:
                # SRP: swap two genes' targets (preserve tour structure)
                if gi.uav_id >= 0 and gj.uav_id >= 0:
                    # Both are UAV→Target genes
                    genes[i] = Gene(uav_id=gi.uav_id, target_id=gj.target_id, cost=float(cm[gi.uav_id, gj.target_id]))
                    genes[j] = Gene(uav_id=gj.uav_id, target_id=gi.target_id, cost=float(cm[gj.uav_id, gi.target_id]))
                # Skip if either is a tour gene (uav_id=-1) to avoid breaking tour structure

        new_ind.genes = genes
        return new_ind

    @staticmethod
    def _extract_json(text: str) -> str | None:
        text = text.strip()
        if text.startswith("{"):
            return text
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return None
