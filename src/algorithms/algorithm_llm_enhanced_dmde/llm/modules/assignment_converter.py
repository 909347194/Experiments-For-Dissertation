# -*- coding: utf-8 -*-
"""assignment_converter.py — 离散 assignment → DMDE 统一基因编码转换器

职责：
    将 LLM 生成的离散分配方案（assignment）转换为 DMDE 的统一基因编码（Individual）。
    独立模块，方便测试和复用。

对应论文：
    公式 (3-3): 矩阵关系三元组 g_s = (U_i, T_j, C(i,j))
    公式 (3-4): 巡游关系三元组 g'_s = (T_j, T_{j+1}, Tc(j,j+1))
    规则 3.1 (N=M):  U 和 T 均不重复，一一对应。
    规则 3.2 (N>M):  U 不重复，T 可重复，每个 T 至少出现一次。
    规则 3.3 (N<M):  U 可重复（巡游），T 不重复，每个 U 至少出现一次。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...representation.encoder import Gene, Individual

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """单个 solution 转换结果（含修复元数据）。

    Attributes:
        individual:     转换后的 Individual（失败时为 None）。
        raw_valid:      LLM 原始输出是否直接可行（未经修复）。
        repaired:       是否经过修复才转换成功。
        repair_distance:修复时修改的 UAV-target assignment 数量。
        raw_cost:       修复前的 assignment 代价值（从 cost_matrix 计算）。
        repaired_cost:  修复后的 assignment 代价值。
    """
    individual: Individual | None = None
    raw_valid: bool = False
    repaired: bool = False
    repair_distance: int = 0
    raw_cost: float = float("inf")
    repaired_cost: float = float("inf")


class AssignmentConverter:
    """将离散 assignment 转换为 DMDE 统一基因编码。

    使用方式::

        converter = AssignmentConverter(cost_matrix, n_uavs, n_targets)
        # 单个完整解转换
        solution = [{"uav": 0, "targets": [2]}, {"uav": 1, "targets": [0]}]
        result = converter.convert(solution)
        if result.individual is not None:
            print(f"cost={result.repaired_cost}, repaired={result.repaired}")
        # 批量转换多个完整解
        individuals, stats = converter.convert_batch([solution1, solution2])
    """

    def __init__(
        self,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
    ) -> None:
        self._cm = cost_matrix
        self._n = n_uavs
        self._m = n_targets

    @property
    def model_type(self) -> str:
        """根据 N 和 M 关系推断模型类型。"""
        if self._n == self._m:
            return "balanced"
        elif self._n > self._m:
            return "overloaded"
        else:
            return "srp"

    def convert(self, assignments: list[dict[str, Any]]) -> ConversionResult:
        """将一组完整 assignments 转换为 Individual（含完整 gene 序列）。

        对于 balanced/overloaded，assignments 列表包含 N 个字典，
        每个字典对应一个 UAV 的分配。
        对于 srp，assignments 列表包含 N 个字典，
        每个字典对应一个 UAV 的巡游路线。

        如果 assignments 存在 target 重复/缺失（LLM 常见错误），
        会自动尝试修复后再转换。

        Args:
            assignments: 一个完整解的 assignment 字典列表。

        Returns:
            ConversionResult，包含 Individual 和修复元数据。
        """
        if not assignments:
            return ConversionResult()

        raw_cost = self._compute_assignment_cost(assignments)

        # 先尝试直接转换
        result = self._try_convert(assignments)
        if result is not None:
            return ConversionResult(
                individual=result,
                raw_valid=True,
                repaired=False,
                repair_distance=0,
                raw_cost=raw_cost,
                repaired_cost=raw_cost,
            )

        # 直接转换失败 → 尝试修复 target 重复/缺失
        repaired, distance = self._repair_duplicates(assignments)
        if repaired is not None:
            repaired_cost = self._compute_assignment_cost(repaired)
            ind = self._try_convert(repaired)
            if ind is not None:
                logger.debug("修复后转换成功 (distance=%d)", distance)
                return ConversionResult(
                    individual=ind,
                    raw_valid=False,
                    repaired=True,
                    repair_distance=distance,
                    raw_cost=raw_cost,
                    repaired_cost=repaired_cost,
                )

        return ConversionResult(raw_cost=raw_cost)

    def _try_convert(self, assignments: list[dict[str, Any]]) -> Individual | None:
        """尝试转换 assignments（含验证）。"""
        # 逐个验证格式
        for a in assignments:
            is_valid, err_msg = self.validate(a)
            if not is_valid:
                logger.debug("Assignment 验证失败: %s", err_msg)
                return None

        # 验证全局约束
        batch_ok, batch_err = self.validate_batch(assignments)
        if not batch_ok:
            logger.debug("Batch 验证失败: %s", batch_err)
            return None

        mt = self.model_type
        try:
            if mt == "balanced":
                return self._convert_balanced(assignments)
            elif mt == "overloaded":
                return self._convert_overloaded(assignments)
            else:
                return self._convert_srp(assignments)
        except Exception as e:
            logger.debug("Assignment 转换失败: %s", e)
            return None

    def validate(self, assignment: dict[str, Any]) -> tuple[bool, str]:
        """验证 assignment 是否满足 model_type 约束。

        Args:
            assignment: 单个 assignment 字典。

        Returns:
            (is_valid, error_message)。
        """
        mt = self.model_type

        # 基础格式检查
        if not isinstance(assignment, dict):
            return False, "assignment 不是字典"

        uav = assignment.get("uav")
        targets = assignment.get("targets", [])

        if not isinstance(uav, int):
            return False, f"uav 不是整数: {uav}"
        if not isinstance(targets, list):
            return False, f"targets 不是列表: {targets}"
        if not all(isinstance(t, int) for t in targets):
            return False, "targets 中包含非整数元素"

        if mt == "balanced":
            return self._validate_balanced(assignment)
        elif mt == "overloaded":
            return self._validate_overloaded(assignment)
        else:
            return self._validate_srp(assignment)

    def validate_batch(self, assignments: list[dict[str, Any]]) -> tuple[bool, str]:
        """验证一组 assignments 整体是否满足 model_type 约束。

        与 validate() 不同，此方法检查 assignments 之间的全局约束
        （如 UAV 不重复、目标全覆盖等）。

        Args:
            assignments: assignment 字典列表。

        Returns:
            (is_valid, error_message)。
        """
        mt = self.model_type

        if not assignments:
            return False, "assignments 列表为空"

        # 先逐个验证格式
        for i, a in enumerate(assignments):
            ok, msg = self.validate(a)
            if not ok:
                return False, f"第 {i} 个 assignment 格式错误: {msg}"

        if mt == "balanced":
            return self._validate_batch_balanced(assignments)
        elif mt == "overloaded":
            return self._validate_batch_overloaded(assignments)
        else:
            return self._validate_batch_srp(assignments)

    def repair(self, assignment: dict[str, Any]) -> dict[str, Any]:
        """修复不可行的 assignment。

        对于单个 assignment 的格式问题进行修复（如目标越界、空 targets 等）。
        全局约束问题（如 UAV 重复）需要在 batch 层面处理。

        Args:
            assignment: 原始 assignment 字典。

        Returns:
            修复后的 assignment 字典。
        """
        mt = self.model_type
        a = dict(assignment)  # 浅拷贝

        # 确保 uav 是整数
        if not isinstance(a.get("uav"), int):
            try:
                a["uav"] = int(a["uav"])
            except (ValueError, TypeError):
                a["uav"] = 0

        # 确保 targets 是整数列表
        targets = a.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        cleaned = []
        for t in targets:
            try:
                cleaned.append(int(t))
            except (ValueError, TypeError):
                continue
        a["targets"] = cleaned

        # 确保 targets 非空
        if not a["targets"]:
            a["targets"] = [0]

        # 确保 targets 在合法范围内
        a["targets"] = [t % self._m for t in a["targets"]]

        # 确保 uav 在合法范围内
        if mt == "srp":
            a["uav"] = a["uav"] % self._n
        else:
            a["uav"] = a["uav"] % self._n

        return a

    def _repair_duplicates(
        self, assignments: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]] | None, int]:
        """修复 target 重复/缺失问题（LLM 常见错误）。

        balanced/overloaded 模型下，LLM 经常生成有重复 target 的解。
        本方法检测重复，将重复的 UAV 重新分配到缺失的 target。

        Args:
            assignments: 原始 assignment 字典列表。

        Returns:
            (repaired, distance) 元组。
            repaired: 修复后的列表，无法修复时为 None。
            distance: 修复时修改的 assignment 数量（0 表示无需修复）。
        """
        mt = self.model_type
        if mt == "srp":
            # SRP 的约束更复杂，暂不修复
            return None, 0

        if len(assignments) != self._n:
            return None, 0

        # 收集所有 target（守卫：跳过空 targets）
        all_targets: list[int] = []
        for a in assignments:
            t = a.get("targets", [])
            if not t:
                return None, 0  # 空 targets 无法修复
            all_targets.append(t[0])
        target_counts: dict[int, int] = {}
        for t in all_targets:
            target_counts[t] = target_counts.get(t, 0) + 1

        # 找出重复和缺失
        duplicated_uavs: list[int] = []  # 有重复 target 的 UAV 索引
        seen_targets: set[int] = set()
        for i, a in enumerate(assignments):
            t = a["targets"][0]
            if t in seen_targets:
                duplicated_uavs.append(i)
            else:
                seen_targets.add(t)

        if not duplicated_uavs:
            return None, 0  # 没有重复，不需要修复

        missing_targets = sorted(set(range(self._m)) - seen_targets)
        if len(missing_targets) != len(duplicated_uavs):
            # 缺失数 != 重复数，无法一对一修复
            logger.debug(
                "修复失败: %d 个重复 UAV, %d 个缺失 target",
                len(duplicated_uavs), len(missing_targets),
            )
            return None, 0

        # 修复：将重复 UAV 重分配到缺失 target（最小代价匹配）
        repaired = [dict(a) for a in assignments]  # 浅拷贝
        n_fix = len(duplicated_uavs)
        if n_fix == 1:
            # 单个重复：直接选代价最小的缺失 target
            uav_id = repaired[duplicated_uavs[0]]["uav"]
            best_tgt = min(missing_targets, key=lambda t: self._cm[uav_id, t])
            repaired[duplicated_uavs[0]] = {"uav": uav_id, "targets": [best_tgt]}
        else:
            # 多个重复：尝试最优匹配，回退到贪心
            try:
                from scipy.optimize import linear_sum_assignment
                cost_sub = np.array([
                    [self._cm[repaired[i]["uav"], t] for t in missing_targets]
                    for i in duplicated_uavs
                ])
                row_ind, col_ind = linear_sum_assignment(cost_sub)
                for r, c in zip(row_ind, col_ind):
                    uav_id = repaired[duplicated_uavs[r]]["uav"]
                    repaired[duplicated_uavs[r]] = {"uav": uav_id, "targets": [missing_targets[c]]}
            except ImportError:
                # scipy 不可用，回退贪心
                remaining = list(missing_targets)
                for idx in duplicated_uavs:
                    uav_id = repaired[idx]["uav"]
                    best_tgt = min(remaining, key=lambda t: self._cm[uav_id, t])
                    repaired[idx] = {"uav": uav_id, "targets": [best_tgt]}
                    remaining.remove(best_tgt)

        logger.debug(
            "修复 %d 个重复 target: UAV %s → target %s",
            len(duplicated_uavs),
            [repaired[i]["uav"] for i in duplicated_uavs],
            [repaired[i]["targets"][0] for i in duplicated_uavs],
        )
        return repaired, n_fix

    def _compute_assignment_cost(self, assignments: list[dict[str, Any]]) -> float:
        """计算 assignments 的代价值（从 cost_matrix 直接求和）。

        不经过 fitness_evaluator，仅用于修复前后 cost 对比。
        """
        total = 0.0
        for a in assignments:
            uav = a.get("uav", 0)
            targets = a.get("targets", [])
            if not targets:
                continue
            for t in targets:
                if 0 <= uav < self._cm.shape[0] and 0 <= t < self._cm.shape[1]:
                    total += float(self._cm[uav, t])
        return total

    # ── 转换逻辑 ──────────────────────────────────────────────

    def _convert_balanced(self, assignments: list[dict[str, Any]]) -> Individual:
        """balanced (N=M): 每个 {"uav": i, "targets": [j]} → Gene(i, j, C_UT[i][j])。

        assignments 包含 N 个字典，每个对应一个 UAV 的分配。
        """
        genes = []
        for a in assignments:
            uav = a["uav"]
            target = a["targets"][0]
            cost = float(self._cm[uav, target])
            genes.append(Gene(uav_id=uav, target_id=target, cost=cost))
        return Individual(genes=genes, model_type="balanced")

    def _convert_overloaded(self, assignments: list[dict[str, Any]]) -> Individual:
        """overloaded (N>M): 同 balanced 格式，N 个 assignment 各含 1 个 target。"""
        genes = []
        for a in assignments:
            uav = a["uav"]
            target = a["targets"][0]
            cost = float(self._cm[uav, target])
            genes.append(Gene(uav_id=uav, target_id=target, cost=cost))
        return Individual(genes=genes, model_type="overloaded")

    def _convert_srp(self, assignments: list[dict[str, Any]]) -> Individual:
        """srp (N<M): UAV→Target + Target→Target 巡游基因序列。

        assignments 包含 N 个字典，每个对应一个 UAV 的巡游路线：
        {"uav": i, "targets": [j1, j2, j3]} →
            Gene(uav=i, target=j1, cost=C_UT[i][j1])
            Gene(uav=-1, target=j2, cost=C_TT[j1][j2])
            Gene(uav=-1, target=j3, cost=C_TT[j2][j3])
        """
        genes = []
        for a in assignments:
            uav = a["uav"]
            targets = a["targets"]
            for seq, tgt in enumerate(targets):
                if seq == 0:
                    cost = float(self._cm[uav, tgt])
                    genes.append(Gene(uav_id=uav, target_id=tgt, cost=cost))
                else:
                    prev_tgt = targets[seq - 1]
                    cost = float(self._cm[self._n + prev_tgt, tgt])
                    genes.append(Gene(uav_id=-1, target_id=tgt, cost=cost))
        return Individual(genes=genes, model_type="srp")

    # ── 单个 assignment 验证 ──────────────────────────────────

    def _validate_balanced(self, assignment: dict[str, Any]) -> tuple[bool, str]:
        """balanced: targets 恰好 1 个元素，U/T 在范围内。"""
        uav = assignment["uav"]
        targets = assignment["targets"]

        if len(targets) != 1:
            return False, f"balanced 要求 targets 长度为 1，实际为 {len(targets)}"
        if not (0 <= uav < self._n):
            return False, f"uav={uav} 超出范围 [0, {self._n})"
        if not (0 <= targets[0] < self._m):
            return False, f"target={targets[0]} 超出范围 [0, {self._m})"
        return True, ""

    def _validate_overloaded(self, assignment: dict[str, Any]) -> tuple[bool, str]:
        """overloaded: targets 恰好 1 个元素，U/T 在范围内。"""
        uav = assignment["uav"]
        targets = assignment["targets"]

        if len(targets) != 1:
            return False, f"overloaded 要求 targets 长度为 1，实际为 {len(targets)}"
        if not (0 <= uav < self._n):
            return False, f"uav={uav} 超出范围 [0, {self._n})"
        if not (0 <= targets[0] < self._m):
            return False, f"target={targets[0]} 超出范围 [0, {self._m})"
        return True, ""

    def _validate_srp(self, assignment: dict[str, Any]) -> tuple[bool, str]:
        """srp: targets 长度 ≥1，有序，U/T 在范围内。"""
        uav = assignment["uav"]
        targets = assignment["targets"]

        if len(targets) < 1:
            return False, "srp 要求 targets 长度 ≥ 1"
        if not (0 <= uav < self._n):
            return False, f"uav={uav} 超出范围 [0, {self._n})"
        for t in targets:
            if not (0 <= t < self._m):
                return False, f"target={t} 超出范围 [0, {self._m})"
        if len(targets) != len(set(targets)):
            return False, "srp 中 targets 不应有重复"
        return True, ""

    # ── batch 验证 ────────────────────────────────────────────

    def _validate_batch_balanced(
        self, assignments: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """balanced batch: U 不重复, T 不重复, 长度 == N。"""
        if len(assignments) != self._n:
            return False, f"balanced 要求 {self._n} 个 assignment，实际 {len(assignments)}"

        uavs = [a["uav"] for a in assignments]
        targets = [a["targets"][0] for a in assignments]

        if len(set(uavs)) != len(uavs):
            return False, f"UAV 有重复: {uavs}"
        if len(set(targets)) != len(targets):
            return False, f"Target 有重复: {targets}"
        if set(targets) != set(range(self._m)):
            return False, f"Target 未全覆盖: 缺少 {set(range(self._m)) - set(targets)}"
        return True, ""

    def _validate_batch_overloaded(
        self, assignments: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """overloaded batch: U 不重复, 每个 T 至少出现一次, 长度 == N。"""
        if len(assignments) != self._n:
            return False, f"overloaded 要求 {self._n} 个 assignment，实际 {len(assignments)}"

        uavs = [a["uav"] for a in assignments]
        targets = [a["targets"][0] for a in assignments]

        if len(set(uavs)) != len(uavs):
            return False, f"UAV 有重复: {uavs}"
        if set(targets) != set(range(self._m)):
            missing = set(range(self._m)) - set(targets)
            return False, f"Target 未全覆盖: 缺少 {missing}"
        return True, ""

    def _validate_batch_srp(
        self, assignments: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """srp batch: 每个 U 至少出现一次, T 不重复。"""
        # 收集所有 UAV 和 Target
        all_uavs = set()
        all_targets = set()

        for a in assignments:
            all_uavs.add(a["uav"])
            for t in a["targets"]:
                if t in all_targets:
                    return False, f"Target {t} 在 srp 中重复出现"
                all_targets.add(t)

        if set(range(self._n)) - all_uavs:
            missing = set(range(self._n)) - all_uavs
            return False, f"UAV 未全覆盖: 缺少 {missing}"
        if set(range(self._m)) - all_targets:
            missing = set(range(self._m)) - all_targets
            return False, f"Target 未全覆盖: 缺少 {missing}"
        return True, ""

    # ── 批量转换 ──────────────────────────────────────────────

    def convert_batch(
        self, solutions: list[list[dict[str, Any]]]
    ) -> tuple[list[Individual], dict[str, Any]]:
        """批量转换多个完整解为 Individuals。

        每个完整解是一个 assignment 字典列表（一个 LLM 响应中的所有 assignments）。
        转换时会验证每个完整解的全局约束，并记录修复统计。

        Args:
            solutions: 完整解列表，每个元素是一组 assignment 字典。

        Returns:
            (individuals, stats) 元组。
            individuals: 成功转换的 Individual 列表。
            stats: 转换统计字典，包含：
                - n_raw_valid: LLM 原始输出直接可行的数量
                - n_repaired: 经过修复才可行的数量
                - n_failed: 无法转换的数量
                - repair_distances: 每个修复解的修改距离列表
                - raw_costs: 所有可行解的修复前代价值列表
                - repaired_costs: 所有可行解的修复后代价值列表
        """
        individuals = []
        n_raw_valid = 0
        n_repaired = 0
        n_failed = 0
        repair_distances: list[int] = []
        raw_costs: list[float] = []
        repaired_costs: list[float] = []

        for sol_idx, solution in enumerate(solutions):
            result = self.convert(solution)
            if result.individual is not None:
                individuals.append(result.individual)
                raw_costs.append(result.raw_cost)
                repaired_costs.append(result.repaired_cost)
                if result.raw_valid:
                    n_raw_valid += 1
                elif result.repaired:
                    n_repaired += 1
                    repair_distances.append(result.repair_distance)
            else:
                n_failed += 1
                logger.debug("跳过无效完整解 #%d: %s", sol_idx, solution)

        stats = {
            "n_raw_valid": n_raw_valid,
            "n_repaired": n_repaired,
            "n_failed": n_failed,
            "repair_distances": repair_distances,
            "raw_costs": raw_costs,
            "repaired_costs": repaired_costs,
        }
        return individuals, stats