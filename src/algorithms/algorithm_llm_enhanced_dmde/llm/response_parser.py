# -*- coding: utf-8 -*-
"""response_parser.py — LLM 响应解析器

职责：
    将 LLM 输出的 JSON 字符串解析为结构化的 LLMDecision 数据类。
    处理各种异常情况（非法 JSON、缺失字段、类型错误等），
    确保即使 LLM 输出格式有误也能返回安全的默认决策。

使用方式::

    parser = ResponseParser()
    decision = parser.parse(llm_output_json)
    print(decision.operator_strategy)  # "rand/1"
    print(decision.cr_adjustment)      # 0.1
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLMDecision 数据类
# ---------------------------------------------------------------------------

@dataclass
class LLMDecision:
    """LLM 决策结果。

    Attributes:
        operator_strategy:      DE 算子策略名称（如 "rand/1", "best/2", "current-to-pbest"）。
        cr_adjustment:          交叉率调整值（正 = 更多探索，负 = 更多开发）。
        temperature_adjustment: 温度调整值（正 = 更随机匹配，负 = 更贪心匹配）。
        extinction_trigger:     是否强制触发灭绝（None = 使用原始 GMR 判断）。
        reasoning:              LLM 的决策理由。
    """

    operator_strategy: str = "default"
    cr_adjustment: float | None = None
    temperature_adjustment: float | None = None
    extinction_trigger: bool | None = None
    reasoning: str = ""

    def __repr__(self) -> str:
        return (
            f"LLMDecision(strategy={self.operator_strategy!r}, "
            f"cr_adj={self.cr_adjustment}, "
            f"temp_adj={self.temperature_adjustment}, "
            f"extinct={self.extinction_trigger})"
        )


# ---------------------------------------------------------------------------
# 有效算子策略列表
# ---------------------------------------------------------------------------

VALID_STRATEGIES = frozenset({
    "default",
    "rand/1",
    "rand/2",
    "best/1",
    "best/2",
    "current-to-pbest/1",
    "rand-to-best/1",
})


# ---------------------------------------------------------------------------
# ResponseParser
# ---------------------------------------------------------------------------

class ResponseParser:
    """LLM 响应解析器。

    从 LLM 输出的 JSON 字符串中提取决策信息，
    并进行类型校验和范围约束。
    """

    def parse(self, llm_output: str) -> LLMDecision:
        """解析 LLM 输出为 LLMDecision。

        支持以下容错：
        1. 从 markdown 代码块中提取 JSON。
        2. 处理 trailing commas 等常见 JSON 格式问题。
        3. 缺失字段使用默认值。
        4. 类型不匹配时尝试转换。
        5. 完全解析失败时返回默认决策。

        Args:
            llm_output: LLM 输出的原始文本。

        Returns:
            LLMDecision 实例。
        """
        if not llm_output or not llm_output.strip():
            return LLMDecision(reasoning="Empty LLM output")

        # 尝试提取 JSON
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return LLMDecision(
                reasoning=f"Failed to extract JSON from LLM output: {llm_output[:200]}",
            )

        # 解析 JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error: %s. Raw: %s", e, json_str[:200])
            return LLMDecision(
                reasoning=f"JSON decode error: {e}",
            )

        if not isinstance(data, dict):
            return LLMDecision(
                reasoning=f"Expected JSON object, got {type(data).__name__}",
            )

        # 提取字段
        return self._extract_decision(data)

    def _extract_json(self, text: str) -> str | None:
        """从文本中提取 JSON 字符串。

        支持从 markdown 代码块中提取。

        Args:
            text: 原始文本。

        Returns:
            JSON 字符串，提取失败返回 None。
        """
        text = text.strip()

        # 尝试直接解析
        if text.startswith("{"):
            return text

        # 从 markdown 代码块中提取
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 找到第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

        return None

    def _extract_decision(self, data: dict[str, Any]) -> LLMDecision:
        """从解析后的字典中提取 LLMDecision。

        Args:
            data: 解析后的 JSON 字典。

        Returns:
            LLMDecision 实例。
        """
        # operator_strategy
        strategy = data.get("operator_strategy", "default")
        if not isinstance(strategy, str):
            strategy = str(strategy)
        if strategy not in VALID_STRATEGIES:
            logger.warning("Unknown operator strategy: %s, using default", strategy)
            strategy = "default"

        # cr_adjustment
        cr_adj = self._parse_float_range(
            data.get("cr_adjustment"), -0.3, 0.3, "cr_adjustment"
        )

        # temperature_adjustment
        temp_adj = self._parse_float_range(
            data.get("temperature_adjustment"), -0.3, 0.3, "temperature_adjustment"
        )

        # extinction_trigger
        extinction = data.get("extinction_trigger")
        if extinction is not None and not isinstance(extinction, bool):
            # 尝试转换
            if isinstance(extinction, str):
                extinction = extinction.lower() in ("true", "1", "yes")
            else:
                extinction = None

        # reasoning
        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)

        return LLMDecision(
            operator_strategy=strategy,
            cr_adjustment=cr_adj,
            temperature_adjustment=temp_adj,
            extinction_trigger=extinction,
            reasoning=reasoning,
        )

    @staticmethod
    def _parse_float_range(
        value: Any,
        min_val: float,
        max_val: float,
        field_name: str,
    ) -> float | None:
        """解析浮点值并约束到指定范围。

        Args:
            value:      原始值。
            min_val:    最小值。
            max_val:    最大值。
            field_name: 字段名（用于日志）。

        Returns:
            约束后的浮点值，解析失败返回 None。
        """
        if value is None:
            return None

        try:
            f = float(value)
        except (ValueError, TypeError):
            logger.warning("Cannot parse %s value: %s", field_name, value)
            return None

        if not (-1.0 <= f <= 1.0):
            logger.warning(
                "%s out of range [-1.0, 1.0]: %s, clamping",
                field_name, f,
            )

        # 约束到有效范围
        return max(min_val, min(max_val, f))
