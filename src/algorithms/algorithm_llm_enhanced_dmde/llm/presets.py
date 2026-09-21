# -*- coding: utf-8 -*-
"""presets.py — 语义化策略档位定义

设计目标：
    将 LLM 的决策从「输出 3 个标量数值」改为「选一个语义明确的策略档位」。
    每个档位 = {mutation_strategy, CR, F, gmr_mode} 的固定组合，
    语义清晰、可解释、消除数值微调的无效探索。

档位清单（6 个）：
    explore   — 全局探索：高 CR、高 F、rand/1、不开灭绝
    balanced  — 探索-开发均衡：中等参数、混合策略
    exploit   — 局部开发：低 CR、低 F、best/1、不开灭绝
    recover   — 停滞恢复：强制灭绝 + 探索型参数，跳出局部最优
    rand-1    — 纯探索策略：固定 rand/1，排除 best 干扰
    best-1    — 纯开发策略：固定 best/1，排除 rand 干扰

冻结/hold 逻辑：
    LLM 还可以选择 "hold"（保持当前档位不变），
    由 search_controller 的 parse_response 处理。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyPreset:
    """单个策略档位的参数定义。"""
    name: str                    # 档位 ID（英文小写，LLM 输出用）
    label: str                   # 中文标签（日志/可视化用）
    description: str             # 一句话语义说明（注入 prompt）
    mutation_strategy: str       # "rand/1" | "best/1" | "mixed"
    cr: float                    # 交叉率
    f: float                     # 缩放因子
    gmr_mode: str                # "auto" | "on" | "off"
    # 以下字段用于 prompt 中展示给 LLM
    when_to_use: str             # 适用场景说明


# ── 预设档位表 ────────────────────────────────────────────────

PRESETS: dict[str, StrategyPreset] = {
    "explore": StrategyPreset(
        name="explore",
        label="全局探索",
        description="高步长、高交叉率，侧重探索未知区域",
        mutation_strategy="rand/1",
        cr=0.8,
        f=1.2,
        gmr_mode="off",
        when_to_use=(
            "种群多样性低、搜索停滞、需要跳出当前区域时。"
            "DE/rand/1 用 3 个随机个体构造变异向量，不依赖最优个体，"
            "天然具有强探索能力。"
        ),
    ),
    "balanced": StrategyPreset(
        name="balanced",
        label="均衡",
        description="探索与开发兼顾的中间档位",
        mutation_strategy="mixed",
        cr=0.5,
        f=0.7,
        gmr_mode="auto",
        when_to_use=(
            "搜索状态不明确、没有强烈信号指向某一方向时。"
            "混合策略（rand/1 + best/2 按 CR 比例混合）"
            "在探索与开发之间自动平衡。"
        ),
    ),
    "exploit": StrategyPreset(
        name="exploit",
        label="局部开发",
        description="小步长、低交叉率，侧重精细搜索当前最优附近",
        mutation_strategy="best/1",
        cr=0.3,
        f=0.5,
        gmr_mode="off",
        when_to_use=(
            "搜索正在稳步改进、可行解占比高、需要加速收敛时。"
            "DE/best/1 以最优个体为基准构造变异向量，"
            "所有变异都朝最优方向收缩，收敛快。"
        ),
    ),
    "recover": StrategyPreset(
        name="recover",
        label="停滞恢复",
        description="强制灭绝重置 + 探索型参数，打破搜索僵局",
        mutation_strategy="rand/1",
        cr=0.8,
        f=1.0,
        gmr_mode="on",
        when_to_use=(
            "搜索深度停滞（stagnation_raw 很高）、"
            "其他档位均未带来改善时。"
            "GMR=on 强制灭绝最差个体并重新初始化，"
            "配合 rand/1 大步长注入多样性。"
            "⚠️ 会临时牺牲 fitness，仅在穷尽其他手段后使用。"
        ),
    ),
    "rand-1": StrategyPreset(
        name="rand-1",
        label="纯 rand/1",
        description="固定 DE/rand/1 策略，排除 best 干扰",
        mutation_strategy="rand/1",
        cr=0.6,
        f=0.8,
        gmr_mode="auto",
        when_to_use=(
            "想测试纯探索策略的效果、或当前混合策略中"
            "rand/1 贡献更大时。"
            "不依赖最优个体，每个个体独立探索，"
            "适合多样性充足但收敛方向不明确的阶段。"
        ),
    ),
    "best-1": StrategyPreset(
        name="best-1",
        label="纯 best/1",
        description="固定 DE/best/1 策略，排除 rand 干扰",
        mutation_strategy="best/1",
        cr=0.4,
        f=0.6,
        gmr_mode="auto",
        when_to_use=(
            "想测试纯开发策略的效果、或当前最优个体质量很高时。"
            "所有变异以最优个体为锚点，收敛速度快，"
            "但容易陷入局部最优。"
        ),
    ),
}

# ── hold 特殊值 ───────────────────────────────────────────────

HOLD_PRESET_NAME = "hold"

# ── 便捷函数 ──────────────────────────────────────────────────

def get_preset(name: str) -> StrategyPreset | None:
    """按名称获取预设档位。hold 返回 None。"""
    if name == HOLD_PRESET_NAME:
        return None
    return PRESETS.get(name)


def list_preset_names() -> list[str]:
    """返回所有档位名称（含 hold）。"""
    return [HOLD_PRESET_NAME] + list(PRESETS.keys())


def format_presets_for_prompt() -> str:
    """将所有档位格式化为 prompt 中的表格文本。"""
    lines = [
        "| Preset | Mutation | CR | F | GMR | When to use |",
        "|--------|----------|----|----|-----|-------------|",
    ]
    for p in PRESETS.values():
        lines.append(
            f"| **{p.name}** ({p.label}) | {p.mutation_strategy} | "
            f"{p.cr} | {p.f} | {p.gmr_mode} | {p.when_to_use} |"
        )
    return "\n".join(lines)


def format_presets_compact() -> str:
    """紧凑格式：每行一个档位，用于 system prompt。"""
    parts = []
    for p in PRESETS.values():
        parts.append(
            f"- **{p.name}** ({p.label}): {p.description}\n"
            f"  mutation={p.mutation_strategy}, CR={p.cr}, F={p.f}, GMR={p.gmr_mode}\n"
            f"  Use when: {p.when_to_use}"
        )
    return "\n".join(parts)