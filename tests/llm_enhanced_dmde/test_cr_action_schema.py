# -*- coding: utf-8 -*-
"""test_cr_action_schema.py — CR 动作空间两级化（hold / set）回归测试

背景（v2 实测缺陷）：
    旧 schema 强制 LLM 每次输出一个 CR 数字，导致 CR_t = flip(CR_{t-1})：
    corr(CR_t, CR_{t-1}) = -0.967，翻转率 0.98，且在所有证据条件下翻转率
    均为 1.00（包括 prompt 明说"状态未变、应保持不变"的 49 次）。
    根因：模型没有「不改」的出口；解析失败时旧代码回退到一个新的 CR 值（0.5），
    等于默认翻转。

本文件锁定修复后的行为：
    - hold 是默认路径：解析失败 / 字段缺失 / 格式非法一律回退 hold（cr=None）
    - 只有 cr_action == "set" 且给出合法数值时才采纳新 CR
    - cr_action == "hold" 时，模型即便仍填了 cr 值也必须忽略
    - system prompt 在冻结时注入冻结段，非冻结时不注入
"""

from __future__ import annotations

import json
import pytest

from algorithms.algorithm_llm_enhanced_dmde.llm.modules.search_controller import (
    LLMSearchControllerModule,
)
from algorithms.algorithm_llm_enhanced_dmde.llm.prompts import (
    get_search_controller_prompt,
)
from algorithms.algorithm_llm_enhanced_dmde.llm.base_module import ModuleState


def make_module(config=None):
    return LLMSearchControllerModule(llm_client=None, config=config or {})


def state_with(cr=0.8, frozen=False, reason=""):
    st = ModuleState(generation=100, max_generations=1000)
    st.cr = cr
    st.cr_frozen = frozen
    st.cr_frozen_reason = reason
    return st


# ── 解析：set 路径 ────────────────────────────────────────────────
def test_set_action_parses_cr_value():
    m = make_module()
    out = json.dumps({
        "evidence_read": "df=+5.4% while df_shadow=0%",
        "cr_action": "set",
        "cr": 0.8,
        "restart_fraction": 0.1,
        "reasoning": "improvement under current CR",
    })
    d = m.parse_response(out)
    assert d["cr_action"] == "set"
    assert d["cr"] == 0.8
    assert d["restart_fraction"] == 0.1


def test_set_action_clamps_to_nearest_choice():
    m = make_module()
    d = m.parse_response(json.dumps({"cr_action": "set", "cr": 0.77}))
    assert d["cr_action"] == "set"
    assert d["cr"] == 0.8  # 钳位到最近候选


def test_set_action_with_invalid_number_falls_back_to_hold():
    """声明 set 却给出非法数值 → 回退 hold，而不是猜一个值。"""
    m = make_module()
    d = m.parse_response(json.dumps({"cr_action": "set", "cr": "high"}))
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


def test_set_action_with_missing_cr_falls_back_to_hold():
    m = make_module()
    d = m.parse_response(json.dumps({"cr_action": "set"}))
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


# ── 解析：hold 路径 ───────────────────────────────────────────────
def test_hold_action_returns_none_cr():
    m = make_module()
    d = m.parse_response(json.dumps({
        "evidence_read": "df and df_shadow both within noise",
        "cr_action": "hold",
        "cr": None,
        "restart_fraction": 0.2,
    }))
    assert d["cr_action"] == "hold"
    assert d["cr"] is None
    assert d["restart_fraction"] == 0.2


def test_hold_action_ignores_stray_cr_value():
    """防御：模型说 hold 却仍填了 cr —— 必须忽略该值。"""
    m = make_module()
    d = m.parse_response(json.dumps({"cr_action": "hold", "cr": 0.3}))
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


def test_hold_synonyms_are_normalized():
    m = make_module()
    for word in ["keep", "unchanged", "none", "false", "no", "freeze", "HOLD"]:
        d = m.parse_response(json.dumps({"cr_action": word, "cr": 0.4}))
        assert d["cr_action"] == "hold", f"{word} 应归一化为 hold"
        assert d["cr"] is None


def test_set_synonyms_are_normalized():
    m = make_module()
    for word in ["change", "adjust", "update", "modify", "SET"]:
        d = m.parse_response(json.dumps({"cr_action": word, "cr": 0.6}))
        assert d["cr_action"] == "set", f"{word} 应归一化为 set"
        assert d["cr"] == 0.6


# ── 解析：兜底路径（本 bug 的核心回归点）──────────────────────────
def test_unparseable_output_defaults_to_hold_not_flip():
    """关键回归：无 JSON 时不得返回新的 CR 值（旧版返回 0.5 = 默认翻转）。"""
    m = make_module()
    d = m.parse_response("I think we should keep exploring.")
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


def test_invalid_json_defaults_to_hold():
    m = make_module()
    d = m.parse_response("{cr: 0.8,}")
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


def test_non_object_json_defaults_to_hold():
    m = make_module()
    d = m.parse_response("[0.8, 0.1]")
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


def test_bare_cr_without_action_is_implicit_set():
    """旧 schema 风格（只给 cr）：视为隐式 set，保持向后兼容。"""
    m = make_module()
    d = m.parse_response(json.dumps({"cr": 0.7, "restart_fraction": 0.0}))
    assert d["cr_action"] == "set_implicit"
    assert d["cr"] == 0.7


def test_no_action_and_no_cr_defaults_to_hold():
    """既没说要改、也没给值 → hold。"""
    m = make_module()
    d = m.parse_response(json.dumps({"restart_fraction": 0.1}))
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


def test_empty_output_defaults_to_hold():
    m = make_module()
    d = m.parse_response("")
    assert d["cr_action"] == "hold"
    assert d["cr"] is None


# ── apply_decision：hold 不得改动 CR ──────────────────────────────
def test_apply_hold_keeps_current_cr():
    m = make_module()
    st = state_with(cr=0.8)
    m.apply_decision({"cr_action": "hold", "cr": None, "restart_fraction": 0.2}, st)
    assert st.cr == 0.8
    assert st.extra["llm_cr_action"] == "hold"


def test_apply_set_changes_cr():
    m = make_module()
    st = state_with(cr=0.8)
    m.apply_decision({"cr_action": "set", "cr": 0.5, "restart_fraction": 0.0}, st)
    assert st.cr == 0.5
    assert st.extra["llm_cr_action"] == "set"


def test_apply_ignores_cr_when_action_is_hold():
    """即使决策里带了 cr，只要 action 是 hold 就不能改。"""
    m = make_module()
    st = state_with(cr=0.8)
    m.apply_decision({"cr_action": "hold", "cr": 0.2, "restart_fraction": 0.0}, st)
    assert st.cr == 0.8


# ── prompt：冻结段注入 ────────────────────────────────────────────
def test_prompt_includes_frozen_section_only_when_frozen():
    normal = get_search_controller_prompt(cr_choices=[0.5, 0.8], shadow_cr=0.5)
    frozen = get_search_controller_prompt(
        cr_choices=[0.5, 0.8], shadow_cr=0.5, cr_frozen=True,
    )
    assert "FROZEN" in frozen
    assert "FROZEN" not in normal


def test_frozen_prompt_forces_cr_null():
    frozen = get_search_controller_prompt(cr_choices=[0.5, 0.8], cr_frozen=True)
    assert "null (CR is FROZEN" in frozen


def test_prompt_requires_two_level_action_field():
    p = get_search_controller_prompt(cr_choices=[0.5, 0.8])
    assert "cr_action" in p
    assert '"hold"' in p
    assert "evidence_read" in p
    # 在输出格式块内部，证据判读必须排在动作之前、动作排在取值之前
    # （自回归顺序 = 决策顺序，迫使模型先提交证据再命名数值）
    fmt_block = p[p.index("## Decision Format"):]
    assert (fmt_block.index("evidence_read")
            < fmt_block.index("cr_action")
            < fmt_block.index('"cr"'))


def test_prompt_lists_invalid_reasons_for_changing_cr():
    """prompt 必须显式禁止「被咨询 / 用久了 / 想探索」这类伪理由。"""
    p = get_search_controller_prompt(cr_choices=[0.5, 0.8])
    assert "NOT valid reasons" in p
    assert "I was consulted" in p


def test_prompt_includes_confound_section_when_restart_was_applied():
    """上一轮执行过 restart → df vs df_shadow 被混淆，必须告知模型不可用。"""
    clean = get_search_controller_prompt(cr_choices=[0.5, 0.8], shadow_cr=0.5)
    confounded = get_search_controller_prompt(
        cr_choices=[0.5, 0.8], shadow_cr=0.5, restart_confounded=True,
    )
    assert "CONFOUNDED" in confounded
    assert "CONFOUNDED" not in clean


def test_prompt_confound_section_demands_hold_without_cr_evidence():
    confounded = get_search_controller_prompt(cr_choices=[0.5, 0.8], restart_confounded=True)
    assert 'cr_action: "hold"' in confounded


def test_non_frozen_prompt_keeps_backward_compatible_defaults():
    """未启用冻结/影子时 prompt 仍可生成（向后兼容）。"""
    p = get_search_controller_prompt()
    assert "cr_action" in p
    assert "FROZEN" not in p
