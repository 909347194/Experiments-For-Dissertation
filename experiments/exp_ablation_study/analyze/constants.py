# -*- coding: utf-8 -*-
"""共享常量：场景、配置、标签、颜色。"""

SCENARIOS = {
    "S1": "S1_balanced_N10_M10",
    "S2": "S2_srp_N10_M20",
}
CONFIGS = {
    "A0": "A0_dmde",
    "A1": "A1_cr_control",
    "A2": "A2_pop_init",
    "A3": "A3_full",
}
CONFIG_LABELS = {
    "A0": "Vanilla DMDE",
    "A1": "+CR Control",
    "A2": "+PopInit",
    "A3": "Full LLM-DMDE",
}
CONFIG_COLORS = {
    "A0": "#1f77b4",
    "A1": "#ff7f0e",
    "A2": "#2ca02c",
    "A3": "#d62728",
}
SCENARIO_LABELS = {
    "S1": r"balanced ($N=M=10$)",
    "S2": r"srp ($N=10, M=20$)",
}

# GMR 模式颜色（用于 GMR 模式分布图）
GMR_MODE_COLORS = {
    "auto": "#1f77b4",
    "on": "#d62728",
    "off": "#2ca02c",
}
GMR_MODE_LABELS = {
    "auto": "Auto (formula)",
    "on": "Force Extinction",
    "off": "No Extinction",
}