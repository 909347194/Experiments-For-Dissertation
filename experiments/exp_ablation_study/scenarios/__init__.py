# -*- coding: utf-8 -*-
"""scenarios — 场景定义包"""
from .s1_balanced import build_s1_cost_matrix_and_evaluator, S1_CONFIG
from .s2_srp import build_s2_cost_matrix_and_evaluator, S2_CONFIG

__all__ = [
    "build_s1_cost_matrix_and_evaluator", "S1_CONFIG",
    "build_s2_cost_matrix_and_evaluator", "S2_CONFIG",
]