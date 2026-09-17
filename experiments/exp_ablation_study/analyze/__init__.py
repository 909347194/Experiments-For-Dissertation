# -*- coding: utf-8 -*-
"""消融实验结果分析包。

用法：python -m analyze
"""
from .stats import compute_stats, mannwhitney_test, p_mark
from .tables import (
    generate_markdown_table,
    generate_latex_table1,
    generate_latex_table2,
    generate_latex_table3,
    generate_llm_decision_summary,
)

__all__ = [
    "compute_stats",
    "mannwhitney_test",
    "p_mark",
    "generate_markdown_table",
    "generate_latex_table1",
    "generate_latex_table2",
    "generate_latex_table3",
    "generate_llm_decision_summary",
]