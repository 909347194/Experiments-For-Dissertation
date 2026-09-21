# -*- coding: utf-8 -*-
"""消融实验结果分析包。

用法：python -m analyze
"""
from .stats import (
    compute_stats,
    compute_convergence_gens,
    compute_f_stats,
    compute_gmr_stats,
    compute_parameter_coupling,
    mannwhitney_test,
    p_mark,
)
from .tables import (
    generate_markdown_table,
    generate_latex_table1,
    generate_latex_table2,
    generate_latex_table3,
    generate_llm_decision_summary,
    generate_convergence_speed_table,
    generate_latex_convergence_speed_table,
    generate_f_stats_table,
    generate_gmr_stats_table,
    generate_parameter_coupling_table,
)
