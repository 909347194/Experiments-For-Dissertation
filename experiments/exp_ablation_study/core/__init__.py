# -*- coding: utf-8 -*-
"""core — 消融实验核心模块包"""
from .scenario import build_scenario
from .runner import run_dmde_baseline, run_llm_dmde
from .results import load_results, load_modules_config
from .recorder import RunRecorder
from .config import ExperimentConfig

# 向后兼容：save_results 委托给 RunRecorder.save_results
def save_results(results, output_dir):
    return RunRecorder.save_results(results, output_dir)

__all__ = [
    "build_scenario",
    "run_dmde_baseline", "run_llm_dmde",
    "load_results", "load_modules_config", "save_results",
    "RunRecorder", "ExperimentConfig",
]