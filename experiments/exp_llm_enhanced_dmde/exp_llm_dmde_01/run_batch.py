# -*- coding: utf-8 -*-
"""run_batch.py — 批量运行单个实验

用法：
    # 默认运行2次
    python run_batch.py

    # 指定运行次数
    python run_batch.py --runs 30

    # 通过环境变量控制运行次数
    EXP_N_RUNS=30 python run.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def main():
    parser = argparse.ArgumentParser(description="批量运行 LLM-DMDE 实验")
    parser.add_argument("--runs", type=int, default=2, help="运行次数 (默认: 2)")
    args = parser.parse_args()

    run_py = SCRIPT_DIR / "run.py"
    if not run_py.exists():
        print(f"❌ run.py 不存在: {run_py}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"▶ 运行 LLM-DMDE 实验")
    print(f"  脚本: {run_py}")
    print(f"  运行次数: {args.runs}")
    print(f"{'='*60}\n")

    import os
    env = {**os.environ, "EXP_N_RUNS": str(args.runs)}

    t0 = time.time()
    result = subprocess.run([sys.executable, str(run_py)], env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n❌ 实验失败 (exit={result.returncode})")
        sys.exit(1)

    print(f"\n✅ 实验完成 ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
