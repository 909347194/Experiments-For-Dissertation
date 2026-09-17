# -*- coding: utf-8 -*-
"""消融实验结果分析 — 入口脚本。

实际逻辑已拆分到 analyze/ 包中。
运行：python analyze.py  或  python -m analyze
"""
from analyze.main import main

if __name__ == "__main__":
    main()