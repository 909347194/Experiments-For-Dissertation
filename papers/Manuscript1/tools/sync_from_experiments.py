#!/usr/bin/env python3
"""从 experiments/ 同步实验生成的表格与图片到论文工程。

背景
----
``experiments/exp_ablation_study/analyze/`` 会生成
``experiments/exp_ablation_study/figures/ablation_tables.tex``：一份**完整独立
的中文文档**（``article`` + ``ctex``，需 xelatex 编译），以及若干 ``*.png`` 结果图。

论文正文是英文、用 pdflatex 编译，因此那份文件不能直接 ``\\input``：
  1. 它自带 ``\\documentclass`` / ``\\begin{document}``，是完整文档而非片段；
  2. caption 与表头是中文，pdflatex 无法渲染；
  3. ``ctex`` 与 ``elsarticle`` 混用会冲突。

本脚本把其中每个 ``table`` 环境抽成**一个独立片段文件**，按 ``\\label{tab:xxx}``
命名为 ``tbf_xxx.tex``，统一放在 ``tables/imported/``（该目录不入库，可随时重新生成）。
翻译/调整后请在 ``tables/`` 下另存为正式文件 ``tbl_<用途>.tex``，再在正文中 ``\\input``。

用法
----
    python3 tools/sync_from_experiments.py                 # 只导出表格片段
    python3 tools/sync_from_experiments.py --copy-figures   # 同时拷贝结果图到 figures/
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent          # papers/Manuscript1
REPO_ROOT = PAPER_DIR.parents[1]       # 仓库根目录

EXP_FIG_DIR = REPO_ROOT / "experiments" / "exp_ablation_study" / "figures"
SRC_TEX = EXP_FIG_DIR / "ablation_tables.tex"

TABLES_DIR = PAPER_DIR / "tables"
OUT_DIR = TABLES_DIR / "imported"
FIGURES_DIR = PAPER_DIR / "figures"

TABLE_ENV_RE = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.DOTALL)
LABEL_RE = re.compile(r"\\label\{(?:tab:)?([^}]+)\}")
CAPTION_RE = re.compile(r"\\caption\{(.+?)\}", re.DOTALL)


def slugify(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()
    return text or "table"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-figures", action="store_true",
                    help="把实验结果图（*.png/*.pdf）拷贝到 figures/")
    args = ap.parse_args()

    if not SRC_TEX.exists():
        print(f"[sync] 找不到源文件：{SRC_TEX}", file=sys.stderr)
        print("[sync] 请先在 experiments/ 下运行："
              "uv run python run_ablation.py --runs 1", file=sys.stderr)
        return 1

    tables = TABLE_ENV_RE.findall(SRC_TEX.read_text(encoding="utf-8"))
    if not tables:
        print("[sync] 源文件中没有找到 table 环境，什么都没做")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []

    for idx, block in enumerate(tables, 1):
        block = re.sub(r"\n{3,}", "\n\n", block).strip()
        label = LABEL_RE.search(block)
        stem = slugify(label.group(1)) if label else f"table_{idx:02d}"
        out = OUT_DIR / f"tbl_{stem}.tex"

        caption = CAPTION_RE.search(block)
        cap_txt = re.sub(r"\s+", " ", caption.group(1)).strip() if caption else "(no caption)"

        header = (
            "% ============================================================================\n"
            f"%  自动导出，来源: experiments/exp_ablation_study/figures/ablation_tables.tex\n"
            f"%  原始 caption: {cap_txt}\n"
            "%\n"
            "%  【待办】以下内容仍为中文（caption / 表头 / 行标签），pdflatex 无法编译。\n"
            "%          请在本文件基础上翻译为英文，另存为 tables/tbl_<用途>.tex 后再引用；\n"
            "%          本目录（tables/imported/）不入库，可随时用 make sync 重新生成。\n"
            "% ============================================================================\n\n"
        )
        out.write_text(header + block + "\n", encoding="utf-8")
        written.append((out.name, cap_txt))

    print(f"[sync] 已导出 {len(written)} 张表格 -> {OUT_DIR.relative_to(REPO_ROOT)}/")
    for name, cap in written:
        print(f"        {name}   ({cap[:48]}{'...' if len(cap) > 48 else ''})")
    print("[sync] 提醒：需翻译为英文后另存为 tables/tbl_*.tex 才可 \input")

    if args.copy_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        copied = 0
        for pat in ("*.png", "*.pdf"):
            for src in sorted(EXP_FIG_DIR.glob(pat)):
                shutil.copy2(src, FIGURES_DIR / src.name)
                copied += 1
                print(f"[sync] 图片 -> figures/{src.name}")
        if not copied:
            print("[sync] 未发现可拷贝的图片")
    else:
        found = sorted(p.name for p in EXP_FIG_DIR.glob("*.png"))
        if found:
            print("[sync] figures/ 可用图片（加 --copy-figures 自动拷贝）：")
            for name in found:
                print(f"        {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
