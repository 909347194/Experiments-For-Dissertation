#!/usr/bin/env python3
"""检查「会被渲染的文本」里是否混入了中文。

为什么需要
----------
期刊要求英文稿，且 `pdflatex` 无法渲染中文：一旦中文出现在正文、宏参数
（如 ``\\todo{...}``）或 bib 字段里，编译会直接报
``Unicode character ... not set up for use with LaTeX``。
中文只允许出现在 ``%`` 注释里。

做法
----
按行去掉注释（第一个未被 ``\\`` 转义的 ``%`` 之后的内容）与宏定义行里的
注释部分，再检查剩余文本是否含 CJK 字符。

用法
----
    python3 tools/check_no_cjk.py          # 检查默认范围
退出码：0 = 干净，1 = 发现需要清理的中文。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
TARGETS = ["main.tex", "config", "sections", "frontmatter", "bib"]
SUFFIXES = {".tex", ".bib", ".sty"}

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
# 第一个未被转义的 %
COMMENT_RE = re.compile(r"(?<!\\)%")

# \verb / \verb* 内的内容按字面处理，先替换掉避免误判
VERB_RE = re.compile(r"\\verb\*?(?P<d>.)(?P<body>.*?)(?P=d)")


def strip_comment(line: str) -> str:
    m = COMMENT_RE.search(line)
    return line[: m.start()] if m else line


def iter_files() -> list[Path]:
    files: list[Path] = []
    for t in TARGETS:
        p = PAPER_DIR / t
        if p.is_file() and p.suffix in SUFFIXES:
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in SUFFIXES:
                    files.append(f)
    return files


def main() -> int:
    bad: list[tuple[Path, int, str]] = []
    for f in iter_files():
        for i, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = VERB_RE.sub("", raw)
            line = strip_comment(line)
            if CJK_RE.search(line):
                bad.append((f, i, line.strip()))

    if not bad:
        print(f"[cjk] 检查 {len(iter_files())} 个文件：正文无中文（注释中的中文允许）")
        return 0

    print(f"[cjk] 发现 {len(bad)} 处会进入排版的 CJK 字符，pdflatex 会报错：")
    for f, i, line in bad:
        print(f"  {f.relative_to(PAPER_DIR)}:{i}: {line[:100]}")
    print("[cjk] 请把中文移到 % 注释中，或改写为英文")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
