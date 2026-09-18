#!/usr/bin/env python3
"""检查 Elsevier Highlights 是否满足投稿硬性要求。

规则
----
- 3--5 条；
- 每条（含空格）不超过 85 个字符；
- 不允许用缩写与"首次出现"的专有名词（本脚本只做长度与条数检查）。

用法
----
    python3 tools/check_highlights.py [frontmatter/highlights.tex]
退出码：0 = 通过，1 = 不满足要求。
"""

from __future__ import annotations

import sys
from pathlib import Path

MAX_LEN = 85
MIN_ITEMS, MAX_ITEMS = 3, 5
DEFAULT = Path(__file__).resolve().parent.parent / "frontmatter" / "highlights.tex"


def load_items(path: Path) -> list[str]:
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        # 容忍 Markdown 列表符号
        items.append(line.lstrip("-*").strip())
    return items


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[highlights] 找不到文件：{path}")
        return 1

    items = load_items(path)
    ok = True

    print(f"[highlights] 共 {len(items)} 条（要求 {MIN_ITEMS}--{MAX_ITEMS} 条）")
    if not (MIN_ITEMS <= len(items) <= MAX_ITEMS):
        print("  条数不符合要求")
        ok = False

    for i, it in enumerate(items, 1):
        n = len(it)
        flag = "OK" if n <= MAX_LEN else f"超长 {n - MAX_LEN}"
        mark = "  " if n <= MAX_LEN else "!!"
        if n > MAX_LEN:
            ok = False
        print(f"{mark} [{i}] {n:>3} 字符  {flag}")
        if n > MAX_LEN:
            print(f"      {it}")

    print("[highlights] 通过" if ok else "[highlights] 未通过，请修改后重试")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
