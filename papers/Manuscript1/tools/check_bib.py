#!/usr/bin/env python3
"""参考文献库体检：重复 key、必填字段缺失、TODO 占位条目。

用法：
    python3 tools/check_bib.py
退出码：0 = 通过（可能有提示），1 = 发现必须修的问题。
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

BIB_DIR = Path(__file__).resolve().parent.parent / "bib"

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),", re.IGNORECASE)
FIELD_RE = re.compile(r"^\s*(\w+)\s*=", re.MULTILINE)

REQUIRED = {
    "article": {"author", "title", "journal", "year"},
    "inproceedings": {"author", "title", "booktitle", "year"},
    "book": {"author", "title", "publisher", "year"},
    "misc": {"author", "title", "year"},
}


def strip_comments(text: str) -> str:
    """去掉整行注释（BibTeX 中 % 开头的行是注释）。"""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%"))


def main() -> int:
    if not BIB_DIR.is_dir():
        print(f"[bib] 目录不存在：{BIB_DIR}")
        return 1

    keys: dict[str, list[str]] = defaultdict(list)
    problems: list[str] = []
    todos: list[str] = []
    total = 0

    for bib in sorted(BIB_DIR.glob("*.bib")):
        raw = strip_comments(bib.read_text(encoding="utf-8"))
        for m in ENTRY_RE.finditer(raw):
            etype, key = m.group(1).lower(), m.group(2).strip()
            total += 1
            keys[key].append(bib.name)

            # 截取该条目的正文（到下一个 @ 之前）
            nxt = raw.find("@", m.end())
            body = raw[m.end(): nxt if nxt != -1 else len(raw)]
            fields = {f.lower() for f in FIELD_RE.findall(body)}

            missing = REQUIRED.get(etype, set()) - fields
            if missing:
                problems.append(f"  {key} ({etype} in {bib.name}) 缺少字段: "
                                f"{', '.join(sorted(missing))}")
            if re.search(r"\bTODO\b", body, re.IGNORECASE):
                todos.append(key)

    print(f"[bib] 共 {total} 条条目")

    dup = {k: v for k, v in keys.items() if len(v) > 1}
    if dup:
        print("[bib] 重复 key：")
        for k, v in dup.items():
            print(f"  {k}  出现于 {', '.join(v)}")
        problems.append("存在重复 key")

    if problems:
        print("[bib] 需要修复：")
        for p in problems:
            print(p)
    else:
        print("[bib] 必填字段检查通过")

    if todos:
        print(f"[bib] 待核对条目（含 TODO，投稿前务必补齐出版信息）：{', '.join(sorted(set(todos)))}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
