#===============================================================================
#  Manuscript1 —— LaTeX 构建
#
#  常用命令：
#     make            编译出 build/main.pdf（自动多轮 + BibTeX）
#     make watch      持续编译，改文件即自动重编（草稿写作时用）
#     make clean      删除中间产物，保留 PDF
#     make distclean  删除 build/ 整个目录
#     make check      检查未定义引用 / 重复 bib key / 中文是否误入正文
#     make highlights 检查 highlights 每条是否 ≤ 85 字符
#     make sync       从 experiments/ 同步生成的表格与图
#
#  注：GNU Make 4+ 支持 .RECIPEPREFIX，因此本文件不需要 Tab 缩进
#===============================================================================

.RECIPEPREFIX = >
.DEFAULT_GOAL := all

MAIN    := main
BUILD   := build
LATEXMK := latexmk
# 可用 PYTHON=/path/to/python3 make check 覆盖（多版本/虚拟环境下有用）
PYTHON  ?= python3

.PHONY: all pdf watch clean distclean check highlights sync

all: pdf

pdf:
> @$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)
> @echo ">>> 产物: $(BUILD)/$(MAIN).pdf"

watch:
> @$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(MAIN)

# 保留 PDF，只清中间文件
clean:
> @$(LATEXMK) -c $(MAIN)
> @echo ">>> 已清理中间产物（保留 PDF）"

distclean:
> @$(LATEXMK) -C $(MAIN)
> @rm -rf $(BUILD)
> @echo ">>> 已删除 $(BUILD)/"

# 常见问题自检：未定义引用、重复 key、pdflatex 不支持的中文
check:
> @echo "--- 未定义引用 / 标签 ---"
> @grep -nE "Warning.*(undefined|multiply defined)" $(BUILD)/$(MAIN).log || echo "  无"
> @echo "--- 重复 bib key / 必填字段 ---"
> @$(PYTHON) tools/check_bib.py
> @echo "--- 正文是否混入中文（中文只能出现在注释里）---"
> @$(PYTHON) tools/check_no_cjk.py

highlights:
> @$(PYTHON) tools/check_highlights.py frontmatter/highlights.tex

sync:
> @$(PYTHON) tools/sync_from_experiments.py
