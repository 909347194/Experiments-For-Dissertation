# tables/ —— 论文用表

## 约定

- **一个文件一张表**：`tbl_<用途>.tex`，例如 `tbl_solution_quality.tex`、`tbl_time_breakdown.tex`
- 文件内容是**片段**：只有 `\begin{table}...\end{table}`，**不要** `\documentclass`、`\begin{document}`、`\usepackage`
- 三线表统一用 `booktabs`（`\toprule / \midrule / \bottomrule`），禁止竖线与 `\hline`
- 引用：正文用 `\tabref{tab:xxx}`，标签统一 `tab:` 前缀

## 使用方式

```latex
\input{tables/tbl_solution_quality}
```

## 从实验同步（重要）

`experiments/exp_ablation_study/analyze/` 生成的
`analysis/figures/ablation_tables.tex` 是**独立的中文文档**（`xelatex + ctex`），
**不能**直接 `\input` 进本工程，原因有三个：

1. 它自带 `\documentclass` 与 `\begin{document}`；
2. 表格 caption 与表头是中文，`pdflatex` 无法渲染；
3. `ctex` 宏包与 `elsarticle` 冲突。

因此同步由脚本完成，它会抽出所有 `table` 环境写成片段：

```bash
make sync
# 或
python3 tools/sync_from_experiments.py --copy-figures
```

产物：`tables/imported/tbl_*.tex`（一表一文件，文件名取自表格标签）。

**同步后必须做的一件事**：把 caption、表头、行标签翻译成英文
（源文件里是中文，`pdflatex` 编译会报 `Unicode character not set up`），
然后另存为 `tables/tbl_<用途>.tex`，再在正文中 `\input{tables/tbl_<用途>}`。

## 表格清单（与实验产出的对应关系）

`make sync` 会把实验表格导出到 `tables/imported/`，一表一文件，文件名取自表格标签：

| 实验产出 | 内容 | 建议放置章节 |
| --- | --- | --- |
| `tbl_ablation_results.tex` | 解质量对比（Best / Mean ± Std / Median） | 4.3 Ablation Study |
| `tbl_time_breakdown.tex` | 计算时间分口径对比 | 4.1 Setup 或 4.3 |
| `tbl_mannwhitney.tex` | Mann-Whitney U 显著性检验 | 4.3 |
| `tbl_synergy.tex` | 协同效应分析（A3 增益 vs A1+A2） | 4.3 |
| `tbl_convergence_speed.tex` | 达到 X% 改进所需代数 | 4.3 |
| `tbl_initial_pop.tex` | 初始种群质量对比（PopInit 效果） | 4.3 |

`tables/imported/` 是**生成目录，不入库**（已 gitignore），可随时重新生成；
正式表格请翻译后另存为 `tables/tbl_<用途>.tex` 纳入版本控制。
