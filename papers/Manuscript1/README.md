# 论文草稿

## 目录组织

目标是 **Elsevier 旗下偏进化计算、智能优化、无人系统方向的期刊**，而且正文希望压缩成 **5 个主要章节**；

**1. Introduction**

**2. Problem Formulation and DMDE**

**3. Proposed LLM-Enhanced DMDE**

- 3.1 Overall Framework
- 3.2 LLM-Guided Population Initialization
- 3.3 LLM-Guided Adaptive Crossover Rate Control

**4. Experimental Results and Discussion**

- 4.1 Experimental Setup
- 4.2 Comparison with Existing Methods
- 4.3 Ablation Study
- 4.4 Scalability Analysis

**5. Conclusion**

---

## LaTeX 工程结构

```
Manuscript1/
├── main.tex                 # 唯一入口：导言区装配 + 章节顺序 + 参考文献
├── Makefile                 # 一键构建 / 清理 / 自检
├── latexmkrc                # 产物统一输出到 build/，源码目录保持干净
├── README.md                # 本文件
│
├── config/                  # 配置层（只管「怎么排」，不含正文）
│   ├── packages.tex         #   所有 \usepackage 集中在此
│   ├── commands.tex         #   自定义命令与数学符号（\CR、\todo 等）
│   └── metadata.tex         #   标题 / 作者 / 单位 / 通讯作者
│
├── sections/                # 正文层：一个文件一个章节
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_problem_formulation.tex
│   ├── 03_proposed_method.tex
│   ├── 04_experiments.tex
│   ├── 05_conclusion.tex
│   └── 06_appendix.tex      # 默认关闭，需在 main.tex 取消注释
│
├── frontmatter/             # Elsevier 投稿必需件（不占正文章节号）
│   ├── highlights.tex               # 3--5 条，每条 ≤85 字符
│   ├── declaration_of_interest.tex
│   ├── credit_author_statement.tex
│   └── data_availability.tex
│
├── bib/references.bib       # 参考文献库
├── figures/                 # 插图（\graphicspath 已指向此处）
├── tables/                  # 表格片段（一个文件一张表）
├── tools/                   # 辅助脚本（同步实验产出、bib 体检、highlights 检查）
└── build/                   # 编译产物（已 gitignore）
```

**分层原则**：`config/` 决定版式，`sections/` 只写内容，`bib`+`figures`+`tables`
只放素材。改版式不动正文，改正文不动版式。

## 构建

需要 TeX Live；`make` 会自动跑多轮 pdflatex 与 BibTeX。

```bash
make                # 生成 build/main.pdf
make watch          # 持续编译（写作时用）
make clean          # 清中间产物，保留 PDF
make distclean      # 删除 build/
make check          # 未定义引用 / 重复 bib key / 正文混入中文自检
make highlights     # highlights 条数与长度检查
make sync           # 从 experiments/ 同步表格与图片
```

### 依赖的 TeX 宏包

| 宏包 | 用途 | Debian/Ubuntu 安装 |
| --- | --- | --- |
| `elsarticle` | Elsevier 文档类与参考文献样式 | `texlive-publishers` |
| `booktabs`, `subcaption`, `caption` | 三线表、子图 | `texlive-latex-extra` |
| `algorithm`, `algpseudocode` | 算法伪代码 | `texlive-science` |
| `siunitx` | 单位排版（`\SI{5000}{m}`） | `texlive-science` |

一行装齐：

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra \
                 texlive-science texlive-publishers latexmk
```

## 写作约定

1. **一章一文件**：正文只写在 `sections/` 下；`main.tex` 里只出现 `\input`，不写正文。
2. **标签前缀**：图 `fig:`、表 `tab:`、公式 `eq:`、章节 `sec:`；引用统一走
   `\figref{} / \tabref{} / \equref{} / \secref{}`，便于全文改格式。
3. **中文只进注释**：正文用英文（Elsevier 投稿要求），`pdflatex` 无法渲染中文，
   因此中文只能出现在 `%` 注释里，**不要**写进 `\todo{}` 等会被渲染的宏。
4. **待办可见化**：用 `\todo{...}` 标记未完成处，编译后显示为红色。
   投稿前一键隐藏：把 `config/commands.tex` 里 `\reviewmode` 改为 `0`。
5. **一张表一个文件**：`tables/tbl_*.tex` 只含 `table` 环境片段，详见 `tables/README.md`。
6. **不手工敲文献**：条目从 DBLP / 期刊官网导出 BibTeX，避免卷期页码错漏；
   `make check` 会提示重复 key 与含 TODO 的待核对条目。

## 与 experiments/ 的衔接

实验产出的表格是**中文独立文档**（`xelatex + ctex`），不能直接 `\input`。
用脚本抽取为片段：

```bash
make sync      # 导出到 tables/imported/（一表一文件，该目录不入库）
python3 tools/sync_from_experiments.py --copy-figures   # 同时把结果图拷进 figures/
```

同步后需人工把 caption / 表头翻译为英文并另存为 `tables/tbl_<用途>.tex`，
再在 `sections/04_experiments.tex` 中 `\input{tables/tbl_<用途>}`。
细节见 `tables/README.md` 末尾的清单。

