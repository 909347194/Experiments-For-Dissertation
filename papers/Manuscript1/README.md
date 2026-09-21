# 论文草稿 — Manuscript1

> 目标期刊：Elsevier 旗下偏进化计算 / 智能优化 / 无人系统方向的期刊。
> 正文目标章节数：**5**（外加可选 appendix）。
> 编译引擎：**pdflatex**（论文为英文）。

---

## 1. 5 分钟快速开始

```bash
cd papers/Manuscript1
sudo apt install texlive-latex-recommended texlive-latex-extra \
                 texlive-science texlive-publishers latexmk   # 一次性
make                # 生成 build/main.pdf（首次约 30 s，连续编译 < 5 s）
```

打开 `build/main.pdf` 验收。能跑通这套命令就算接入成功，可以跳到 §7 写正文。

---

## 2. 章节规划（与 README 同名旧文档保留）

| 章节 | 标题 | 文件 |
| --- | --- | --- |
| — | Abstract | `sections/00_abstract.tex` |
| 1 | Introduction | `sections/01_introduction.tex` |
| 2 | Problem Formulation and DMDE | `sections/02_problem_formulation.tex` |
| 3 | Proposed LLM-Enhanced DMDE | `sections/03_proposed_method.tex` |
|   | 3.1 Overall Framework | （同文件 `\subsection`） |
|   | 3.2 Action-Space Control Conditions | （同文件 `\subsection`） |
|   | 3.3 LLM-Guided Search Control | （同文件 `\subsection`） |
| 4 | Experimental Results and Discussion | `sections/04_experiments.tex` |
|   | 4.1 Experimental Setup | |
|   | 4.2 Comparison with Existing Methods | |
|   | 4.3 Ablation Study | |
|   | 4.4 Scalability Analysis | |
| 5 | Conclusion | `sections/05_conclusion.tex` |
| A | Appendix（可选，默认关闭） | `sections/06_appendix.tex` |

---

## 3. 依赖

### 3.1 系统宏包（Debian/Ubuntu）

| 用途 | 宏包 | apt 包 |
| --- | --- | --- |
| Elsevier 文档类 | `elsarticle.cls` | `texlive-publishers` |
| 三线表、子图、caption | `booktabs` `subcaption` `caption` | `texlive-latex-extra` |
| 算法伪代码 | `algorithm` `algorithmicx` `algpseudocode` | `texlive-science` |
| 单位排版 | `siunitx` | `texlive-science` |
| BibTeX 自动构建 | — | `latexmk` |

一行装齐：

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra \
                 texlive-science texlive-publishers latexmk
```

> **不要用 `texlive-full`**：4 GB+ 且与本工程无关。
> TeX Live 2023 起均验证通过；公司 2020 缺 `subcaption` 的新版，需额外补 `tlmgr install subcaption`。

### 3.2 Python 工具

`make check` / `make sync` 依赖 `python3`，无第三方依赖（只用标准库 `re/pathlib/argparse`）。

> **Makefile 会自动挑选能用的 `python3`**：按顺序测试 `/usr/bin/python3` → `/usr/local/bin/python3` → `/opt/homebrew/bin/python3` → 系统 `PATH`，挑首个 `--version` 能跑通的。多版本或 pyenv shim 损坏（`.python-version` 指向未安装版本）时也能直接用 `make check`。
> 显式覆盖：`make check PYTHON=/path/to/python3`。

---

## 4. 工程结构

```
Manuscript1/
├── main.tex              # 唯一入口：导言区 + 章节顺序 + 参考文献（无正文）
├── Makefile              # 一键构建（pdflatex → bibtex → pdflatex ×2）
├── latexmkrc             # 产物统一进 build/；扩展名清理
├── README.md             # 本文件
│
├── config/               # ★版式层★（"怎么排"，不含正文）
│   ├── packages.tex      #   所有宏包集中入口
│   ├── commands.tex      #   \CR \NP \todo \reviewmode 等自定义
│   └── metadata.tex      #   标题 / 作者 / 单位 / 通讯作者 / 摘要字数限制
│
├── sections/             # ★正文层★（"写什么"，一章一文件）
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_problem_formulation.tex
│   ├── 03_proposed_method.tex      # 含 3.1/3.2/3.3 三个 \subsection（解耦 + 动作空间 + 搜索控制）
│   ├── 04_experiments.tex          # 含 4.1/4.2/4.3/4.4
│   ├── 05_conclusion.tex
│   └── 06_appendix.tex             # 默认 \iffalse 包裹，需时打开
│
├── frontmatter/          # ★Elsevier 投稿必需件★（不占章节号）
│   ├── highlights.tex              #   3–5 条，每条 ≤ 85 字符
│   ├── declaration_of_interest.tex
│   ├── credit_author_statement.tex
│   └── data_availability.tex
│
├── bib/references.bib    # 参考文献库
├── figures/              # ★素材层★：插图（\graphicspath 已指向此）
├── tables/               # ★素材层★：表格片段（一文件一表）
│   ├── README.md                   #   命名约定与同步流程
│   └── imported/                   #   make sync 产物，已 gitignore
├── tools/                # 辅助脚本（check_bib / check_highlights / check_no_cjk / sync_from_experiments）
└── build/                # 编译产物（已 gitignore，不入库）
```

**分层约束**：
- `sections/` 只 `\input` 自己章节的内容与 `tables/tbl_*.tex` 的表片段；不写 `\usepackage` / `\renewcommand`。
- `config/` 不出现任何章节正文或表格片段。
- 新增素材一律走 `figures/` / `tables/`，**不要**把图粘到章节文件里。

---

## 5. 所有 make 命令

| 命令 | 作用 | 何时用 |
| --- | --- | --- |
| `make` | 完整构建，生成 `build/main.pdf` | 默认；CI 用 |
| `make watch` | 监听源文件变化，变化即重编 | 写作时挂在前台 |
| `make check` | 自检：未定义引用 / 重复 bib key / 正文混入中文 | 每次 commit 前 |
| `make highlights` | 检查 highlights 条数（3–5）与每条字数（≤85） | 投稿前 |
| `make sync` | 从 `experiments/` 抽取表格与图片到 `tables/imported/` | 新跑完实验后 |
| `make clean` | 清中间产物，保留 `build/main.pdf` | 普通清理 |
| `make distclean` | 删 `build/` 整个目录 | 调试 / 重新构建 |

---

## 6. 写作约定（务必先读完再动手）

1. **一章一文件**。`main.tex` 里只有 `\input`；正文一律写在 `sections/`。
2. **标签前缀固定**。图 `fig:`、表 `tab:`、公式 `eq:`、章节 `sec:`、算法 `alg:`。
   引用走 `\figref{}` / `\tabref{}` / `\equref{}` / `\secref{}` / `\algref{}`，
   不要直接写 "Figure 3"，便于全文格式调整。
3. **正文只用英文**。Elsevier 投稿要求英文正文，`pdflatex` 无法渲染中文。
   **中文只能出现在 `%` 注释里**——把中文塞进 `\todo{}` 会让编译直接失败
   `Unicode character ... not set up`。`tools/check_no_cjk.py` 已加入 `make check`。
4. **待办标记用 `\todo{...}`**，编译后显示为红色文本。投稿前把
   `config/commands.tex` 里 `\reviewmode` 从 `1` 改成 `0` 即一键隐藏。
5. **一张表一个文件**。`tables/tbl_*.tex` 只含 `table` 环境片段，
   在正文里 `\input{tables/tbl_xxx}` 引入。详见 `tables/README.md`。
6. **不手敲文献**。从 DBLP / 期刊官网导出 BibTeX，键名形如 `authorYEARkeyword`。
   `make check` 会检查重复 key 与含 TODO 的待核对条目。
7. **图片先转 PDF/PNG**。矢量图用 PDF，光栅图用 PNG/JPEG。`\includegraphics`
   不带扩展名即可，`\graphicspath` 已指向 `figures/`。

---

## 7. 典型工作流

### 7.1 写一段新正文

```bash
# 1. 编辑 sections/03_proposed_method.tex（已经有占位 TODO）
# 2. 边写边看
make watch
# 3. 自检
make check
```

如需新增章节：建 `sections/07_<name>.tex` → 在 `main.tex` 的 `\input` 列表里加一行。

### 7.2 加一张图

```bash
# 1. 把图片放进 figures/（推荐 PDF 矢量图）
cp /path/to/your/figure.pdf figures/fig_framework.pdf
# 2. 在正文里：
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{fig_framework}
  \caption{Overall framework of the proposed LLM-enhanced DMDE.}
  \label{fig:framework}
\end{figure}
# 3. 引用：\figref{fig:framework}
```

### 7.3 加一张表（最常见：从实验同步）

实验产出（`experiments/exp_ablation_study/figures/ablation_tables.tex`）是
中文独立文档（xelatex + ctex），**不能直接 `\input` 进本工程**。流程：

```bash
make sync                        # → tables/imported/tbl_<name>.tex（不入库）
# 把 caption、表头、标签翻译成英文，另存为：
cp tables/imported/tbl_xxx.tex   tables/tbl_xxx.tex
# 编辑 tables/tbl_xxx.tex 调整 caption、用 booktabs 三线表风格
# 在正文里：
\input{tables/tbl_xxx}
```

详见 `tables/README.md` 末尾的清单。

### 7.4 加一条文献

```bibtex
% 在 bib/references.bib 末尾追加（示例）
@article{storn1997differential,
  author  = {Storn, Rainer and Price, Kenneth},
  title   = {Differential evolution --- {A} simple and efficient heuristic
             for global optimization over continuous spaces},
  volume  = {11},
  number  = {4},
  pages   = {341--359},
  year    = {1997},
  journal = {Journal of Global Optimization},
  doi     = {10.1023/A:1008202821328}
}
```

正文里用 `\cite{storn1997differential}`。`make check` 会核对重复 key 与
"还含 TODO 的条目"，提醒补全信息。

---

## 8. 投稿前自检清单

```bash
make distclean && make           # 干净构建必须无 error
make check                       # 未定义引用 / 重复 key / 中文混排 = 0
make highlights                  # 3–5 条，每条 ≤85 字符
```

然后手动：

- [ ] `config/commands.tex` 里 `\reviewmode` 已置 0（关闭红色 TODO）
- [ ] `config/metadata.tex` 已填真实作者、单位、通讯邮箱、关键词
- [ ] `frontmatter/` 四份全部展开（高亮 / 利益声明 / 作者贡献 / 数据可用性）
- [ ] `bib/references.bib` 没有带 `TODO:` 的条目
- [ ] `figures/` 中所有图都被 `\input` / `\includegraphics` 引用过，无孤儿文件

---

## 9. 常见问题

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `LaTeX Error: File 'xxx.sty' not found.` | 缺宏包 | 见 §3.1 表格补 apt 包 |
| `Unicode character 补 (U+88DC) not set up.` | `\todo{}` 或正文中混了中文 | 中文挪到 `%` 注释；运行 `make check` |
| `Citation 'xxx' undefined.` | bib 没编进 | `make clean && make`（latexmk 会自动跑 bibtex） |
| `Label 'xxx' multiply defined.` | 两个标签同名 | `grep -rn 'label{xxx}' sections/` 定位后改前缀 |
| `Overfull \hbox ... 100pt.` | 长 URL/公式撑出边界 | 用 `\sloppy` / `\\` / `microtype`（已默认启用） |
| 中文表格同步过来编译失败 | 源是 `xelatex+ctex`，本工程是 `pdflatex` | 把表头/caption 翻译成英文再保存（§7.3） |
| `make` 后没有 `build/main.pdf` | `pdflatex` 中途失败；看 stdout 末行 `!` 报错 | 缺宏包或源文件语法错，按行号改 |

---

## 10. 与 experiments/ 的衔接（背景）

`experiments/exp_ablation_study/` 下的分析脚本会产出中文表格（`xelatex+ctex`
渲染）。这些表格不能被 `elsarticle`（`pdflatex`）直接 `\input`，所以用
`tools/sync_from_experiments.py` 抽出成片段：

```bash
make sync                                       # 仅表格
python3 tools/sync_from_experiments.py --copy-figures   # 同时拷图
```

抽出的片段落到 `tables/imported/`（**不入库**）。翻译 caption/表头/标签后，
另存为 `tables/tbl_<用途>.tex`，再在 `sections/04_experiments.tex` 中 `\input`。

完整的图—表—章节对应清单见 `tables/README.md` 末尾。