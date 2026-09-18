# figures/ —— 论文用图

## 约定

- **文件名**：`fig_<用途>.pdf`，例如 `fig_framework.pdf`、`fig_convergence_S1.pdf`、`fig_cr_trajectory_S1.pdf`
- **格式**：优先 **PDF（矢量）**；散点/热力图等位图导出为 300 dpi 以上的 PNG
- **引用**：正文用 `\figref{fig:xxx}`，标签统一 `fig:` 前缀

## 使用方式

`config/packages.tex` 已设置 `\graphicspath{{figures/}}`，
所以正文里直接写文件名即可，不需要写目录：

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\linewidth]{fig_convergence_S1.pdf}
  \caption{...}
  \label{fig:convergence}
\end{figure}
```

## 从实验同步

实验生成的图（`experiments/exp_ablation_study/figures/*.png`）用下述命令取回：

```bash
make sync                          # 只导出表格片段
python3 tools/sync_from_experiments.py --copy-figures   # 同时拷贝图片
```

注意：同步过来的图片文件名形如 `convergence_S1.png`、`cr_trajectory_S1.png`，
建议改名为 `fig_*.pdf` 形式后再引用，保持本目录命名一致。

## 不要放在这里的

- 中间产物、截图、PPT 导出件（用临时目录）
- 期刊要求单独提交的 `graphical_abstract`（单独命名为 `graphical_abstract.*` 放在本目录也可，但不要写进 `main.tex`）
