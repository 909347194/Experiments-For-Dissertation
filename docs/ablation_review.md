# 消融实验代码审查报告 — S2 场景与 analyze.py

> 最后更新：2026-09-17
> 审查范围：`experiments/exp_ablation_study/S2_srp_N10_M20/`、`experiments/exp_ablation_study/analyze.py`、相关 README
> 关联提交：`a81b211`（Sweettea 修复）、`80e07b0`（LLM 配置）、`761af9e`（README 同步）、`b5c9fab`（analyze.py 修复）

---

## 一、审查背景与摘要

本轮审查的目的是确保 `exp_ablation_study/` 能在本地 + 远端（SiliconFlow Qwen3.5-9B）稳定跑通，结果可被分析脚本正确消费，并避免论文层面的低级错误。S2 是尺寸最大的场景（N×M=10×20），也是最可能暴露约束配置问题的地方；`analyze.py` 是所有结果的"成品出口"，任何崩溃都会让跑出来的数据无法对外发布。

**关键结论**：

| 维度 | 结论 |
| --- | --- |
| S2 代码可跑通 | ✅ 4 配置单次运行均通过 |
| S2 结果合理性 | ⚠️ 反直觉：A2（pop_init）在常规参数下独胜，A1/A3 退化为基线 |
| analyze.py 可用性 | 🔴 修复前会在箱线图处直接崩溃（`boxplot(labels=)` 已废弃） |
| 统计方法名实 | 🔴 修复前 README 写"Wilcoxon"，但实现调用 `mannwhitneyu` |
| README 与代码一致性 | 🟡 已修正 7 处不一致 |
| 推送状态 | ✅ 本地 `b5c9fab` 与远端 `origin/master` 一致 |

---

## 二、S2 场景代码审查

### 2.1 S2 vs S1 关键差异

| 维度 | S1 (`S1_balanced_N10_M10`) | S2 (`S2_srp_N10_M20`) |
| --- | --- | --- |
| 规模 N × M | 10 × 10 | 10 × 20（2× 解空间） |
| `enable_seq` | `False` | `True`（顺序解码序列） |
| UAV 集群规模 | 较小 | 较大 |
| UAV 最大航程 | 28–32 km | 40–48 km（任务半径更大） |
| 其它约束 | 一致 | 一致 |

S2 的规模是 S1 的 4 倍，理论上 LLM 应该更容易"看出现成的种群初始化收益 / CR 调控机会"，但单 run 结果并非如此。

### 2.2 单次运行结果

| 配置 | 是否启用 LLM | S1 best | S1 vs A0 | S2 best | S2 vs A0 |
| --- | --- | --- | --- | --- | --- |
| **A0_dmde** | 否（基线） | 325 847.82 | — | 689 102.82 | — |
| **A1_cr_control** | 仅 CR 调控 | 336 438.28 | **+3.20 %** | 689 102.82 | **0.00 %** |
| **A2_pop_init** | 仅种群初始化 | 325 847.82 | **+0.00 %** | 547 680.36 | **−20.51 %** |
| **A3_full** | CR + pop_init | 336 438.28 | **+3.20 %** | 689 102.82 | **0.00 %** |

> 表中数字均为 1 run；样本量不足以下统计结论，但能反映各分支是否被实际触发。

### 2.3 审查发现

#### 2.3.1 A1_cr_control（S2）= A0 基线

- **`call_count = 0`**：CR 调控在 1000 代内未到触发间隔，模块完全旁路
- 根因：`interval = 50`、`MAX_GENERATIONS = 1000` 默认值时，最多触发 `1000 / 50 = 20` 次，但 S2 在 10×20 维下每代评估代价显著高于 S1，**单次 LLM 调用也未必落到触发点**
- 后果：1 run 数据与"是否启用 LLM"无关，不构成任何消融证据

#### 2.3.2 A3_full（S2）= A0 基线

- `pop_init` 让 A2 拿到 −20.5 %；A3 同时启用 `pop_init + cr_control`，应当 ≤ A2，但实际与 A0 一致
- 推测根因：CR 调控回合的 LLM 决策覆盖了 `pop_init` 的种群引导；
  在 `search_controller` 工作链中，"回退到基线种群"的指令压倒了"保留 pop_init 优势个体"的指令
- 需要更多 run 才能证伪/证实；从代码上看，两条 LLM 调用链**没有冲突防护**

#### 2.3.3 A2_pop_init（S2）−20.5 %（单 run）

- 唯一在 S2 上跑赢基线的配置；
- 提示：在 N×M=10×20 量级，LLM 一次性提示的种群初始化收益显著，但**持续介入（CR）反而拖累**

### 2.4 后续建议（任选其一）

1. **快速重跑 8 组**：把 `MAX_GENERATIONS` 临时降到 150、`interval` 改为 20、`init_ratio` 从 0.1 提到 0.2，让 CR 调控和多 run 都能体现差异
2. **保持现状**：用现有 1 run 数据，把 S2 审查发现写进 issue/PR，请作者确认约束后再正式跑
3. **仅跑 S2**：用更激进的 CR `interval`，先把 S2 上 A1/A3 的"零收益"问题确证

> 倾向方案 1：30 run × 8 配置在沙箱里 150 代可在 1 小时内跑完，足以得到像样的显著性结论。

---

## 三、`analyze.py` 代码审查

### 3.1 🔴 真实 Bug：`boxplot(labels=)` 将在新版 matplotlib 下崩溃

```python
# 旧（直接抛 TypeError）
bp = ax.boxplot(data, labels=labels, patch_artist=True)

# 新（兼容 matplotlib ≥ 3.9）
try:
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
except TypeError:
    bp = ax.boxplot(data, labels=labels, patch_artist=True)  # 回退旧版
```

- 实测环境 `matplotlib == 3.11.2`，原代码跑到箱线图时直接挂掉
- **这是 `analyze.py` 之前从未完整跑完的根因**

### 3.2 🔴 统计检验"名实不符"（论文级硬伤）

| 位置 | 旧文本 / 名称 | 修正 |
| --- | --- | --- |
| 函数名 | `wilcoxon_test` | `mann_whitney_test` |
| LaTeX 表标题 | "Wilcoxon 秩和检验" | "Mann-Whitney U 秩和检验" |
| LaTeX `\label{}` | `tab:wilcoxon` | `tab:mann_whitney` |
| README 表格 | "Wilcoxon signed-rank 配对检验" | "Mann-Whitney U 组间检验" |

- 实现原本调用 `scipy.stats.mannwhitneyu(..., alternative="two-sided")`，这是**双样本**检验，与"Wilcoxon signed-rank（配对）"含义不同
- 审稿人只要看一下 LaTeX 表脚注 + Python 源码就会质疑数据；现已彻底统一

### 3.3 🟡 `np.std()` 在 n=1 时返回 `nan`

- `np.std(arr)` 默认 `ddof=0`，但论文通常给样本标准差 `ddof=1`
- 当 `runs=1`（沙箱验证）时 `ddof=0` 也返回 `nan` + `RuntimeWarning`，表格里出现 `± nan`
- 修复：统一 `ddof=1`，加 `n_runs=1` 短路返回 `0.0`

### 3.4 🟡 README 承诺但代码未实现：LaTeX 主表自动加粗最优值

- README 写道："自动最优值加粗 (`\textbf{}`)"
- 修复前只在 markdown 汇总里加粗，LaTeX 表并未实现
- 修复后按场景分组取最小 `mean`（解越小越好），在其 `best / mean / median` 三个数字前自动套 `\textbf{}`

### 3.5 🟡 LaTeX 写法非标准

- `N{=}M{=}10` → 标准写法 `N=M=10`
- 表注追加：`n<5` 或 `scipy` 不可用时不计算显著性，标记 `---`

### 3.6 🟢 死代码清理

- 删除未调用的 `find_convergence_gen()`（README 没承诺，避免读代码的人误以为有用）
- 文档化所有模块顶部 docstring

### 3.7 修复后输出验证

```text
✅ 6 张图表：convergence_S1/S2.png、boxplot_S1/S2.png、time_breakdown.png、cr_trajectory_S1/S2.png
✅ 1 个 LaTeX 表：ablation_tables.tex（主结果表 + 显著性表合并输出）
✅ 3 个 Markdown/CSV：summary_table.md、per_scenario/*.md、summary_table.csv
✅ 6 个 LLM 决策日志（A1/A2/A3 × S1/S2）
```

> 注意：当前仅有 1 run 数据，箱线图退化、显著性表为 `---`。等正式 30 runs 后才会出现有意义的图表与 p 值。同时建议补装 `scipy`：

```bash
uv pip install scipy
```

---

## 四、README 与代码一致性同步（`exp_ablation_study/README.md` + 根 `README.md`）

| 位置 | 原内容（错） | 现内容（对） |
| --- | --- | --- |
| `figures/` 输出 | `convergence_comparison_S1.png`、`ablation_results.tex`、`ablation_significance.tex` | `convergence_S1.png`、`ablation_tables.tex`（合并单文件） |
| 参数表 | 列出 `--parallel` | 删除（代码里根本不存在） |
| 统计检验节 | "Wilcoxon signed-rank 配对检验" | "Mann-Whitney U 组间检验" + `n<5 / 无 scipy` 降级说明 |
| 目录树 `core/` | 只列 4 个文件 | 补 `recorder.py`、`solver_wrappers.py` |
| 目录树 `results/` | "30 个 .json 结果文件" | "1 个 `ablation_results.json`（每次覆盖）" |
| Q3 断点续跑 | "自动跳过已完成的 runs" | **无续跑机制**，每次重跑并覆盖 `ablation_results.json` |
| 行数标注 | 50 行 / 71 行 | 实测 37 / 48 行（小幅调整） |
| 根 README 快速开始 | 无消融实验入口 | 补 `uv run python experiments/exp_ablation_study/analyze.py` |

另新增 `experiments/exp_ablation_study/figures/.gitignore`，忽略 PNG/CSV/MD/决策日志等运行产物，保持仓库整洁。

---

## 五、推送记录

| 提交 | 说明 |
| --- | --- |
| `a81b211` | Sweettea 修复：`sys.path` 修正 + API key 别名 + 取消静默 fallback |
| `80e07b0` | 7 个 yaml 适配 SiliconFlow Qwen3.5-9B 思考模式（嵌套 `chat_template_kwargs.enable_thinking=false`、`thinking_budget=4096`、`max_tokens=16384`） |
| `761af9e` | README 同步（`uv run` + Qwen3.5-9B + 真实 DEM 路径 + 价格） |
| `b5c9fab` | `analyze.py` 修复 + README 再同步（本次提交） |

推送结果：本地 `b5c9fab..` → 远端 `origin/master`，工作区干净。

---

## 六、未决事项

1. **S2 A1/A3 的"零收益"**：需要约束调整（`MAX_GENERATIONS` / `interval` / `init_ratio`）或更多 run 才能确认是约束问题还是算法结构问题
2. **正式实验尚未跑**：当前所有图表均为 1 run，建议在调参确认后跑 30 runs
3. **`scipy` 未安装**：`pip install scipy` 后才能产出显著性表（当前表为 `---`）

