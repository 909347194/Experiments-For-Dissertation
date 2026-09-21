# 消融实验 (Ablation Study)

## 实验目的

验证 LLM-DMDE 单一核心模块的独立贡献：

- **Search Controller**：LLM-Guided Adaptive Search Control（在进化过程中依据观测到的进化状态自适应调整 CR / F / GMR）

原设计中的 LLM 增强种群初始化（PopInit）已从本项目中移除，理由见
`docs/prompt_design_search_controller.md`。

## 消融配置

| 编号 | 配置名 | Search Controller | 目的 |
|------|--------|-------------------|------|
| **A0** | Vanilla DMDE | ✗ | 基线（纯传统 DMDE，公式 3-9/3-11/3-12 耦合） |
| **A1** | DMDE + Search Controller | ✓ | 验证解耦自适应控制模块的贡献 |

### 扩展对照条件（动作空间控制）

除主消融外，另设两组动作空间对照，用于隔离"LLM 能否控制某个通道"这一变量。
它们位于场景目录下的独立子目录，由各自的 `run.py` 单独运行：

| 子目录 | 动作空间 | CR 取值 | 目的 |
|--------|----------|---------|------|
| `A1_cr_control` | CR + F + GMR（解耦） | LLM 每 50 代决策 | 主对照：完全解耦 |
| `A1_coupled` | 仅 CR（F/GMR 由公式跟随） | LLM 每 50 代决策 | 复现原 DMDE 耦合结构 |
| `A1_nocr` | F + GMR（CR 锁定） | 固定 0.3 | 移除 CR 通道后的残余效应 |

三者共用同一套观测特征接口，但**每组只向 LLM 暴露其可控制的通道**，
避免模型"看到却动不了"造成的混淆。

## 实验场景

选择原则：一个最简单场景（排除复杂度干扰）+ 一个最复杂场景（验证高难度下有效性）

| 场景 | Model | N (UAVs) | M (Targets) | 选择理由 |
|------|-------|----------|-------------|----------|
| **S1** | balanced (N=M) | 10 | 10 | 小规模基准（一对一匹配） |
| **S2** | srp (N<M) | 15 | 30 | 需同时决定分配 + 巡回顺序（高复杂度） |

> ⚠️ 目录名 `S2_srp_N10_M20` 是历史遗留：场景已由 `55ef7fa` 增强至 N=15, M=30，
> 实际规模以 `scenarios/s2_srp.py` 中的 `S2_CONFIG` 为准。

**总计**: 2 配置 × 2 场景 × 30 次独立运行 = **120 组实验**（扩展对照另计）

## 目录结构

```
experiments/exp_ablation_study/
├── README.md                  # 本文档
├── run_ablation.py            # 统一入口脚本（A0 / A1）
├── analyze.py                 # 统计分析 + 可视化 + LaTeX 表格生成
│
├── core/                      # 核心逻辑包（各文件职责单一）
│   ├── config.py              # 实验配置数据类
│   ├── scenario.py            # 场景构建 + DEM 加载
│   ├── runner.py              # A0 / A1 单次运行封装
│   ├── recorder.py            # 过程记录器（每代 fitness/CR/F + LLM 决策）
│   ├── solver_wrappers.py     # 求解器包装器（无侵入式过程记录）
│   └── results.py             # 结果保存/加载 + YAML 解析
│
├── analyze/                   # 分析包（python -m analyze）
│   ├── constants.py           # 场景/配置/标签/颜色常量
│   ├── stats.py               # 统计与检验
│   ├── tables.py              # Markdown + LaTeX 表格生成
│   ├── plots.py               # 收敛/CR/F/GMR/时间分解图
│   └── main.py                # 分析主入口
│
├── scenarios/                 # 场景定义包
│   ├── s1_balanced.py         # S1: balanced N=M=10
│   └── s2_srp.py              # S2: srp N=10,M=20
│
├── S1_balanced_N10_M10/       # 场景 1 工作目录
│   ├── shared_config.py       # 薄包装层 → scenarios.s1_balanced
│   ├── data/chengguan_district_dem.tif  # DEM 地形数据
│   ├── A0_dmde/               # Vanilla DMDE
│   │   ├── run.py             # 无 LLM 配置
│   │   ├── config/
│   │   └── results/ablation_results.json   # 汇总结果（内含各 run 的数组）
│   ├── A1_cr_control/         # Search Controller（解耦，主对照）
│   │   ├── run.py             # 自动读取 llm_config.yaml
│   │   ├── config/llm_config.yaml
│   │   └── results/ablation_results.json
│   ├── A1_coupled/            # 仅 CR 通道
│   ├── A1_nocr/               # F + GMR，CR 锁定 0.3
│   └── exp_cr_response/       # 离线 CR 响应面扫描
│
├── S2_srp_N10_M20/            # 场景 2 工作目录（同 S1 结构）
│   └── ...
│
└── figures/                   # 分析输出目录（运行时生成）
    ├── convergence_S1.png / convergence_S2.png       # 收敛曲线对比
    ├── cr_trajectory_S1.png / cr_trajectory_S2.png   # CR 变化轨迹
    ├── boxplot_S1.png / boxplot_S2.png               # 解质量箱线图
    ├── time_breakdown.png                            # 时间开销堆叠柱状图
    ├── ablation_tables.tex   # SCI 标准三线表（解质量 + 时间 + 显著性）
    ├── summary_table.md / summary_table.csv          # 汇总表
    └── llm_decisions_*.json  # LLM 决策日志
```

## 快速开始

### 前置准备

```bash
# 1. 确保已配置 LLM API（推荐 SiliconFlow + Qwen3.5-9B，详见 config/llm_config.yaml）
# 检查 .env 文件中是否有：GuiJiLiuDongAIYunFuWu_API_KEY=sk-xxx（或 SILICONFLOW_API_KEY=sk-xxx）

# 2. 确保 DEM 数据存在（S1/S2 各一份，文件名均为 chengguan_district_dem.tif）
ls -la S1_balanced_N10_M10/data/chengguan_district_dem.tif S2_srp_N10_M20/data/chengguan_district_dem.tif
```

### 运行实验

```bash
# 【推荐】运行全部 4 组实验（每组 30 runs）
uv run python run_ablation.py --runs 30

# 只运行场景 S1（2 组 × 30 runs）
uv run python run_ablation.py --scenario S1 --runs 30

# 只运行 A1 配置（Search Controller）
uv run python run_ablation.py --config A1 --runs 30

# 单独运行某一组（调试用）
cd S1_balanced_N10_M10/A1_cr_control && uv run python run.py --runs 5

# 运行扩展动作空间对照
cd S2_srp_N10_M20/A1_coupled && uv run python run.py --runs 3
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--runs` | 30 | 独立运行次数（建议≥30 保证统计显著性） |
| `--scenario` | all | 选择场景：S1, S2, 或 all |
| `--config` | all | 选择配置：A0, A1, 或 all |

> 说明：早期版本曾计划 `--parallel` 并行参数，当前 `run_ablation.py` **未实现**该参数，各组按顺序串行执行。

## 分析与可视化

依赖：`numpy`（必需）；`matplotlib`（可选，缺省则跳过图表）；`scipy`（可选，缺省则跳过统计检验，显著性表显示 `---`）。通常随实验环境一并安装。

```bash
# 生成所有图表和 LaTeX 表格
uv run python -m analyze

# 输出文件（figures/ 目录，运行时生成）：
# - figures/convergence_S1.png, convergence_S2.png    # 收敛曲线对比
# - figures/cr_trajectory_S1.png, cr_trajectory_S2.png # CR 变化轨迹
# - figures/f_trajectory_S1.png, f_trajectory_S2.png   # F 值轨迹
# - figures/gmr_distribution_S1.png, ...               # GMR 模式分布
# - figures/boxplot_S1.png, boxplot_S2.png            # 解质量箱线图
# - figures/time_breakdown.png                        # 时间开销堆叠柱状图
# - figures/ablation_tables.tex                       # 解质量/时间/显著性 + 收敛速度表
# - figures/summary_table.md, summary_table.csv       # 汇总表（Markdown / Excel）
# - figures/llm_decisions_<场景>_A1_seed<种子>.json   # LLM 决策日志
```

### LaTeX 表格使用说明

`figures/ablation_tables.tex` 是一份**完整的可编译文档**（含 `\documentclass` 与
解质量表、时间表、显著性表、收敛速度表）：

```latex
\usepackage{booktabs}  % 若只把表格片段拷进自己的论文，请在导言区添加

% 方式一：直接编译整份文档（中文需 xelatex）
%   xelatex figures/ablation_tables.tex

% 方式二：只取你需要的表
%   打开 ablation_tables.tex，复制对应 \begin{table}...\end{table} 片段即可
```

**表格特性**：
- ✅ 自动最优值加粗（`\textbf{}`，在场景分组内取最优）
- ✅ 显著性标记：`*` (p<0.05)、`**` (p<0.01)、`n.s.` 不显著、`---` 未检验
- ✅ 专业三线表风格（无竖线）
- ✅ 表注说明统计方法与阈值

## 评估指标

### 解质量指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **Best Fitness** | 最优解质量 | 30 runs 中的最小值 |
| **Mean ± Std** | 平均解质量 ± 离散度 | 均值和标准差 |
| **Median** | 中位数（鲁棒性） | 50% 分位值 |

### 收敛性能指标

| 指标 | 含义 | 计算方式 |
|------|--------|----------|
| **Conv. Gen** | 收敛速度 | 达到 95% 最终最优解所需代数 |
| **Success Rate** | 成功率 | 适应度低于阈值的运行比例 |

### 时间开销指标（分口径）

| 指标 | 包含内容 | 用途 |
|------|----------|------|
| **Total Time(s)** | 全部耗时 | 实际总耗时 |
| **DMDE Time(s)** | 仅进化计算 | 算法本身开销 |
| **LLM Time(s)** | Search Controller 全部调用 | 进化阶段 LLM 代价 |
| **LLM Calls** | 总调用次数 | 评估 API 成本 |
| **Avg Latency(ms)** | 平均延迟 | LLM 响应速度 |

### 参数解耦指标

| 指标 | 含义 | 用途 |
|------|------|------|
| **CR-F 相关系数** | CR 与 F 的样本相关性 | 验证解耦成功（应≈0，而 A0 应≈±1） |
| **F 覆写次数** | LLM 直接指定 F 的次数 | 衡量 F 通道被实际使用 |
| **GMR 模式分布** | auto / on / off 占比 | 衡量灭绝机制通道的使用 |

### 统计检验

| 检验 | 用途 | 显著性阈值 |
|------|------|------------|
| **Mann-Whitney U 秩和检验** | A1 vs A0 组间（非配对）检验 | p < 0.05 (*) / p < 0.01 (**) |

> 实现说明：`analyze/stats.py` 使用 `scipy.stats.mannwhitneyu`（双侧）。样本量 `n < 5`
> 或未安装 `scipy` 时跳过检验，表格对应单元格显示 `---`。

## 预期结论与验证点

| 对比 | 预期结果 | 验证点 | 理论依据 |
|------|----------|--------|----------|
| **A1 vs A0** | A1 更优 (p<0.05) | 自适应控制 > 固定公式 | 动态参数调整适应搜索状态 |
| **A1 CR-F 相关性** | ≈0 | 三通道解耦成功 | 公式 3-11 的 F(CR) 依赖被打破 |
| **A1_coupled vs A1** | 解耦 ≥ 耦合 | 解耦带来额外自由度 | CR 不再是唯一杠杆 |
| **A1_nocr vs A0** | 差异小 | CR 是主要杠杆 | F/GMR 通道残余效应有限 |

> ⚠️ 实测提示：在当前静态 UAV 分配任务上，离线固定 CR=0.3 在 9/9 组对照中优于
> 所有在线配置。因此本实验的定位是**机制研究与负面结果报告**，而非"LLM 一定更快"。
> 详见 `docs/prompt_design_search_controller.md`。

## 常见问题 (FAQ)

### Q1: 如何切换 LLM 模型？
修改 `S*/A1_cr_control/config/llm_config.yaml` 中的 `model` 字段：
```yaml
providers:
  siliconflow:
    model: Qwen/Qwen3.5-9B  # 或 Qwen/Qwen3.6-35B-A3B（如需更强推理）
```

### Q2: 如何更换场景（如 N=20, M=30）？
1. 在 `scenarios/` 目录创建新场景文件（参考 `s1_balanced.py`）
2. 复制 `S1_balanced_N10_M10/` 为新目录
3. 修改 `shared_config.py` 导入新场景
4. 运行 `python run_ablation.py --scenario S3`

### Q3: 实验中断后如何恢复？
目前 `run.py` 每次运行都会**从头重跑全部 runs，并覆盖** `results/ablation_results.json`
（尚无断点续跑机制）。中断后需重新运行；如需控制单次耗时，可用 `--runs` 指定较少次数：
```bash
uv run python run_ablation.py --runs 5   # 只跑前 5 个种子
```

### Q4: 如何自定义随机种子？
默认使用 `42, 43, ..., 71`（30 个连续种子）。可在 `core/config.py` 中修改：
```python
RANDOM_SEEDS = list(range(42, 72))  # 或自定义列表
```

### Q5: LLM API 成本估算？
以 SiliconFlow + Qwen3.5-9B 为例（输入/输出 $0.10/$0.15 per M tokens）：
- 单次 Search Controller 决策: ~2000 tokens × $0.0001 = $0.0002
- 每 run 决策次数 ≈ MAX_GENERATIONS / interval = 500 / 50 = 10 次
- 120 组实验总成本：约 **$0.25 - $0.5**

### Q6: 结果文件为什么不在仓库里？
`**/results/*.json` 已加入 `.gitignore`（运行时生成产物，体积大且可重跑）。
需要数据时本地运行实验即可；历史提交中的结果已由
`chore: 清除所有实验结果` 提交从跟踪中移除。

## 参考文献

[1] Wilcoxon F. Individual comparisons by ranking methods[J]. Biometrics Bulletin, 1945.
[2] 论文方法论章节（LLM-DMDE 框架说明）
