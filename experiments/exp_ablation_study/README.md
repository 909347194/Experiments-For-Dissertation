# 消融实验 (Ablation Study)

## 实验目的

验证 LLM-DMDE 单一核心模块的独立贡献：

- **Search Controller**：LLM-Guided Adaptive Search Control（在进化过程中依据观测到的进化状态自适应调整 CR / F / GMR）

原设计中的 LLM 增强种群初始化（PopInit）已从本项目中移除，理由见
`docs/prompt_design_search_controller.md`。

## 消融配置

| 编号 | 配置名 | 子目录 | Search Controller | CR 取值 | 目的 |
|------|--------|--------|-------------------|---------|------|
| **A0** | Vanilla DMDE | `A0_dmde` | ✗ | 公式 3-9 动态 | 基线（纯传统 DMDE） |
| **A1** | DMDE + SC (Preset) | `A1_cr_control` | ✓ 解耦 | LLM 选择 preset | 主对照：完全解耦 |
| **A1c** | DMDE + SC (Coupled) | `A1_coupled` | ✓ 耦合 | LLM 仅控 CR | 复现原 DMDE 耦合结构 |
| **A1n** | DMDE + SC (No CR) | `A1_nocr` | ✓ CR 锁定 | 固定 0.3 | 移除 CR 通道后的残余效应 |

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

**总计**: 4 配置 × 2 场景 × 30 次独立运行 = **240 组实验**

## 目录结构

```
experiments/exp_ablation_study/
├── README.md                  # 本文档
├── run_ablation.py            # 统一入口脚本（A0 / A1 / A1c / A1n）
├── analyze.py                 # 统计分析 + 可视化 + LaTeX 表格生成
│
├── core/                      # 核心逻辑包（各文件职责单一）
│   ├── config.py              # 实验配置数据类
│   ├── scenario.py            # 场景构建 + DEM 加载
│   ├── runner.py              # A0 / A1 单次运行封装（支持 model 覆盖）
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
│   └── s2_srp.py              # S2: srp N=15,M=30
│
├── S1_balanced_N10_M10/       # 场景 1 工作目录
│   ├── shared_config.py       # 薄包装层 → scenarios.s1_balanced
│   ├── data/chengguan_district_dem.tif  # DEM 地形数据
│   ├── A0_dmde/               # Vanilla DMDE
│   ├── A1_cr_control/         # Search Controller（解耦，主对照）
│   ├── A1_coupled/            # 仅 CR 通道
│   ├── A1_nocr/               # F + GMR，CR 锁定 0.3
│   └── exp_cr_response/       # 离线 CR 响应面扫描
│
├── S2_srp_N10_M20/            # 场景 2 工作目录（同 S1 结构）
│   └── ...
│
└── figures/                   # 分析输出目录（运行时生成）
```

## 快速开始

### 前置准备

```bash
# 1. 确保已配置 LLM API
#    .env 文件中需要：SILICONFLOW_API_KEY=sk-xxx

# 2. 确保 DEM 数据存在
ls -la S1_balanced_N10_M10/data/chengguan_district_dem.tif
ls -la S2_srp_N10_M20/data/chengguan_district_dem.tif
```

### 运行实验

```bash
# 运行全部 4 配置 × 2 场景 = 8 组（每组 30 runs）
uv run python run_ablation.py --runs 30

# 只运行场景 S1
uv run python run_ablation.py --scenario S1 --runs 30

# 只运行 A1 配置（解耦 SC）
uv run python run_ablation.py --config A1 --runs 30

# 覆盖模型（命令行指定，不修改 YAML）
uv run python run_ablation.py --model Qwen/Qwen3.8-27B --runs 30

# 覆盖模型 + 回退模型
uv run python run_ablation.py \
    --model Qwen/Qwen3.8-27B \
    --fallback-model Qwen/Qwen3-14B \
    --runs 30

# 组合使用：指定场景 + 配置 + 模型
uv run python run_ablation.py \
    --scenario S2 --config A1 \
    --model Qwen/Qwen3.8-27B \
    --runs 30

# 单独运行某一组（调试用）
cd S1_balanced_N10_M10/A1_cr_control && uv run python run.py --runs 5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--runs` | 30 | 独立运行次数（建议≥30 保证统计显著性） |
| `--scenario` | all | 选择场景：`S1`, `S2`, 或 `all` |
| `--config` | all | 选择配置：`A0`, `A1`, `A1c`, `A1n`, 或 `all` |
| `--model` | (YAML) | 覆盖 `llm_config.yaml` 中的 `model`，通过环境变量传递给子进程 |
| `--fallback-model` | (YAML) | 覆盖 `fallback_model`（主模型失败时的回退） |

> `--model` 不修改原始 YAML 文件，而是生成临时副本覆盖。
> A0（Vanilla DMDE）无 LLM，`--model` 对其无效。

## 分析与可视化

依赖：`numpy`（必需）；`matplotlib`（可选，缺省则跳过图表）；`scipy`（可选，缺省则跳过统计检验）。

```bash
# 生成所有图表和 LaTeX 表格
uv run python -m analyze

# 输出文件（figures/ 目录）：
# - convergence_S1.png / convergence_S2.png       # 收敛曲线对比
# - cr_trajectory_S1.png / cr_trajectory_S2.png   # CR 变化轨迹
# - f_trajectory_S1.png / f_trajectory_S2.png     # F 值轨迹
# - gmr_distribution_S1.png / ...                 # GMR 模式分布
# - boxplot_S1.png / boxplot_S2.png               # 解质量箱线图
# - time_breakdown.png                            # 时间开销堆叠柱状图
# - preset_distribution_S1.png / ...              # LLM 档位选择分布
# - ablation_tables.tex                           # LaTeX 三线表
# - summary_table.md / summary_table.csv          # 汇总表
# - llm_decisions_<场景>_A1_seed<种子>.json       # LLM 决策日志
```

### LaTeX 表格

`figures/ablation_tables.tex` 是一份完整的可编译文档：

```latex
\usepackage{booktabs}  % 三线表

% 方式一：直接编译
%   xelatex figures/ablation_tables.tex

% 方式二：复制表格片段到自己的论文
```

**表格特性**：最优值加粗、显著性标记（`*` p<0.05 / `**` p<0.01 / `n.s.`）、三线表风格。

## 评估指标

| 类别 | 指标 | 含义 |
|------|------|------|
| 解质量 | Best Fitness / Mean±Std / Median | 最优、平均、中位数 |
| 收敛 | Conv. Gen (95%) | 达到 95% 最终最优解的代数 |
| 时间 | Total / DMDE / LLM Time(s) | 分口径耗时 |
| LLM | LLM Calls / Avg Latency / Preset 分布 | API 成本与决策行为 |
| 解耦 | CR-F 相关系数 / F 覆写次数 / GMR 模式分布 | 参数解耦验证 |
| 统计 | Mann-Whitney U (p<0.05 / p<0.01) | A1 vs A0 显著性 |

## 预期结论与验证点

| 对比 | 预期结果 | 验证点 |
|------|----------|--------|
| A1 vs A0 | A1 更优 (p<0.05) | 自适应控制 > 固定公式 |
| A1 CR-F 相关性 | ≈0 | 三通道解耦成功 |
| A1c vs A1 | 解耦 ≥ 耦合 | 解耦带来额外自由度 |
| A1n vs A0 | 差异小 | CR 是主要杠杆 |

> ⚠️ 实测提示：离线固定 CR=0.3 在多数对照中优于在线配置。
> 本实验定位是**机制研究与负面结果报告**，而非"LLM 一定更快"。

## 常见问题

### Q1: 如何切换 LLM 模型？

**方式一**：命令行覆盖（推荐，不改文件）
```bash
uv run python run_ablation.py --model Qwen/Qwen3.8-27B --runs 30
```

**方式二**：修改 YAML
```yaml
# config/llm_config.yaml
model: Qwen/Qwen3.8-27B
fallback_model: Qwen/Qwen3-14B
```

### Q2: 实验中断后如何恢复？

目前每次运行从头重跑，覆盖 `results/ablation_results.json`。用 `--runs` 控制规模：
```bash
uv run python run_ablation.py --runs 5
```

### Q3: 如何自定义随机种子？

修改 `shared_config.py` 中的 `SEEDS`：
```python
SEEDS = list(range(42, 72))  # 默认 30 个连续种子
```

### Q4: LLM API 成本估算？

以 SiliconFlow + Qwen3.8-27B 为例：
- 单次决策 ~4000 tokens
- 每 run ~15 次决策（事件触发）
- 240 组 × 15 次 × 4000 tokens ≈ 14M tokens
- 成本约 **$1 - $3**