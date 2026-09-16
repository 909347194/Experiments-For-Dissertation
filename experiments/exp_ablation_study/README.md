# 消融实验 (Ablation Study)

## 实验目的

验证 LLM-DMDE 两个核心模块的独立贡献及协同效应：
- **PopInit**：LLM-Guided Population Initialization
- **CR Control**：LLM-Guided Adaptive Crossover Rate Control

## 消融配置

| 编号 | 配置名 | PopInit | CR Control | 目的 |
|------|--------|---------|------------|------|
| **A0** | Vanilla DMDE | ✗ | ✗ | 基线（纯传统 DMDE） |
| **A1** | DMDE + CR Control | ✗ | ✓ | 验证 CR 自适应模块贡献 |
| **A2** | DMDE + PopInit | ✓ | ✗ | 验证初始化模块贡献 |
| **A3** | Full LLM-DMDE | ✓ | ✓ | 验证双模块协同效应 |

## 实验场景

选择原则：一个最简单场景（排除复杂度干扰）+ 一个最复杂场景（验证高难度下有效性）

| 场景 | Model | N (UAVs) | M (Targets) | 选择理由 |
|------|-------|----------|-------------|----------|
| **S1** | balanced (N=M) | 10 | 10 | 小规模基准，PopInit 最简单（一对一匹配） |
| **S2** | srp (N<M) | 10 | 20 | PopInit 需同时决定分配 + 巡回顺序（高复杂度） |

**总计**: 4 配置 × 2 场景 × 30 次独立运行 = **240 组实验**

## 目录结构

```
experiments/exp_ablation_study/
├── README.md                  # 本文档
├── run_ablation.py            # 统一入口脚本
├── analyze.py                 # 统计分析 + 可视化 + LaTeX 表格生成
│
├── core/                      # 核心逻辑包（各文件 <120 行）
│   ├── config.py              # 实验配置数据类
│   ├── scenario.py            # 场景构建 + DEM 加载
│   ├── runner.py              # A0-A3 单次运行封装
│   └── results.py             # 结果保存/加载 + YAML 解析
│
├── scenarios/                 # 场景定义包
│   ├── s1_balanced.py         # S1: balanced N=M=10
│   └── s2_srp.py              # S2: srp N=10,M=20
│
├── S1_balanced_N10_M10/       # 场景 1 工作目录
│   ├── shared_config.py       # 薄包装层 → scenarios.s1_balanced
│   ├── data/chengguan_district_dem.tif  # DEM 地形数据
│   ├── A0_dmde/               # Vanilla DMDE
│   │   ├── run.py             # 37 行，无 LLM 配置
│   │   ├── config/
│   │   └── results/           # 30 个 .json 结果文件
│   ├── A1_cr_control/         # 仅 CR Control
│   │   ├── run.py             # 48 行，自动读取 llm_config.yaml
│   │   ├── config/llm_config.yaml
│   │   └── results/
│   ├── A2_pop_init/           # 仅 PopInit
│   │   ├── run.py
│   │   ├── config/llm_config.yaml
│   │   └── results/
│   └── A3_full/               # Full LLM-DMDE
│       ├── run.py
│       ├── config/llm_config.yaml
│       └── results/
│
├── S2_srp_N10_M20/            # 场景 2 工作目录（同 S1 结构）
│   └── ...
│
└── figures/                   # analyze.py 输出目录
    ├── convergence_comparison_S1.png
    ├── convergence_comparison_S2.png
    ├── boxplot_S1.png
    ├── boxplot_S2.png
    ├── time_breakdown.png
    ├── ablation_results.tex   # SCI 标准三线表（解质量）
    ├── ablation_significance.tex  # Wilcoxon 检验表
    └── *.csv                  # 原始数据导出
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
# 【推荐】运行全部 8 组实验（每组 30 runs，约需 2-4 小时）
uv run python run_ablation.py --runs 30

# 只运行场景 S1（4 组 × 30 runs）
uv run python run_ablation.py --scenario S1 --runs 30

# 只运行 A3 配置（Full LLM-DMDE）
uv run python run_ablation.py --config A3 --runs 30

# 单独运行某一组（调试用）
cd S1_balanced_N10_M10/A3_full && uv run python run.py --runs 5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--runs` | 30 | 独立运行次数（建议≥30 保证统计显著性） |
| `--scenario` | all | 选择场景：S1, S2, 或 all |
| `--config` | all | 选择配置：A0, A1, A2, A3, 或 all |
| `--parallel` | 1 | 并行进程数（慎用，可能触发 API 限流） |

## 分析与可视化

```bash
# 生成所有图表和 LaTeX 表格
uv run python analyze.py

# 输出文件：
# - figures/convergence_comparison_S1.png  # 收敛曲线对比
# - figures/boxplot_S1.png                 # 解质量箱线图
# - figures/time_breakdown.png             # 时间开销堆叠柱状图
# - figures/ablation_results.tex           # SCI 标准主表（带显著性标记）
# - figures/ablation_significance.tex      # Wilcoxon p 值矩阵表
# - figures/summary_table.csv              # 汇总数据（Excel 可读）
```

### LaTeX 表格使用说明

生成的 `.tex` 文件可直接插入论文：

```latex
\usepackage{booktabs}  % 在导言区添加

% 插入主结果表
\input{figures/ablation_results.tex}

% 插入显著性检验表
\input{figures/ablation_significance.tex}
```

**表格特性**：
- ✅ 自动最优值加粗 (`\textbf{}`)
- ✅ 显著性标记：`*` (p<0.05), `**` (p<0.01)
- ✅ 专业三线表风格（无竖线）
- ✅ 完整表注说明缩写和统计方法

## 评估指标

### 解质量指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **Best Fitness** | 最优解质量 | 30 runs 中的最小值 |
| **Mean ± Std** | 平均解质量 ± 离散度 | 均值和标准差 |
| **Median** | 中位数（鲁棒性） | 50% 分位值 |

### 收敛性能指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **Conv. Gen** | 收敛速度 | 达到 95% 最终最优解所需代数 |
| **Success Rate** | 成功率 | 适应度低于阈值的运行比例 |

### 时间开销指标（分口径）

| 指标 | 包含内容 | 用途 |
|------|----------|------|
| **Total Time(s)** | 全部耗时 | 实际总耗时 |
| **DMDE Time(s)** | 仅进化计算 | 算法本身开销 |
| **LLM Init Time(s)** | PopInit 模块 | 初始化阶段 LLM 代价 |
| **LLM CR Time(s)** | CR Control 模块 | 进化阶段 LLM 代价 |
| **LLM Calls** | 总调用次数 | 评估 API 成本 |
| **Avg Latency(ms)** | 平均延迟 | LLM 响应速度 |

### 统计检验

| 检验 | 用途 | 显著性阈值 |
|------|------|------------|
| **Wilcoxon signed-rank** | A3 vs A0/A1/A2 配对检验 | p < 0.05 (*) / p < 0.01 (**) |

## 预期结论与验证点

| 对比 | 预期结果 | 验证点 | 理论依据 |
|------|----------|--------|----------|
| **A1 vs A0** | A1 更优 (p<0.05) | CR 自适应 > 固定公式 | 动态参数调整适应搜索状态 |
| **A2 vs A0** | A2 收敛更快 | 知识驱动初始化 > 纯随机 | LLM 提供高质量起点 |
| **A3 vs A1** | A3 ≥ A1 | PopInit 提供更好的起点 | 双重优势叠加 |
| **A3 vs A2** | A3 ≥ A2 | CR Control 持续优化 | 进化中动态调整 |
| **A3 vs A0** | A3 显著优于 (p<0.01) | 双模块协同增益 | 1+1>2 效应 |

**关键验证**: `A3 > max(A1, A2)` 证明两个模块**互补而非冗余**。

## 常见问题 (FAQ)

### Q1: 如何切换 LLM 模型？
修改 `S*/A*/config/llm_config.yaml` 中的 `model` 字段：
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
结果自动保存到 `results/` 目录，重新运行会自动跳过已完成的 runs：
```bash
uv run python run_ablation.py --runs 30  # 自动检测已有结果
```

### Q4: 如何自定义随机种子？
默认使用 `42, 43, ..., 71`（30 个连续种子）。可在 `core/config.py` 中修改：
```python
RANDOM_SEEDS = list(range(42, 72))  # 或自定义列表
```

### Q5: LLM API 成本估算？
以 SiliconFlow + Qwen3.5-9B 为例（输入/输出 $0.10/$0.15 per M tokens）：
- 单次 PopInit: ~2000 tokens × $0.0001 = $0.0002
- 单次 CR Control: ~500 tokens × $0.0001 = $0.00005
- 240 组实验总成本：约 **$0.4 - $0.8**

## 参考文献

[1] Wilcoxon F. Individual comparisons by ranking methods[J]. Biometrics Bulletin, 1945.
[2] 论文方法论章节（LLM-DMDE 框架说明）