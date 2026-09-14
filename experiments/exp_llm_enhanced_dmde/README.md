# LLM 增强 DMDE 实验

本目录包含 LLM 增强 DMDE 算法的全部实验，以及与原始 DMDE 基线的对比实验。

## 目录结构

```
experiments/
├── exp_dmde/                          # DMDE 基线实验（对照组）
│   ├── exp_dmde_01/                   # N=M 平衡指派
│   ├── exp_dmde_02/                   # N>M 多对一
│   └── exp_dmde_03/                   # N<M 群巡游 (SRP)
│
└── exp_llm_enhanced_dmde/             # LLM 增强实验（本目录）
    ├── exp_llm_dmde_01/               # N=M 平衡指派（LLM 版）
    ├── exp_llm_dmde_02/               # N>M 多对一（LLM 版）
    ├── exp_llm_dmde_03/               # N<M 群巡游（LLM 版）
    └── exp_llm_dmde_comparison/       # 对比实验脚本
```

## 三大场景

| 场景 | 代号 | UAV : Target | 基线 | LLM 版 | 含义 |
|------|------|-------------|------|--------|------|
| 平衡指派 | `nm` | 10 : 10 | exp_dmde_01 | exp_llm_dmde_01 | 一一对应，每架 UAV 打一个目标 |
| 多对一 | `ngt` | 10 : 4 | exp_dmde_02 | exp_llm_dmde_02 | 多架 UAV 协同攻击同一目标 |
| 群巡游 | `nlt` | 4 : 10 | exp_dmde_03 | exp_llm_dmde_03 | 每架 UAV 巡游多个目标 (SRP) |

### 实验规模

| 规模 | N=M | N>M | N<M |
|------|-----|-----|-----|
| small | 5U / 5T | 5U / 2T | 2U / 5T |
| **medium** (默认) | 10U / 10T | 10U / 4T | 4U / 10T |
| large | 20U / 20T | 20U / 8T | 8U / 20T |

### 约束配置

| 约束类型 | N=M | N>M | N<M |
|----------|-----|-----|-----|
| 航程约束 (max_range) | ✅ | ✅ | ✅ |
| 时间窗约束 (time_window) | ✅ | ✅ | ✅ |
| 时序约束 (sequence_group) | — | ✅ | ✅ |
| 同时到达约束 (sync) | — | ✅ | — |

---

## 环境准备

### 1. 安装依赖

项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
# 安装 uv（如未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖（自动创建 .venv）
uv sync
```

或使用 pip：

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

LLM 实验需要 DeepSeek API Key。在项目根目录创建 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> DMDE 基线实验不需要 API Key，可独立运行。

---

## 如何启动

### 方式一：运行单个实验

**DMDE 基线（无需 API Key）：**

```bash
# N=M 平衡指派，运行 30 次
cd experiments/exp_dmde/exp_dmde_01
uv run python run.py

# 指定运行次数
EXP_N_RUNS=50 uv run python run.py

# 切换规模
EXP_SCALE=large uv run python run.py
```

**LLM-DMDE 增强版：**

```bash
# N=M 平衡指派
cd experiments/exp_llm_enhanced_dmde/exp_llm_dmde_01
uv run python run.py

# N>M 多对一
cd experiments/exp_llm_enhanced_dmde/exp_llm_dmde_02
uv run python run.py

# N<M 群巡游
cd experiments/exp_llm_enhanced_dmde/exp_llm_dmde_03
uv run python run.py
```

### 方式二：运行对比实验（推荐）

对比脚本会**同时运行 DMDE 基线和 LLM-DMDE**，并自动生成对比报告。

```bash
cd experiments/exp_llm_enhanced_dmde/exp_llm_dmde_comparison
```

**单场景对比：**

```bash
# N=M 场景，30 次运行（默认）
uv run python run_comparison.py

# N>M 场景
uv run python run_comparison.py --scenario ngt

# N<M 场景
uv run python run_comparison.py --scenario nlt
```

**全场景对比：**

```bash
# 三个场景各跑 30 次
uv run python run_comparison.py --scenarios all
```

**多规模实验：**

```bash
# 三个场景 × 三种规模，共 9 组实验
uv run python run_comparison.py --scenarios all --sizes small medium large
```

**生成 LaTeX 表格（用于学位论文）：**

```bash
# 全场景 + 全规模 + LaTeX 输出
uv run python run_comparison.py --scenarios all --sizes small medium large --format latex

# 同时生成 Markdown 和 LaTeX
uv run python run_comparison.py --scenarios all --format both
```

**大规模实验（自定义求解器参数）：**

```bash
# 大规模需要更大种群和更多代数
uv run python run_comparison.py --sizes large --solver-params '{"pop_size":150,"max_generations":2000}'
```

**消融实验：**

```bash
# 禁用 LLM 模块，分析各模块贡献
uv run python run_comparison.py --ablation
```

**只跑其中一个算法：**

```bash
# 只跑基线
uv run python run_comparison.py --only baseline

# 只跑 LLM 版
uv run python run_comparison.py --only llm
```

### 方式三：使用已有结果直接生成对比

如果实验结果已有，可以跳过运行，直接生成对比报告：

```bash
uv run python run_comparison.py \
  --baseline-data ../../exp_dmde/exp_dmde_01/results/xxx_data.json \
  --llm-data ../exp_llm_dmde_01/results/yyy_data.json
```

---

## 输出说明

### 实验结果目录

每个实验的结果保存在各自的 `results/` 目录下：

```
exp_llm_dmde_01/results/
├── llm_dmde_balanced_run_data.json    # 完整实验数据（含 LLM 决策日志）
├── llm_log_run_0.md                   # 每次运行的 LLM 交互日志
└── figures/                           # 可视化图表
    ├── convergence.png                # 收敛曲线
    ├── cost_matrix.png                # 代价矩阵热力图
    ├── assignment.png                 # 分配方案
    ├── dem3d.png                      # DEM 三维地形
    └── comparison.png                 # 场景对比
```

### 对比实验输出

```
exp_llm_dmde_comparison/results/
├── comparison_nm.md                   # N=M 对比表 (Markdown)
├── comparison_ngt.md                  # N>M 对比表
├── comparison_nlt.md                  # N<M 对比表
├── summary_report.md                  # 汇总报告
├── meta_nm.json                       # 元数据
├── figures/
│   ├── boxplot_nm.png                 # 箱线图
│   ├── convergence_nm.png             # 收敛曲线
│   ├── improvement_nm.png             # 改进幅度
│   ├── feasibility_nm.png             # 可行解率
│   ├── time_nm.png                    # 耗时对比
│   └── scaling.png                    # 多规模缩放曲线
└── latex/
    ├── summary_table.tex              # 汇总 LaTeX 表格
    ├── table_nm.tex                   # N=M 场景表格
    ├── table_ngt.tex                  # N>M 场景表格
    └── table_nlt.tex                  # N<M 场景表格
```

### LaTeX 表格使用

将生成的 `.tex` 文件复制到论文项目中，按如下方式引用：

```latex
\usepackage{booktabs}  % 三线表

% 方式 1：直接输入汇总表
\input{results/latex/summary_table.tex}

% 方式 2：单独引用某场景
\input{results/latex/table_nm.tex}
```

---

## LLM 配置

LLM 模型配置在各实验的 `config/llm_config.yaml` 中：

```yaml
provider: deepseek          # deepseek | openai | custom
model: deepseek-flash       # 模型名称
temperature: 0.7
max_tokens: 8192
reasoning_effort: none      # none | low | high | max
```

Prompt 模板在 `config/prompts/search_controller.txt`，可根据场景自定义。

---

## 常见问题

**Q: 运行报错 `No API key`？**
A: LLM 实验需要配置 `.env` 文件中的 `DEEPSEEK_API_KEY`。基线实验不需要。

**Q: 大规模实验很慢？**
A: 使用 `--solver-params` 调整种群大小和代数，或减少 `--runs` 次数先验证流程。

**Q: 如何只重绘图表不重跑实验？**
A: 各实验目录下有 `plot_from_saved.py`，指向已有的结果 JSON 即可：
```bash
uv run python plot_from_saved.py --data results/xxx_data.json
```

**Q: `uv run` 和 `python` 有什么区别？**
A: `uv run` 会自动激活项目的 `.venv` 虚拟环境，确保依赖版本一致。建议统一使用 `uv run`。
