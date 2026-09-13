# exp_dmde_01: LLM-DMDE vs 纯 DMDE 对比实验（N=M 平衡指派）

## 实验目的

在 N=M（5 UAV ↔ 5 Target）平衡指派场景下，对比 LLM-DMDE 与纯 DMDE 的求解性能差异。

## 实验流程

```
                    ┌─────────────────┐
                    │  1. 环境准备     │
                    │  DEM + 雷达威胁  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  2. 场景构建     │
                    │  UAV/Target/约束 │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  3. 代价矩阵     │
                    │  CostMatrixBuilder│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐   │    ┌─────────▼─────────┐
     │  DMDE 基线       │   │    │  LLM-DMDE          │
     │  DMDESolver      │   │    │  LLMEnhancedDMDE   │
     │  (30 runs)       │   │    │  Solver (30 runs)  │
     └────────┬────────┘   │    └─────────┬─────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  4. 指标统计     │
                    │  compute_metrics │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  5. 可视化       │
                    │  收敛曲线/箱线图 │
                    └─────────────────┘
```

## 依赖关系

```
experiments/exp_dmde/exp_dmde_01/           ← DMDE 基线实验
├── run.py                                  ← 直接运行
├── data_store.py                           ← 数据持久化
└── visualization/                          ← 复用的可视化工具

experiments/exp_llm_enhanced_dmde/exp_dmde_01/  ← LLM-DMDE 实验
├── run.py                                  ← 直接运行
├── data_store.py                           ← 数据持久化（含 LLM 决策日志）
├── config/llm_config.yaml                  ← LLM 配置（provider/model）
└── visualization/

src/algorithms/algorithm_dmde/              ← DMDE 算法（独立）
src/algorithms/algorithm_llm_enhanced_dmde/ ← LLM-DMDE 算法（独立）
src/environments/environment_dmde/          ← 共享：环境层
src/models/model_dmde/                      ← 共享：模型层
src/utils/utils_dmde/                       ← 共享：指标计算
```

## 运行步骤

### Step 1: 环境准备

```bash
# 安装依赖
pip install openai python-dotenv

# 配置 API Key（LLM-DMDE 需要）
# 在项目根目录 .env 中设置：
DEEPSEEK_API_KEY=sk-***
```

### Step 2: 运行 DMDE 基线（30 次）

```bash
cd /path/to/Experiments-For-Dissertation
EXP_N_RUNS=30 python experiments/exp_dmde/exp_dmde_01/run.py
```

输出：`experiments/exp_dmde/exp_dmde_01/results/exp_dmde_01_data.json`

### Step 3: 运行 LLM-DMDE（30 次）

```bash
EXP_N_RUNS=30 python experiments/exp_llm_enhanced_dmde/exp_dmde_01/run.py
```

输出：`experiments/exp_llm_enhanced_dmde/exp_dmde_01/results/exp_llm_dmde_01_data.json`

### Step 4: 消融实验（可选）

修改 `run.py` 中的 `modules` 参数：

```python
# 仅搜索控制（策略 + CR 联合决策）
modules={"search_controller": {"enabled": True, "interval": 50}}

# 仅种群初始化
modules={"population_init": {"enabled": True}}

# 全部启用
modules={
    "population_init": {"enabled": True},
    "search_controller": {"enabled": True, "interval": 50},
}
```

## 实验配置矩阵

| 配置                  | solver                  | modules                                                        | 说明           |
| --------------------- | ----------------------- | -------------------------------------------------------------- | -------------- |
| A: DMDE 基线          | `DMDESolver`            | N/A                                                            | 纯 DMDE        |
| B: LLM-DMDE (vanilla) | `LLMEnhancedDMDESolver` | `{}`                                                           | 退化一致性验证 |
| C: LLM-DMDE (SC)      | `LLMEnhancedDMDESolver` | `{"search_controller": {"enabled": true, "interval": 50}}`     | 仅搜索控制     |
| D: LLM-DMDE (PI)      | `LLMEnhancedDMDESolver` | `{"population_init": {"enabled": true}}`                       | 仅种群初始化   |
| E: LLM-DMDE (full)    | `LLMEnhancedDMDESolver` | `{"population_init": {"enabled": true}, "search_controller": {"enabled": true, "interval": 50}}` | 全部启用       |

## 评价指标

| 指标                     | 来源                             | 含义                   |
| ------------------------ | -------------------------------- | ---------------------- |
| `best_fitness`           | `SolverResult`                   | 最优适应度（越低越好） |
| `metrics.mean_fitness`   | `compute_metrics`                | 30 次均值              |
| `metrics.std_fitness`    | `compute_metrics`                | 30 次标准差            |
| `metrics.best_fitness`   | `compute_metrics`                | 30 次最优              |
| `metrics.n_feasible`     | `compute_metrics`                | 可行解次数             |
| `result.elapsed_seconds` | `SolverResult`                   | 单次运行时间           |
| 收敛代数                 | `cost_history` 首次达 95% 最终值 | 收敛速度               |
| Wilco 秩和检验           | 配对检验                         | 统计显著性 (p<0.05)    |

## 场景定义

```python
# 5 UAV
UAV(0, start=(91.05,29.55,3650), speed=(0.20,0.50), range=32000)
UAV(1, start=(91.10,29.53,3640), speed=(0.25,0.55), range=30000)
UAV(2, start=(91.15,29.56,3660), speed=(0.30,0.60), range=29000)
UAV(3, start=(91.03,29.60,3680), speed=(0.20,0.50), range=28000)
UAV(4, start=(91.08,29.58,3670), speed=(0.25,0.55), range=31000)

# 5 Target（T0/T3/T4 有时间窗）
Target(0, pos=(91.12,29.66,3700), w=1.0, tw=(40000,120000))
Target(1, pos=(91.10,29.70,3750), w=0.8)
Target(2, pos=(91.16,29.65,3680), w=0.9)
Target(3, pos=(91.20,29.72,3800), w=0.7, tw=(35000,100000))
Target(4, pos=(91.08,29.68,3720), w=0.6, tw=(30000,90000))

# 约束：航程✓ 时间窗✓ 时序✗ 同步✗
```

## 结果展示

| 图表         | 内容                                   |
| ------------ | -------------------------------------- |
| 收敛曲线对比 | DMDE vs LLM-DMDE，30 次均值±标准差阴影 |
| 箱线图       | 各配置 fitness 分布                    |
| 消融柱状图   | 各 modules 配置的 fitness 对比         |
| 主结果表     | 均值±标准差，最优加粗，Wilcoxon p<0.05 标 † |

## 注意事项

- **API 成本**：30 次 × 4 配置(B~E) ≈ 1200 次 LLM 调用，提前估算费用
- **种子对齐**：DMDE 和 LLM-DMDE 用相同 `seed=run_idx`，支持配对检验
- **先小后大**：先用 `EXP_N_RUNS=3` 跑通脚本，再跑正式 30 次
- **search_controller**：统一搜索控制器，一次 LLM 调用同时决定变异策略和 CR（替代旧版分离的 operator_selection + cr_control）
