# 消融实验 (Ablation Study)

## 实验目的

验证 LLM-DMDE 两个核心模块的独立贡献及协同效应：
- **PopInit**：LLM-Guided Population Initialization
- **CR Control**：LLM-Guided Adaptive Crossover Rate Control

## 消融配置

| 编号 | 配置 | PopInit | CR Control | 目的 |
|---|---|---|---|---|
| **A0** | Vanilla DMDE | ✗ | ✗ | 基线 |
| **A1** | DMDE + CR Control | ✗ | ✓ | 仅 CR 模块贡献 |
| **A2** | DMDE + PopInit | ✓ | ✗ | 仅初始化模块贡献 |
| **A3** | Full LLM-DMDE | ✓ | ✓ | 双模块协同 |

## 实验场景

| 场景 | Model | N | M | 选择理由 |
|---|---|---|---|---|
| **S1** | balanced (N=M) | 10 | 10 | 小规模基准，PopInit 最简单 |
| **S2** | srp (N<M) | 10 | 20 | PopInit 需同时决定分配 + 巡回顺序 |

## 目录结构

```
exp_ablation_study/
├── README.md                          # 本文件
├── run_ablation.py                    # 统一入口
├── analyze.py                         # 跨配置对比分析
│
├── S1_balanced_N10_M10/               # 场景 1：balanced
│   ├── shared_config.py               # 场景公共配置（UAV/Target/约束）
│   ├── data/                          # 共享数据
│   ├── A0_dmde/                       # Vanilla DMDE
│   │   ├── run.py
│   │   ├── config/
│   │   └── results/
│   ├── A1_cr_control/                 # 仅 CR Control
│   │   ├── run.py
│   │   ├── config/llm_config.yaml
│   │   └── results/
│   ├── A2_pop_init/                   # 仅 PopInit
│   │   ├── run.py
│   │   ├── config/llm_config.yaml
│   │   └── results/
│   └── A3_full/                       # Full LLM-DMDE
│       ├── run.py
│       ├── config/llm_config.yaml
│       └── results/
│
├── S2_srp_N10_M20/                    # 场景 2：srp
│   └── (同 S1 结构)
│
└── figures/                           # analyze.py 输出
```

## 运行方式

```bash
# 运行全部 8 组实验（每组 30 runs）
python run_ablation.py --runs 30

# 只运行场景 S1
python run_ablation.py --scenario S1 --runs 30

# 只运行 A3 配置
python run_ablation.py --config A3 --runs 30

# 单独运行某一组
cd S1_balanced_N10_M10/A3_full && python run.py --runs 30
```

## 分析

```bash
# 生成对比表 + 图表
python analyze.py

# 输出到 figures/
# - convergence_comparison_S1.png / S2.png
# - boxplot_S1.png / S2.png
# - time_breakdown.png
# - wilcoxon_table.csv
# - summary_table.csv
```

## 评估指标

| 指标 | 含义 |
|---|---|
| Best Fitness | 30 runs 最优适应度 |
| Mean ± Std | 平均解质量 ± 标准差 |
| Median | 中位数（鲁棒性） |
| Conv. Gen | 达到 95% 最优解所需代数 |
| Total Time(s) | 总耗时 |
| LLM Time(s) | LLM 调用耗时 |
| LLM Calls | LLM 调用次数 |
| Wilcoxon p | A3 vs A0/A1/A2 统计显著性 |

## 预期结论

| 对比 | 预期 | 验证点 |
|---|---|---|
| A1 vs A0 | A1 更优 | CR 自适应 > 固定公式 |
| A2 vs A0 | A2 收敛更快 | 知识驱动初始化 > 纯随机 |
| A3 vs max(A1,A2) | A3 ≥ 两者 | 互补非冗余 |
| A3 vs A0 | A3 显著优于 | 双模块协同增益 |