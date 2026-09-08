# Experiments-For-Dissertation — 项目目录结构

> 生成日期：2026-09-08
> 说明：本文档为项目完整目录结构（已排除 `.git`、`.venv`、`__pycache__` 等无关内容）。

## 目录树

```
Experiments-For-Dissertation/
│
├── .gitignore                  # Git 忽略规则
├── .python-version             # Python 版本声明
├── pyproject.toml              # 项目配置与依赖声明
├── README.md                   # 项目说明
├── uv.lock                     # uv 依赖锁文件
│
├── docs/                       # 项目文档目录
│
├── src/                        # 核心源码（库代码）
│   ├── __init__.py
│   │
│   ├── algorithms/             # 优化算法层
│   │   ├── __init__.py
│   │   └── algorithm_dmde/     # DMDE（离散映射差分进化）算法
│   │       ├── __init__.py
│   │       │
│   │       ├── base/           # 基类 / 接口规范
│   │       │   ├── __init__.py
│   │       │   └── base_optimizer.py    # 优化器接口规范
│   │       │
│   │       ├── operators/      # 连续差分进化算子
│   │       │   ├── __init__.py
│   │       │   ├── crossover.py          # 交叉算子
│   │       │   ├── extinction.py         # 灭绝/重启机制
│   │       │   ├── mutation.py           # 变异算子
│   │       │   └── scale_factor.py       # 缩放因子自适应
│   │       │
│   │       ├── representation/ # 基因表征与空间映射机制
│   │       │   ├── __init__.py
│   │       │   ├── encoder.py            # 统一三元组基因生成器（规则 3.1-3.3）
│   │       │   ├── mapper.py             # 离散→连续正向映射 phi（公式 3-5）
│   │       │   ├── inverse_mapper.py     # 连续→离散反映射 phi'（公式 3-7）
│   │       │   └── repair_rules/         # 反映射冲突消解策略簇（规则 3.4-3.6）
│   │       │       ├── __init__.py
│   │       │       ├── invalid_mutator.py    # 规则 3.6：越界/重复/无效值随机变异修补
│   │       │       ├── nearest_match.py      # 规则 3.4：最近邻空间欧氏距离匹配
│   │       │       └── unique_filter.py      # 规则 3.5：独占分配与掩码(Inf)行/列剔除
│   │       │
│   │       └── solvers/        # 求解器实现层
│   │           ├── __init__.py
│   │           ├── dmde_solver.py          # 离散映射差分求解器（算法 3.1 完整流程）
│   │           └── baseline_solvers/       # 论文对比算法（第三章表 3-5）
│   │               ├── __init__.py
│   │               ├── cmtap_ga.py         # 协同多目标遗传算法（文献[65]）
│   │               ├── pmx_de.py           # 部分映射交叉差分算法
│   │               └── sort_de.py          # 排序差分进化算法（文献[105]）
│   │
│   ├── environments/           # 环境层
│   │   ├── __init__.py
│   │   └── env_dmde/           # DMDE 问题环境
│   │       ├── __init__.py
│   │       ├── context.py      # 环境上下文
│   │       ├── evaluator.py    # 评估器
│   │       ├── instance.py     # 实例定义
│   │       └── transition.py   # 状态转移
│   │
│   ├── models/                 # 领域模型层
│   │   ├── __init__.py
│   │   └── model_dmde/
│   │       ├── __init__.py
│   │       ├── constraints/    # 约束建模
│   │       │   └── __init__.py
│   │       ├── cost/           # 成本/目标建模
│   │       │   └── __init__.py
│   │       └── entities/       # 实体定义
│   │           └── __init__.py
│   │
│   └── utils/                  # 工具层
│       ├── __init__.py
│       └── utils_dmde/
│           ├── __init__.py
│           └── metrics.py      # 评价指标
│
├── experiments/                # 实验层
│   └── exp_dmde/
│       └── exp_dmde_01/
│           ├── __init__.py
│           ├── config/         # 实验配置
│           │   └── .gitkeep
│           ├── data/           # 实验数据
│           │   └── .gitkeep
│           ├── results/        # 实验结果
│           │   └── .gitkeep
│           └── visualization/  # 可视化脚本/图表
│
└── tests/                      # 测试层
    └── dmde/
        └── solver/
            ├── test_encoder.py            # 编码器测试
            ├── test_mapping_operator.py   # 映射算子测试
            └── test_solver.py             # 求解器测试
```

## 模块职责速览

| 层级 | 路径                             | 职责                                        |
| ---- | -------------------------------- | ------------------------------------------- |
| 算法 | `src/algorithms/algorithm_dmde/` | DMDE 算法主体：算子、表征、求解器           |
| 表征 | `.../representation/`            | 离散三元组基因编码 ↔ 连续空间映射及冲突修复 |
| 求解 | `.../solvers/`                   | DMDE 主求解流程 + 论文对比基线算法          |
| 环境 | `src/environments/env_dmde/`     | 问题实例、评估与状态转移                    |
| 模型 | `src/models/model_dmde/`         | 约束、成本、实体等领域建模                  |
| 工具 | `src/utils/utils_dmde/`          | 通用指标与辅助工具                          |
| 实验 | `experiments/exp_dmde/`          | 实验配置、数据、结果与可视化                |
| 测试 | `tests/dmde/`                    | 编码、映射算子与求解器单元测试              |

## 核心数据流

```
encoder(离散基因) --mapper.phi--> 连续个体
    --(DE 算子进化)--> 连续个体 --inverse_mapper.phi'--> 离散基因
inverse_mapper 还原过程中按需调用 repair_rules 消解各类冲突
最终由 dmde_solver 编排并输出结果，与 baseline_solvers 统一口径对比
```

```
[场景输入 (UAV参数/Target参数/地形雷达/协同约束)]
                       │
                       ▼
         models/model_dmde/cost/
     (垂直切面法生成统一代价矩阵 C_cost)
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
algorithms/.../encoder.py       models/.../constraints/
 (按规则3.1-3.3初始化三元组个体)  (单机/协同约束检测与适应度函数)
       │                               ▲
       ▼                               │
algorithms/.../mapper.py               │
 (提取三元组代价值为连续向量)           │
       │                               │
       ▼                               │
algorithms/.../operators/              │
 (混合差变异 + 动态CR + GMR灭绝)       │
       │ (得到连续实数解 C')           │
       ▼                               │
algorithms/.../inverse_mapper.py       │
 (基于规则3.4-3.6掩码消解，映射回三元组) │
       │                               │
       ▼ (得到可行分配个体)             │
       └───────────────────────────────┘
```

## 流程图

```mermaid
graph TD
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px;
    classDef model fill:#fff3e0,stroke:#f57c00,stroke-width:1.5px;
    classDef algo fill:#e8f5e9,stroke:#388e3c,stroke-width:1.5px;
    classDef out fill:#fce4ec,stroke:#c2185b,stroke-width:1.5px;

    subgraph InputData ["1. 初始输入与战场环境"]
        DEM["3D DEM 地形与雷达威胁"]:::input
        UAVs["UAV 属性 (Vmin, Vmax, MaxD)"]:::input
        Targets["目标属性 (坐标, 权重W, 时序/时窗)"]:::input
    end

    subgraph ModelLayer ["2. 建模与代价评估 (models/)"]
        CostEst["垂直切面代价估算器<br/>(cost_estimator)"]:::model
        CostMat["统一航程代价矩阵 C_cost<br/>(N=M / N>M / N<M)"]:::model
        Evaluator["适应度与约束评价器<br/>(Fitness & Penalty)"]:::model
    end

    subgraph AlgoLayer ["3. DMDE 求解器 (algorithms/)"]
        Encoder["统一三元组基因初始化<br/>(encoder.py)"]:::algo
        Mapper["正向映射 φ<br/>(mapper: 提取实数代价向量)"]:::algo
        DE_Op["连续空间差分与交叉<br/>(混合变异 + 动态 CR)"]:::algo
        InvMapper["反映射协调器 φ'<br/>(规则 3.4 最近邻 + 3.5 掩码Inf + 3.6 补齐)"]:::algo
        GMR_Op["GMR 灭绝/重置算子<br/>(保留 P_best)"]:::algo
    end

    subgraph OutputData ["4. 最优决策输出"]
        BestPlan["最优目标分配方案<br/>[(Ui, Tj, Cost), ...]"]:::out
    end

    %% 数据连接
    DEM & UAVs & Targets --> CostEst
    CostEst --> CostMat
    CostMat --> Encoder
    CostMat --> InvMapper

    Encoder -->|离散初始种群| Mapper
    Mapper -->|连续代价值向量| DE_Op
    DE_Op -->|临时连续解 C'| InvMapper
    InvMapper -->|可行离散个体| Evaluator

    Evaluator -->|适应度反馈| DE_Op
    DE_Op -.->|早熟/收敛停滞| GMR_Op
    GMR_Op -.->|注入重置种群| Mapper

    Evaluator -->|满足终止代数| BestPlan
```

```mermaid
flowchart TD
    Start([开始: 初始化实验参数]) --> InitCost[生成三维垂直切面代价矩阵 C_cost]
    InitCost --> InitPop[生成初始种群: 规则 3.1-3.3 构造三元组个体]
    InitPop --> EvalPop[评估初始种群适应度: 航程 + 时间 + 约束违背罚项]

    EvalPop --> LoopGen{代数 gen < MaxGen ?}

    LoopGen -- 否 --> Output([输出全局最优分配解 Ind_best])
    LoopGen -- 是 --> UpdateParams[计算当前代: 动态交叉率 CR & 缩放因子 F]

    UpdateParams --> CheckExtinct{触发 GMR 灭绝条件?}
    CheckExtinct -- 是 --> Extinct[保留 P_best, 随机重生其余劣质个体] --> LoopPop
    CheckExtinct -- 否 --> LoopPop

    subgraph MutationAndCrossover ["连续空间差分演化 (公式 3-10)"]
        LoopPop[遍历种群个体] --> ForwardMap["正向映射: 提取个体代价值向量 C(t)"]
        ForwardMap --> RandCheck{"rand(0,1) < CR ?"}
        RandCheck -- 是 --> Strategy1["探索策略: DE/rand/1 差分计算"]
        RandCheck -- 否 --> Strategy2["开发策略: DE/best/2 差分计算"]
        Strategy1 & Strategy2 --> TempVector["得到临时连续解向量 C'(t)"]
    end

    subgraph InverseMappingProcess ["离散反映射与冲突消解 (规则 3.4 - 3.6)"]
        TempVector --> CopyMat["复制代价矩阵备份 C_work"]
        CopyMat --> MatchLoop["遍历 C'(t) 中的每个代价值"]
        MatchLoop --> Nearest["规则 3.4: 在 C_work 中找欧氏距离最近邻 (i, j)"]
        Nearest --> MaskRule{"检查对应模型类型"}
        MaskRule -- "N=M" --> Mask1["规则 3.5a: 行/列均置 Inf"]
        MaskRule -- "N>M" --> Mask2["规则 3.5b: 仅 UAV 所在行置 Inf"]
        MaskRule -- "N<M" --> Mask3["规则 3.5c: 目标列置 Inf, UAV 行更新为转场代价"]
        Mask1 & Mask2 & Mask3 --> GeneAppend["收集生成有效三元组基因"]
        GeneAppend --> MatchFinish{"该个体分配完成?"}
        MatchFinish -- 否 --> MatchLoop
        MatchFinish -- 是 --> MutatePatch["规则 3.6: 若有残余未分配, 从可用 C_work 随机补齐"]
    end

    MutatePatch --> EvalNew[计算新子代个体的综合适应度 f_new]
    EvalNew --> GreedySelect{"f_new < f_old (贪心选择) ?"}
    GreedySelect -- 是 --> AcceptNew[接收新个体并更新种群及 P_best]
    GreedySelect -- 否 --> KeepOld[保留原个体]

    AcceptNew & KeepOld --> NextInd{个体遍历完毕?}
    NextInd -- 否 --> LoopPop
    NextInd -- 是 --> GenInc[gen = gen + 1] --> LoopGen
```
