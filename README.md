```

muas_task_allocation/
├── config/                         # 配置文件目录
│   ├── default_config.yaml         # 仿真参数、种群大小、迭代次数等
│   └── scenario_configs.py         # N=M, N>M, N<M 等不同实验场景配置
│
├── data/                           # 地形及地图数据
│   └── dem_map.npy                 # 3D DEM数字高程数据（或生成脚本）
│
├── src/                            # 源代码主目录
│   ├── __init__.py
│   │
│   ├── environment/                # 战场环境模块（第二章）
│   │   ├── __init__.py
│   │   ├── dem_terrain.py          # DEM 高程地图加载与剖面提取
│   │   ├── radar_threat.py         # 雷达 threat 半球计算
│   │   └── cost_estimator.py       # 基于垂直切面的三维航程代价估算器
│   │
│   ├── models/                     # 任务模型与评价模块（第二章）
│   │   ├── __init__.py
│   │   ├── uav_target_problem.py   # 统一目标分配问题定义类
│   │   ├── cost_matrix.py          # 构造 N=M, N>M, N<M 的代价矩阵 C_cost
│   │   └── constraints.py          # 单机/协同约束检测与惩罚计算
│   │
│   ├── algorithms/                 # 核心优化算法模块（第三章）
│   │   ├── __init__.py
│   │   ├── dna_encoder.py          # 统一基因三元组编码与初始化
│   │   ├── mapping_operator.py     # 正向映射与反映射规则 (Rule 3.4 - 3.6)
│   │   ├── dynamic_strategies.py   # 动态CR、混合变异策略、GMR灭绝算子
│   │   └── dmde_solver.py          # DMDE 算法主流程迭代器
│   │
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       └── metrics.py              # 统计指标计算（平均代价、违约率等）
│
├── visualization/                  # 可视化模块
│   ├── __init__.py
│   ├── plot_3d_battlefield.py      # 三维战场与分配结果连线绘制
│   └── plot_convergence.py         # 收敛曲线绘制
│
├── tests/                          # 单元测试
│   ├── test_cost_estimator.py
│   └── test_mapping_rules.py
│
├── main.py                         # 主运行入口脚本
├── requirements.txt                # 依赖包列表 (numpy, scipy, matplotlib, pyyaml)
└── README.md                       # 项目说明文档
```

细化模块：

```
── models/                         # 领域模型与数学评估层（第二章）
│   ├── __init__.py
│   │
│   ├── entities/                   # 1. 物理实体与数据定义
│   │   ├── __init__.py
│   │   ├── uav.py                  # 单机属性 (航程上限MaxD、速度区间[Vmin, Vmax]、油耗等)
│   │   ├── target.py               # 目标属性 (坐标、价值权重W、时窗需求)
│   │   └── solution.py             # 解结构体 (三元组集合: [(UAV_id, Target_id, Cost), ...])
│   │
│   ├── cost/                       # 2. 统一代价体系构建
│   │   ├── __init__.py
│   │   ├── cost_matrix_base.py     # 代价矩阵基类 (统一接口规范)
│   │   ├── balanced_cost.py        # N=M 方阵构建 (公式 2-30)
│   │   ├── overloaded_cost.py      # N>M 非方阵多对一构建 (公式 2-31)
│   │   └── srp_cost.py             # N<M 目标转场对称方阵构建 (SRP群巡游, 公式 2-32)
│   │
│   └── constraints/                # 3. 约束与适应度评估体系
│       ├── __init__.py
│       ├── evaluator.py            # 综合适应度评估器 (公式 2-14: 航程+时间+违约罚函数)
│       ├── single_constraints.py   # 单机飞行性能约束检测 (最大航程、最大飞行时间)
│       └── coop_constraints.py     # 协同约束检测 (时序约束 Tsort、同时到达约束 S_constrain 等)
│
├── algorithms/                     # 求解器与算子层（第三章）
│   ├── __init__.py
│   │
│   ├── base/                       # 1. 算法抽象基类
│   │   ├── __init__.py
│   │   └── base_optimizer.py       # 优化器接口规范 (fit, step, get_best)
│   │
│   ├── representation/             # 2. 基因表征与空间映射机制 (论文核心创新)
│   │   ├── __init__.py
│   │   ├── encoder.py              # 统一三元组基因生成器 (规则 3.1 - 3.3)
│   │   ├── mapper.py               # 离散到连续的正向空间映射 phi (公式 3-5)
│   │   ├── inverse_mapper.py       # 连续到离散的反映射协调器 phi' (公式 3-7)
│   │   └── repair_rules/           # 反映射冲突消解策略簇 (对应论文具体规则)
│   │       ├── __init__.py
│   │       ├── nearest_match.py    # 规则 3.4: 最近邻空间欧氏距离匹配
│   │       ├── unique_filter.py    # 规则 3.5: 独占分配与掩码(Inf)行/列剔除
│   │       └── invalid_mutator.py  # 规则 3.6: 越界、重复及无效值随机变异修补
│   │
│   ├── operators/                  # 3. 差分演化算子组件
│   │   ├── __init__.py
│   │   ├── crossover.py            # 交叉算子 (动态交叉率 CR 计算，公式 3-9)
│   │   ├── mutation.py             # 混合变异算子 (DE/rand/1 与 DE/best/2 切换，公式 3-10)
│   │   ├── scale_factor.py         # 动态缩放因子 F 调节器 (公式 3-11)
│   │   └── extinction.py           # 种群灭绝/灾变算子 (GMR 机制与精英保留，公式 3-12, 2-37)
│   │
│   └── solvers/                    # 4. 具体算法求解器组合
│       ├── __init__.py
│       ├── dmde_solver.py          # 离散映射差分求解器 (算法 3.1 完整流程)
│       └── baseline_solvers/       # 论文对比算法 (用于第三章表 3-5 实验比对)
│           ├── __init__.py
│           ├── sort_de.py          # 排序差分进化算法 (文献[105])
│           ├── pmx_de.py           # 部分映射交叉差分算法
│           └── cmtap_ga.py         # 协同多目标遗传算法 (文献[65])
```
