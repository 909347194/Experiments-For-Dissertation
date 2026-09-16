# LLM-DMDE 实验方案

> 最后更新：2026-09-16

---

## 一、实验总体设计

```
实验一：消融实验（Ablation Study）
    → 证明每个 LLM 模块的独立贡献 + 双模块协同效应

实验二：性能对比实验（Comparative Experiment）
    → 与基线算法对比，证明 LLM-DMDE 的整体优势

实验三：超参数敏感性分析（Sensitivity Analysis）
    → 分析关键超参数对性能的影响
```

---

## 二、实验一：消融实验

### 2.1 消融配置

| 编号 | 配置名 | PopInit | CR Control | 目的 |
|---|---|---|---|---|
| **A0** | Vanilla DMDE | ✗ | ✗ | 基线 |
| **A1** | DMDE + CR Control | ✗ | ✓ | 仅 CR 模块贡献 |
| **A2** | DMDE + PopInit | ✓ | ✗ | 仅初始化模块贡献 |
| **A3** | Full LLM-DMDE | ✓ | ✓ | 双模块协同 |

### 2.2 实验场景

消融实验**不需要全覆盖**所有 assignment model。核心目的是证明模块有效，挑 1-2 个代表性场景即可。

选择原则：
- 一个**最简单**的场景（balanced），排除模型复杂度的干扰
- 一个**PopInit 最复杂**的场景（srp），验证 LLM 在高复杂度下仍然有效

| 场景 | Model | N | M | 选择理由 |
|---|---|---|---|---|
| **S1** | balanced (N=M) | 10 | 10 | 小规模基准，PopInit 最简单 |
| **S2** | srp (N<M) | 10 | 20 | PopInit 需同时决定分配 + 巡回顺序 |

4 配置 × 2 场景 = **8 组实验**

### 2.3 实验设置

| 参数 | 值 | 说明 |
|---|---|---|
| 独立运行次数 | 30 | 保证统计显著性 |
| 随机种子 | 42, 43, ..., 71 | 可复现 |
| 最大迭代代数 | 1000 | 统一终止条件 |
| 种群大小 | 50 | 默认 |
| LLM 注入比例 r | 0.2 | PopInit 默认 |
| CR 触发间隔 p | 50 | CR Control 默认 |
| ζ (zeta) | 3 | DMDE 默认 |
| δ (delta) | 0.3 | 灭绝阈值 |
| LLM 模型 | deepseek-flash | 统一 LLM 后端 |

### 2.4 评估指标

| 指标 | 含义 | 计算方式 |
|---|---|---|
| **Best Fitness** | 最优解质量 | 30 次运行的最优适应度 |
| **Mean ± Std** | 平均解质量 + 离散程度 | 30 次运行的均值 ± 标准差 |
| **Median** | 鲁棒性 | 30 次运行的中位数 |
| **Convergence Gen** | 收敛速度 | 达到 95% 最终最优解所需的代数 |
| **Success Rate** | 成功率 | 适应度低于阈值的运行比例 |
| **Wilcoxon p-value** | 统计显著性 | A3 vs A0/A1/A2 的双侧秩和检验 |

#### 计算时间统计（分口径）

| 口径 | 包含内容 | 用途 |
|---|---|---|
| **Total Wall Time** | 全部耗时 | 衡量实际总耗时 |
| **DMDE Compute Time** | 仅进化部分 | 衡量算法本身计算开销 |
| **LLM Time** | LLM 调用 + 等待 | 衡量 LLM 引入的额外开销 |
| **LLM Init Time** | PopInit 模块耗时 | 分解 PopInit 代价 |
| **LLM CR Time** | CR Control 模块耗时 | 分解 CR Control 代价 |
| **LLM Call Count** | LLM 调用次数 | 评估 API 成本 |

时间统计代码框架：

```python
t_total_start = time.time()

# LLM PopInit
t_llm_init_start = time.time()
# ... LLM 调用 ...
t_llm_init = time.time() - t_llm_init_start

t_llm_cr_total = 0.0
llm_cr_call_count = 0

# DMDE 进化循环
for gen in range(max_gen):
    # LLM CR Control（每 p 代）
    if gen % p == 0:
        t_llm_cr_start = time.time()
        # ... LLM 调用 ...
        t_llm_cr_total += time.time() - t_llm_cr_start
        llm_cr_call_count += 1

    # DMDE 进化步骤
    # ... 进化 ...

t_total = time.time() - t_total_start
t_llm_total = t_llm_init + t_llm_cr_total

# 结果
{
    "total_time": t_total,
    "llm_time": t_llm_total,
    "llm_init_time": t_llm_init,
    "llm_cr_time": t_llm_cr_total,
    "llm_call_count": llm_cr_call_count,
    "avg_llm_latency": t_llm_total / max(llm_cr_call_count, 1),
}
```

### 2.5 结果表格模板

#### 表 1：消融实验结果

| 场景 | 配置 | Best | Mean ± Std | Median | Conv.Gen | Succ.Rate | 总耗时(s) | LLM耗时(s) |
|---|---|---|---|---|---|---|---|---|
| S1 balanced | A0 DMDE | | | | | | | — |
| S1 balanced | A1 +CR | | | | | | | |
| S1 balanced | A2 +PopInit | | | | | | | |
| S1 balanced | A3 Full | | | | | | | |
| S2 srp | A0 DMDE | | | | | | | — |
| S2 srp | A1 +CR | | | | | | | |
| S2 srp | A2 +PopInit | | | | | | | |
| S2 srp | A3 Full | | | | | | | |

#### 表 2：统计显著性（Wilcoxon p-value）

| 场景 | A3 vs A0 | A3 vs A1 | A3 vs A2 | A1 vs A0 | A2 vs A0 |
|---|---|---|---|---|---|
| S1 balanced | | | | | |
| S2 srp | | | | | |

p < 0.05 标记为显著（*），p < 0.01 标记为极显著（**）。

### 2.6 预期结论

| 对比 | 预期 | 解释 |
|---|---|---|
| A1 vs A0 | A1 更优 | LLM 的 CR 自适应 > 固定公式调度 |
| A2 vs A0 | A2 收敛更快 | 知识驱动初始化 > 纯随机初始化 |
| A3 vs A1 | A3 ≥ A1 | PopInit 提供更好的起点 |
| A3 vs A2 | A3 ≥ A2 | CR Control 在进化中持续优化 |
| A3 vs A0 | A3 显著优于 A0 | 双模块协同增益 |

**关键验证**：A3 > max(A1, A2)，证明两个模块互补而非冗余。

### 2.7 可视化

| 图表 | 内容 | 用途 |
|---|---|---|
| 收敛曲线 | 每个场景 4 条曲线（A0-A3） | 直观展示收敛差异 |
| 箱线图 | 每个场景 4 个配置的解质量分布 | 展示鲁棒性和离群值 |
| 时间堆叠柱状图 | DMDE 时间 + LLM Init 时间 + LLM CR 时间 | 展示时间开销构成 |

---

## 三、实验二：性能对比实验

### 3.1 对比算法

| 算法 | 说明 |
|---|---|
| Vanilla DMDE | 基线 |
| GA (遗传算法) | 经典进化算法 |
| PSO (粒子群优化) | 经典群智能算法 |
| CMTAP-GA | 项目内置基线求解器 |
| PMX-DE | 项目内置基线求解器 |
| SORT-DE | 项目内置基线求解器 |
| **LLM-DMDE (ours)** | 本文方法 |

### 3.2 实验场景（全覆盖）

| 场景 | Model | N | M | 特点 |
|---|---|---|---|---|
| S1 | balanced | 10 | 10 | 小规模 |
| S2 | balanced | 30 | 30 | 中规模 |
| S3 | overloaded | 20 | 10 | UAV 冗余 |
| S4 | overloaded | 50 | 20 | 大规模冗余 |
| S5 | srp | 10 | 20 | 目标多于 UAV |
| S6 | srp | 15 | 40 | 大规模巡游 |

7 算法 × 6 场景 × 30 runs = **1260 runs**

### 3.3 评估指标

与消融实验相同，额外增加：
- **排名（Rank）**：每个场景下各算法的排名
- **相对提升（Improvement%）**：LLM-DMDE 相对最优基线的提升百分比

---

## 四、实验三：超参数敏感性分析

在 S1 (balanced N=10) 和 S5 (srp N=10,M=20) 上做。

### 4.1 PopInit 注入比例 r

| r | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|

其他参数固定，每个 r 值 30 runs。

目标：找到最优注入区间，验证"少量高质量候选 > 大量随机候选"。

### 4.2 CR Control 触发间隔 p

| p | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|

目标：验证过频调用（高 API 成本）vs 过疏调用（失去自适应性）的 trade-off。

### 4.3 Top-k 偏好统计

| top_k | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|

目标：验证 S_problem 信息量对 LLM 生成质量的影响。

### 4.4 敏感性分析总 runs

5 值 × 3 参数 × 2 场景 × 30 runs = **900 runs**

---

## 五、实验总工作量

| 实验 | Runs | 估算时间 |
|---|---|---|
| 消融实验 | 8 × 30 = 240 | |
| 性能对比 | 42 × 30 = 1260 | |
| 超参数敏感性 | 30 × 30 = 900 | |
| **合计** | **2400 runs** | |

---

## 六、执行顺序

```
Phase 1: 消融实验（先做，确认模块有效）
    → S1 + S2 × 4 配置 × 30 runs = 240 runs

Phase 2: 性能对比实验（全场景）
    → 6 场景 × 7 算法 × 30 runs = 1260 runs

Phase 3: 超参数敏感性（最后做，用于论文讨论部分）
    → 3 参数 × 5 值 × 2 场景 × 30 runs = 900 runs

Phase 4: 统计分析 + 可视化 + 论文撰写
```

---

## 七、关键注意事项

1. **所有实验必须使用相同随机种子序列**，保证公平对比
2. **LLM 调用需要记录原始输入/输出**，便于事后分析 LLM 决策质量
3. **时间统计必须分口径**（Total / DMDE / LLM），诚实报告 LLM 开销
4. **统计检验必须做**，不能只看均值差异
5. **消融实验场景不求全**，1-2 个代表性场景足够；全场景留给性能对比