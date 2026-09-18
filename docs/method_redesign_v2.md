# LLM-Enhanced DMDE 方法重设计 v2

> 日期：2026-09-18
> 状态：设计定稿
> 基于：消融实验分析 + 代码审查 + prompt 改进迭代

---

## 1. 问题诊断：v1 方法的核心缺陷

### 1.1 LLM 做全局 assignment 决策的结构性困难

v1 方法让 LLM 一次性输出完整 assignment solutions（N×M 个配对），实验数据表明：

| 场景 | LLM 输出 | 可行解率 | 根因 |
|------|---------|---------|------|
| balanced N=M=10 | 10 solutions | **0/10 (0%)** | 排列约束无法满足 |
| srp N=10 M=20 | 10 solutions | **0/10 (0%)** | tour ordering + 覆盖约束 |

LLM 面临三重困难：
1. **协调困难**：同时决定 N×M 个配对，组合空间爆炸
2. **可行性不可保证**：10/10 的解都不可行，依赖程序修复
3. **信号被稀释**：修复后的解质量 ≈ 随机初始化，消融实验不显著

### 1.2 SRP 约束计算 Bug（已修复）

`check_time_window_constraint` 对 SRP tour genes 使用 `C_UT`（UAV 直飞代价）而非 `C_TT`（目标间转移代价），导致 fitness 评估存在约束惩罚噪声。S2 SRP 场景 1000 代 best_fitness 完全不变。

### 1.3 方法论层面的根本问题

v1 的方法论定位是：
> LLM generates candidate solutions → Program validates/repairs → Inject into population

这要求 LLM 做它不擅长的事（全局组合优化），而程序只是"善后"。

---

## 2. v2 方法设计：职责分离

### 2.1 核心原则

**LLM 不碰 feasibility，不直接决定 assignment，只提供搜索先验和搜索控制。**

| 职责 | 承担者 | 理由 |
|------|--------|------|
| 可行性保证 | 确定性机制 | LLM 输出 0% 可行率，程序侧保证 |
| 目标评估 | 数学模型 | fitness 函数是确定性的 |
| 种群进化 | DMDE | 差分进化天然产生多样性 |
| 多样性 | 进化机制本身 | 不需要 LLM 额外控制 |
| 搜索先验 | **LLM** | LLM 擅长结构化信息提取 |
| 自适应搜索强度 | **LLM-CR** | 窄决策 + 闭环反馈 |

### 2.2 架构图

```
                 Multi-UAV Multi-Target Assignment
                              │
                              ▼
                    Structured Instance
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          LLM Structural Prior       Mathematical Indicators
          (搜索先验信息)              (确定性数学指标)
                 │                         │
                 └────────────┬────────────┘
                              ▼
                   Feasible Population
                   (确定性构造 + 可行性保证)
                              │
                              ▼
                             DMDE
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              Evolutionary          LLM CR Control
                Search              (闭环反馈控制)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         Pareto Solutions
```

---

## 3. 模块详细设计

### 3.1 LLM Structural Prior（搜索先验模块）

**职责**：从结构化问题表示中提取搜索先验信息，不输出完整解。

**输入**：
- 结构化问题表示 S_problem（代价矩阵、约束、偏好）
- 数学指标（TopK、contested/difficult targets）

**输出**（JSON）：
```json
{
  "analysis": {
    "difficult_targets": [
      {"target": 9, "reason": "only 2 feasible UAVs, highest cost"},
      {"target": 4, "reason": "only 3 feasible UAVs"}
    ],
    "contested_assignments": [
      {"target": 5, "recommended_uav": 4, "reason": "lowest cost 9560, 8 UAVs prefer it"},
      {"target": 0, "recommended_uav": 6, "reason": "lowest cost 12732"}
    ],
    "uav_groupings": {
      "srp_only": [
        {"uavs": [0, 3, 6], "targets": [2, 10, 16], "reason": "spatial proximity"},
        {"uavs": [1, 4, 7], "targets": [5, 15, 18], "reason": "contested region"}
      ]
    }
  },
  "reasoning": "..."
}
```

**关键设计**：
- LLM 只输出**建议**，不输出完整 assignment
- 每个建议有明确的 reason（可审计、可验证）
- 输出量远小于完整解（几个建议 vs N×M 个配对）

### 3.2 Mathematical Indicators（确定性数学指标）

**职责**：从代价矩阵中计算确定性指标，不经过 LLM。

**指标列表**（当前已实现，直接复用）：
- `TopKTargets_per_UAV`：每 UAV 的 top-K 最小代价目标
- `TopKUAVs_per_target`：每目标的 top-K 最小代价 UAV
- `contested_targets`：被多个 UAV 偏好的高竞争目标
- `difficult_targets`：可行 UAV 很少的目标
- `cost_statistics`：代价矩阵统计（min/max/mean/std）
- `feasibility_statistics`：可行性统计

### 3.3 Feasible Population（确定性构造模块）

**职责**：基于 LLM Prior + 数学指标，用确定性算法构造可行初始种群。

**三阶段构造策略**：

```
Phase 1: Prior-Guided Greedy（1 个高质量解）
  ├── 锁定 difficult targets → 最佳 UAV
  ├── 锁定 contested assignments → 按 LLM 建议或数学指标
  └── 贪心补齐剩余分配

Phase 2: Prior-Guided Perturbation（K-1 个多样化解）
  ├── 基于 Phase 1 解，swap 2-3 对 assignment
  ├── 重新分配 contested targets（不同 UAV）
  └── SRP: 2-opt tour perturbation

Phase 3: Random Initialization（剩余补齐）
  └── 标准 DMDE 随机初始化（PopulationEncoder.generate）
```

**关键设计**：
- 构造算法是确定性的（相同输入 → 相同输出）
- 可行性由构造算法保证（不需要事后修复）
- Phase 1-2 利用 LLM Prior，Phase 3 是纯随机

### 3.4 LLM CR Control（闭环搜索控制）

**职责**：根据搜索状态动态调整交叉率 CR。

**设计**（与 v1 一致，无需改动）：
- 输入：Δf, ΔD, stagnation, stage_history
- 输出：CR 值（标量决策）
- 闭环：每次决策后观察 outcome，调整下一步

### 3.5 Quality-Diversity Selection

**在初始化阶段**：Phase 1-2 构造的解天然具有多样性（不同 Prior + 不同扰动），不需要额外的 Q-D 过滤。

**在进化阶段**：DMDE 的差分变异（rand/1）天然产生多样性，不需要额外机制。

---

## 4. 与 v1 方法的对比

| 维度 | v1 | v2 |
|------|----|----|
| LLM 输出 | 完整 assignment solutions | 结构化 Prior（建议） |
| LLM 决策量 | N×M 个配对 | 几个结构化建议 |
| 可行性保证 | 程序修复（事后） | 确定性构造（事前） |
| 可行解率 | 0%（修复后≈随机） | 100%（构造保证） |
| 多样性来源 | CandidateFilter | 进化机制本身 |
| 数学指标 | 嵌入 LLM prompt | 独立输入，不经过 LLM |
| LLM 角色 | 解的构造者 | 信息的提供者 |
| 规模扩展 | 受限于 LLM 组合推理能力 | 不受限（LLM 只做信息提取） |

---

## 5. 方法论表述

### v1（当前）
> LLM-guided population initialization with structure-aware candidate construction and deterministic quality control.

### v2（重设计）
> LLM-guided evolutionary optimization with structural prior extraction and deterministic feasibility assurance.
>
> Core mechanism:
> **Structured Problem Representation → LLM Structural Prior Extraction → Deterministic Feasible Construction → Evolutionary Search with Adaptive LLM-CR Control**

---

## 6. 实现计划

### Phase 1: Prior 模块（核心）
- [ ] 重写 `population_init.py`：LLM 输出 Prior 而非完整解
- [ ] 重写 prompt：引导 LLM 输出结构化建议
- [ ] 实现 Prior 解析器：提取 difficult/contested/grouping

### Phase 2: 构造模块
- [ ] 实现 `PriorGuidedConstructor`：三阶段构造算法
- [ ] 集成 Mathematical Indicators（复用现有代码）
- [ ] 可行性验证（单元测试）

### Phase 3: 消融实验
- [ ] A0: Vanilla DMDE（基线）
- [ ] A1: DMDE + Math-Only Constructor（无 LLM）
- [ ] A2: DMDE + LLM Prior Constructor
- [ ] A3: DMDE + LLM Prior + CR Control（Full）
- [ ] 运行 S1 balanced + S2 srp 场景

### Phase 4: 论文
- [ ] 方法论章节重写
- [ ] 实验结果更新
- [ ] 与 v1 对比分析

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM Prior 信息量不足 | A1 ≈ A2，Prior 无额外价值 | 对比实验验证；增强 prompt |
| 确定性构造算法过贪心 | 初始种群过快收敛 | Phase 2 加入随机扰动 |
| LLM Prior 误导 | Prior 错误导致构造解质量下降 | 数学指标作为 fallback |
| 规模扩展后 Prior 质量下降 | N=20+ 时 LLM 信息提取能力下降 | 稀疏表示 + 分层 Prior |