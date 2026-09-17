# LLM-DMDE 提示词设计与种群初始化机制分析

> 最后更新：2026-09-17

---

## 一、种群初始化机制分析

### 1.1 核心流程

```
LLM 一次性生成 K 个完整候选解（K = ceil(r × N_pop)）
    ↓
AssignmentConverter 逐个验证 + 转换为 Gene 序列
    ↓
CandidateFilter 质量 + 多样性过滤，选出最终 K 个
    ↓
DMDE 随机初始化补齐 N_pop - K 个
    ↓
合并为初始种群
```

### 1.2 LLM 生成策略：一次生成 vs 分批生成

**当前方案：一次生成 K 个完整解。**

LLM 在一次调用中输出：
```json
{
    "thought": "...",
    "solutions": [
        {"assignments": [{"uav": 0, "targets": [3, 1, 4]}, {"uav": 1, "targets": [2, 0]}]},
        {"assignments": [{"uav": 0, "targets": [2, 4]}, {"uav": 1, "targets": [3, 1, 0]}]},
        ...
    ],
    "reasoning": "..."
}
```

**优点**：
- 一次 API 调用，延迟和成本最低
- LLM 可以在多个候选之间做差异化（thought 中的策略不同）

**风险**：
- K 较大时（如 K=10），单次输出 JSON 很长，容易截断
- LLM 生成多个解时，后面的解质量可能下降（注意力衰减）

**建议**：
- 当前 K = ceil(0.2 × 50) = 10，对于小问题（N=10）可接受
- 如果未来 K > 10 或 N > 30，应考虑分批生成（每次 3-5 个，多次调用）

### 1.3 N<M (SRP) 场景的特殊挑战

SRP 场景下 LLM 需要同时决定：
1. **目标分配**：哪些 Target 分配给哪个 UAV
2. **访问顺序**：每个 UAV 的 Target 巡回顺序

这比 balanced/overloaded 复杂得多——balanced 只需一个排列，SRP 需要一个**分区 + 排序**。

**当前机制的保障**：

| 层级 | 机制 | 作用 |
|---|---|---|
| Prompt | 场景化 system prompt（仅 SRP 规则） | 告诉 LLM 巡回顺序的语义 |
| Prompt | S_problem 含 C_TT + TopKNextTargets | 提供转移代价信息 |
| Prompt | CoT 引导："order by nearest-neighbor" | 引导 LLM 使用贪心排序 |
| 验证 | `_validate_srp()`: targets 长度 ≥ 1 | 每个 UAV 至少访问 1 个目标 |
| 验证 | `_validate_batch_srp()`: T 不重复 | 所有目标恰好访问一次 |
| 验证 | UAV 全覆盖检查 | 每个 UAV 至少出现一次 |
| 修复 | `repair()`: 类型转换 + 范围取模 | 处理 LLM 输出的边界错误 |
| 过滤 | CandidateFilter: fitness 排序 | 淘汰劣解 |
| 兜底 | 3 次重试 + fallback 到标准初始化 | LLM 全部失败时保底 |

**潜在问题**：

1. **巡回顺序的质量无法验证**：LLM 输出的 `[3, 1, 4]` 是否真的是好顺序？
   - 验证只检查格式（不重复、在范围内），不检查代价
   - 但 CandidateFilter 的 fitness 排序会间接淘汰高代价的巡回
   - **结论**：通过 fitness 间接验证，足够

2. **目标分配不均匀**：LLM 可能把大部分目标分配给一个 UAV
   - 验证不检查分配均匀性
   - 但 S_problem 的 TopKTargets 信息会引导 LLM 做合理分配
   - **结论**：依赖 prompt 引导，无硬约束

### 1.4 完整保障机制总结

```
┌─────────────────────────────────────────────────────────────┐
│                    保障机制层次图                              │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Prompt 引导                                         │
│   - 场景化 system prompt（只含当前 model_type 的规则）         │
│   - S_problem（Global Summary + Local Preference）            │
│   - CoT reasoning（difficult → contested → fill → order）    │
│   - Output Validation 指令（格式要求 + 合法性警告）            │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: 格式验证                                            │
│   - 单个 assignment: uav 类型、targets 类型、范围检查          │
│   - batch 验证: UAV 全覆盖、Target 全覆盖、无重复             │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: 转换 + 修复                                         │
│   - 类型强制转换 (int)                                       │
│   - 范围取模 (target % M)                                    │
│   - 空 targets 兜底 ([0])                                    │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: 质量过滤                                            │
│   - fitness 评估 + 排序                                      │
│   - diversity Hamming 距离去重                               │
│   - 贪心选择: 每次选 fitness 最好 + diversity 足够的个体       │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: 兜底机制                                            │
│   - 3 次 LLM 重试（每次重新生成）                             │
│   - 全部失败 → fallback 到标准 DMDE 随机初始化                │
│   - 转换失败的解被跳过，不注入种群                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、提示词设计分析

### 2.1 架构总览

```
prompts/__init__.py
├── System Prompt Registry ──── PROMPT_REGISTRY（通用回退）
├── User Prompt Registry ───── USER_PROMPT_REGISTRY（模板字符串）
├── 场景化 System Prompt ───── _POP_INIT_SCENE_MAP / _SC_SCENE_MAP
│   ├── balanced 版本
│   ├── overloaded 版本
│   └── srp 版本
├── 公共片段 ────────────────── _JSON_FORMAT_COMMON / _REASONING_GUIDE / _OUTPUT_VALIDATION
└── 统一入口 ────────────────── get_prompt(module, prompt_type, prompt_path, model_type, **kwargs)
```

### 2.2 Population Init Prompt 设计

#### System Prompt 结构

```
角色设定（1 行）
    ↓
输出格式（JSON 示例 + thought 字段）
    ↓
场景规则（仅当前 model_type 的规则 + 示例）
    ↓
推理引导（CoT: difficult → contested → fill → order）
    ↓
输出合法性校验（格式要求 + 警告）
    ↓
任务指南（针对当前场景的策略建议）
```

#### User Prompt 结构

```
## Problem (S_problem)
{
    "problem_structure": {N, M, model_type, k},
    "global_summary": {C_UT/C_TT 统计, 可行性统计},
    "local_preference_structure": {
        "TopKTargets_per_UAV": [...],
        "TopKUAVs_per_target": [...],
        "contested_targets": [...],
        "difficult_targets": [...],
        "TopKNextTargets_per_target": [...]  // SRP only
    },
    "cost_matrix": {C_UT, C_TT}  // 小规模时附带
}

## Task
Generate exactly k complete assignment solutions...
```

#### 评估

| 维度 | 评分 | 说明 |
|---|---|---|
| 信息充分性 | ★★★★☆ | Global Summary + Local Preference 覆盖了 LLM 需要的信息 |
| 格式约束 | ★★★★☆ | 三种 model_type 都有明确的 JSON 示例 |
| CoT 引导 | ★★★☆☆ | 有 3 行推理引导，但没有 few-shot example |
| Token 效率 | ★★★★☆ | 场景化 prompt 去掉了无关规则；大问题跳过原始矩阵 |
| 容错性 | ★★★★☆ | Output Validation 指令 + 3 次重试 + fallback |

#### 改进建议

1. **加 1-2 个 few-shot example**（当前只有规则没有完整输入→输出对）
2. **大问题时 TopK 只保留 top-1**（当前 top-3，N=50 时 user prompt 可能过长）
3. **contested/difficult targets 的利用可以更具体**（当前只说"assign first"，没有给具体策略）

### 2.3 Search Controller Prompt 设计

#### System Prompt 结构

```
角色设定 + 任务描述
    ↓
CR 候选值列表（动态注入）
    ↓
CR 机制解释（rand/1 vs best/2 的基因比例关系）
    ↓
场景特定建议（balanced: 中等 CR / overloaded: 低 CR / srp: 高 CR）
    ↓
输出格式（{"cr": 0.5, "reasoning": "..."}）
```

#### User Prompt 结构

```
## Current Search State
{JSON: generation, diversity, stagnation, fitness, ...}

## Recent Trajectory
{最近 5 代的 fitness/diversity/stagnation}

## Task
Select the best CR for the next interval.
```

#### 评估

| 维度 | 评分 | 说明 |
|---|---|---|
| 机制解释 | ★★★★★ | CR 如何影响 rand/1 vs best/2 的基因比例，非常清晰 |
| 场景适配 | ★★★★☆ | 三种场景有不同 CR 建议，但建议较粗 |
| 状态信息 | ★★★★☆ | 14 个特征 + 轨迹，信息充分 |
| 输出格式 | ★★★★★ | 单值 JSON，简单可靠 |
| Token 效率 | ★★★★★ | system prompt ~1.3K chars，user prompt ~500 chars |

#### 改进建议

1. **轨迹信息可以更丰富**（当前只有 5 代的 fitness/diversity/stagnation）
2. **可以加入"上次 CR 选择及其效果"**，让 LLM 做反馈式决策

### 2.4 模块间一致性

| 检查项 | 状态 |
|---|---|
| 两个模块都用 `get_prompt()` 统一接口 | ✅ |
| 都支持 `system_prompt_path` 外部覆盖 | ✅ |
| 都支持 `model_type` 场景化 | ✅ |
| User prompt 都用 `.format()` 模板插值 | ✅ |
| 输出格式都是 JSON | ✅ |
| 都有 `reasoning` 字段 | ✅ |
| PopInit 额外有 `thought` 字段（CoT） | ✅ |

### 2.5 Token 预算估算

| 组件 | balanced | overloaded | srp |
|---|---|---|---|
| System Prompt | ~2.1K chars | ~2.2K chars | ~2.5K chars |
| User Prompt (N=10,M=10) | ~3K chars | ~3K chars | ~4K chars |
| User Prompt (N=50,M=100) | ~8K chars | ~10K chars | ~15K chars |
| **单次调用总计** | **~5K** | **~5K** | **~6.5K** |
| **大问题总计** | **~10K** | **~12K** | **~17.5K** |

大问题（N=50, M=100）时 srp 的 user prompt 可能达到 17.5K chars（~5K tokens），
加上 LLM 输出的 10 个候选解（每个 ~500 chars），总 token 约 8K-10K，在预算范围内。

### 2.6 整体评价

| 维度 | 评分 | 说明 |
|---|---|---|
| 架构设计 | ★★★★☆ | 统一接口 + 注册表 + 场景化 + 模板化 |
| Prompt 内容 | ★★★★☆ | 规则清晰 + CoT 引导 + S_problem 信息充分 |
| 可扩展性 | ★★★★☆ | prompt_path + model_type + 动态参数 |
| Token 效率 | ★★★★☆ | 场景化去噪 + 大问题跳过原始矩阵 |
| 鲁棒性 | ★★★★★ | 验证 + 修复 + 过滤 + 重试 + fallback，五层保障 |
| 一致性 | ★★★★★ | 两个模块的接口/风格/配置完全统一 |

### 2.7 待改进项（按优先级）

| 优先级 | 改进项 | 工作量 |
|---|---|---|
| P1 | PopInit prompt 加 1-2 个 few-shot example | 小 |
| P1 | 大问题时 TopK 自适应缩减（top-1 或 top-2） | 小 |
| P2 | Search Controller 加"上次 CR 效果反馈" | 中 |
| P2 | PopInit 的 contested/difficult 策略更具体 | 小 |
| P3 | 分批生成机制（K > 10 时每次 3-5 个） | 中 |