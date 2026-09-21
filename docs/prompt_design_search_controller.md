# LLM-DMDE Search Controller 提示词设计与机制分析

> 本文替代已删除的 `prompt_design_analysis.md`。
> 原文档同时覆盖"种群初始化（PopInit）"与"搜索控制（Search Controller）"两个模块；
> PopInit 已从代码库中移除，本文只描述**唯一保留的模块：Search Controller**。

---

## 一、模块定位

Search Controller 在 `before_mutation` 钩子处介入，即**每轮变异之前**决定是否调整
DE 的三个搜索机制参数：

| 通道 | 符号 | 原 DMDE 中的来源 | 解耦后 |
|------|------|------------------|--------|
| 交叉概率 | CR | 公式 3-9（由代数推导） | LLM 从候选集选择 / 保持 |
| 缩放因子 | F | 公式 3-11（由 CR 推导） | LLM 从候选集选择 / 保持 |
| 灭绝机制 | GMR | 公式 3-12（由 CR 推导） | LLM 选 `auto` / `on` / `off` |

核心主张：**原 DMDE 用手工方程把 CR、F、GMR 绑在一条链上**（CR → F → GMR），
三者只能协同移动。解耦之后，LLM 可以依据观测到的进化状态与轨迹**独立**调整它们。

---

## 二、三条件动作空间对照

为了隔离"LLM 能控制哪些通道"这一变量，同一套观测接口被复用到三种动作空间下。
三者共用**完全相同的观测集、触发机制与冻结守卫**，唯一差别是"能动什么"：

| 条件 | 目录后缀 | 动作空间 | CR 取值 | 注入的 prompt 段 |
|------|----------|----------|---------|------------------|
| 解耦 | `A1_cr_control` | CR + F + GMR | LLM 决策 | `_SC_ACTIONS_DECOUPLED` + `_SC_PARAM_EFFECT_DECOUPLED` + `_SC_DECISION_DECOUPLED` |
| 耦合 | `A1_coupled` | 仅 CR | LLM 决策（F/GMR 由公式跟随） | `_SC_ACTIONS_COUPLED` + `_SC_PARAM_EFFECT_COUPLED` + `_SC_DECISION_COUPLED` |
| 去 CR | `A1_nocr` | F + GMR | 锁定离线选定值（0.3） | `_SC_ACTIONS_NOCR` + `_SC_PARAM_EFFECT_NOCR` + `_SC_DECISION_NOCR` |

**关键设计约束**：模型不能"看到却动不了"。
因此耦合模式下 **不注入** `f_choices`，也不注入 GMR/F 的冻结段
（模型本没有这两个通道，注入会自相矛盾）；`no_cr` 模式下同理跳过 CR 冻结段。

---

## 三、Prompt 结构

### 3.1 System Prompt 组装链

```
_SC_BASE                      # 角色 + 参考量级阈值（从 solver 注入，权威）
  ├─ {actions_block}          # 按条件三选一：能控制哪些通道
  ├─ {param_effect_block}     # 参数效应说明（与动作空间一致）
  └─ {decision_block}         # 决策策略（默认 hold，先问"要不要改"再问"改成什么"）
      ↓
_SC_SHADOW                    # 可选：影子对照（shadow_cr 非 None 时注入）
      ↓
_SC_FROZEN                    # 可选：CR 通道冻结（cr_frozen 且非 no_cr）
_SC_FROZEN_GMR                # 可选：GMR 冻结（gmr_frozen 且非 coupled）
_SC_FROZEN_F                  # 可选：F 冻结（f_frozen 且非 coupled）
      ↓
_SC_SIGNAL_GUIDE              # 读数口径指引
      ↓
_SC_SCENE_MAP[model_type]     # 场景段：balanced / overloaded / srp
      ↓
_SC_FORMAT*                   # 输出格式（按条件三选一，含冻结时的强制取值）
```

当前 `get_search_controller_prompt(...)` 产出的 system prompt 长度约 **10.6K 字符**。

### 3.2 参考量级阈值从 solver 注入

`_SC_BASE` 中的三个噪声阈值**必须由 solver 配置传入**，不允许在 prompt 里写死：

```python
df_noise_pct     # Δf 噪声底（%），与 solver freeze.df_noise 同源
dd_noise         # ΔD 噪声底，与 solver trigger.dd_threshold 同源
shadow_contrast  # 影子归因阈值(%)，与 solver freeze.shadow_contrast 同源
```

> **历史教训**：早期版本 prompt 写死 `0.02` 而 solver 用 `0.05`，导致模型的
> "读数标准"与采样器的"触发标准"不一致——模型会把噪声当成信号。现已统一。

### 3.3 User Prompt

```
## Current Search State
{state_json}                  # 见 §4 观测特征

## Stage History (each row = one of your previous decisions)
{trajectory_text}             # 最近 5 个 stage 的表格

## Task
Decide CR, F, and GMR for the next stage. ...
```

---

## 四、观测特征（S_t）

`search_controller.build_prompt()` 注入的特征分五组：

**1. 约束可行性读数**（原属观测缺口，后补入）
`feasible_ratio`、`violation_mean`、`violation_max`
——GMR 重启与 F 步长是决定可行性的主要杠杆，不给出这些读数模型无法归因。

**2. 多样性水平与结构**
`diversity`、`diversity_p25`、`diversity_p75`、`delta_diversity`

**3. 趋势与阶段统计**
`delta_fitness_pct`、`stage_length`、`acceptance_rate`、
`gens_since_last_improvement`、`stagnation_raw`（真实值，未封顶）

**4. 上下文与自检**
`generation`、`max_generations`、`model_type`、`prev_action_cr`、
`trigger_reason`、`stage_best_curve`（降采样逐代 best 曲线）、
四个冻结标志位及其原因（`cr_frozen` / `gmr_frozen` / `f_frozen` + `*_reason`）、
`current_f`、`current_gmr_mode`（状态观测，耦合模式下用于预判连带效应）

**5. 影子对照（可归因锚点）**
`shadow_control`: `shadow_cr`、`shadow_delta_fitness_pct`、`shadow_delta_diversity`
——同起点、固定 CR 的影子种群，在同 stage 窗口上的增量。

### Stage History 表格

```
Stage | Len | CR | F | GMR | Best Fitness | df(%) | df_shadow(%) | dD | Accept%
```

保留最近 5 行。表中已含 `df / dD / df_shadow`，因此正文不再重复压缩这些量，
避免同一数字以两种精度出现引发的矛盾读数。

> 注意：`delta_fitness_pct` **未按 stage_length 归一**。长 stage 的 df 天然更大，
> 跨 stage 比较必须先除以 `Len` 列。这条已显式写入 prompt。

---

## 五、决策协议

1. **默认 hold**：对 CR 和 F，先回答"要不要改"，只有答"要"才回答"改成什么"。
   目的是抑制无意义抖动。
2. **被咨询 ≠ 有证据**：控制器也会因 fallback 定时器触发咨询，prompt 明确声明
   "being consulted is not by itself evidence"。
3. **影子归因门**：`|df - df_shadow|` 必须超过 `shadow_contrast` 才能声称
   自己的选择与固定 CR 基线不同。
4. **冻结守卫**：
   - CR 无证据守卫（连续多轮 df 低于噪声底 → `cr_frozen`）
   - GMR / F 连续无效动作守卫 → `gmr_frozen` / `f_frozen`
   冻结时输出格式被强制为 `null` / `"auto"` / `"hold"`，并在 prompt 中说明原因。

---

## 六、实测结论：为什么"输出参数"在本任务上打不过基线

这是本项目最重要的**负面结果**，建议按此定位撰写论文。

### 6.1 现象

在 S1（balanced N=M=10）与 S2（srp N=10, M=20）上，
**离线固定 CR = 0.3 在 9/9 组对照中优于所有在线配置**（含解耦、耦合、去 CR）。

决策日志的典型指纹：

| 观测 | 值 | 含义 |
|------|-----|------|
| `corr(CR_t, CR_{t-1})` | −0.967 | 模型在相邻两步间**反向翻转** |
| CR 翻转率 | 0.98 | 几乎每一步都在改 |
| 被采纳的观测数 | 0 / 22 | 22 个观测特征一条都没真正影响决策 |
| CR 设定次数 vs 增益 Spearman ρ | +0.926 | **改得越多，结果越差** |

### 6.2 四类机制性解释

**(a) 先验可得性**
算子/策略选择是**语义分类**任务——预训练语料里"停滞时该加大扰动"这类知识大量存在，
模型可以直接召回。参数控制则是**数值标定**任务——"CR 该取 0.37 还是 0.41"
没有可召回的先验，模型只能现猜。

**(b) 动作粒度与信噪比**
CR 从 0.1 到 0.9 分 5 档，档间真实效应约 **−0.248 pp**，
而单次运行的噪声标准差是 **3.513 pp**。信噪比约 **1/14**。
模型在拟合噪声，且它每 50 代只有 10 次采样机会。

**(c) credit assignment 让任务更"数值"**
我们为了帮模型归因，加入了 stage 历史、影子对照、接受率等特征。
但这些特征全是数字，反而把任务推离了模型擅长的语义侧。

**(d) 余量本身极小**
S2 的理论余量 −3.99%、S1 仅 −0.02%，
而常数 CR=0.3 已经吃掉了 **−3.98%**。
任务是**静态**的（代价矩阵固定），不存在时变结构可供在线策略剥削。

### 6.3 与 Zhang et al. 2025 的对照

Zhang 2025（LLM-MOEA）的 Table 4 恰好自带这条对照：

| 方法 | 动作形式 | HV |
|------|----------|-----|
| Random | — | 155.59 |
| DDQN | 算子选择 | 176.81 |
| **LLM-SSA** | **参数控制** | **205.26** |
| **LLM-MOEA** | **算子选择** | **232.84** |

即：**同一批人、同一套框架、同一个 LLM，参数控制比算子选择低 12.7% HV。**
我们的数据独立复现了这一差距。所以这不是"prompt 没写好"，而是**任务形式**问题。

### 6.4 三条可选的重设计路线

| 路线 | 动作形式 | 评价 |
|------|----------|------|
| A. 语义策略层选择 | LLM 输出策略档位（explore / exploit / restart），由确定性映射表落到 CR/F/GMR | 与模型先验对齐，且 credit assignment 落在离散语义档上 |
| B. 阶段标签 + bandit | LLM 只输出"当前处于哪个搜索阶段"，由 bandit 做参数层的信用分配 | 数值标定交给统计可靠的组件 |
| C. 作为负面结果发表 | 保留现有设计，明确报告"在本任务上参数控制劣于常数基线"及四项机制分析 | 对社区有独立价值，工作量最小 |

---

## 七、Token 预算

| 组件 | 规模 |
|------|------|
| System prompt（含场景段与格式段） | ~10.6K chars ≈ 3K tokens |
| User prompt（状态 JSON + 5 行 stage 表） | ~1.5K chars ≈ 0.5K tokens |
| 单次调用合计 | ≈ 3.5K tokens |
| 每 run 调用次数 | MAX_GENERATIONS / interval = 500 / 50 = 10 |
| 每 run 合计 | ≈ 35K tokens |

按 SiliconFlow + Qwen3.5-9B 计价（输入/输出 $0.10 / $0.15 per M tokens），
120 组实验成本约 **$0.25 – $0.5**。

---

## 八、鲁棒性保障（五层）

| 层 | 机制 |
|----|------|
| 1. 输出验证 | JSON 解析 + 字段类型/取值范围校验 |
| 2. 取值修复 | 越界值裁剪到候选集内最近合法值 |
| 3. 通道冻结 | 连续无效动作后强制 hold / auto |
| 4. 失败重试 | 解析失败时重试一次 |
| 5. Fallback | 仍失败则沿用 solver 公式（等价于 A0 行为） |

---

## 九、待改进项

| 优先级 | 改进项 | 工作量 |
|--------|--------|--------|
| P0 | 按 §6.4 路线 A/B 重设计动作空间（或明确定位为负面结果） | 中 |
| P1 | Search Controller 加"上次决策及其效果"的显式反馈闭环 | 中 |
| P2 | `stage_best_curve` 的自适应降采样（长 stage 时信息密度不足） | 小 |
| P2 | 场景段的 CR 建议目前较粗（高/中/低三档），可细化到与候选集对齐 | 小 |
