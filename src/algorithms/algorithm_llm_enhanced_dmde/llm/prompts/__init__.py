# -*- coding: utf-8 -*-
"""LLM 提示词统一管理模块

职责：
    集中管理所有 LLM 模块的提示词模板，支持：
    1. 默认提示词（Python 模块定义）
    2. 外部文件覆盖（实验配置目录）
    3. 场景化提示词（N=M/N>M/N<M）— 根据 N,M 关系自动选择
    4. 动态参数插值（.format() 风格）
    5. User Prompt 模板化

使用方式：
    from llm.prompts import get_prompt

    # 获取 system prompt（通用）
    sys_prompt = get_prompt("population_init")

    # 获取场景化 system prompt（根据 N,M 自动选择 balanced/overloaded/srp）
    sys_prompt = get_prompt("population_init", model_type="srp")

    # 获取带参数的 system prompt
    sys_prompt = get_prompt("search_controller", cr_choices=[0.1, 0.3, 0.5])

    # 从外部文件加载 system prompt
    sys_prompt = get_prompt("population_init", prompt_path="/path/to/prompt.txt")

    # 获取 user prompt 模板（支持动态参数）
    user_prompt = get_prompt("population_init", prompt_type="user",
                             problem_json="...", k=5, n_uavs=10, model_type="balanced")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 公共片段 ─────────────────────────────────────────────────

_JSON_FORMAT_COMMON = """\
## Output Format
Briefly consider the relevant problem characteristics before generating the solutions. \\\nRecord a concise rationale in the "thought" field, then output the solutions.
Respond with a JSON object only (no markdown):
"""

_REASONING_GUIDE = """\
## Reasoning (in the "thought" field)
Before generating solutions, reason through the problem structure:
1. Which targets are **difficult** (few feasible UAVs)? Assign these first.
2. Which targets are **contested** (many UAVs prefer them)? Decide allocation early.
3. What is the approximate cost lower bound? Use it to gauge solution quality.
4. For SRP: how should targets be partitioned among UAVs, and what tour order minimizes transition costs?
"""

_OUTPUT_VALIDATION = """\
## Output Validation
- Each solution must contain exactly N assignments (one per UAV).
- UAV IDs must be in [0, N-1], target IDs in [0, M-1].
- **VERIFICATION STEP**: Before outputting, for each solution check:
  1. Count total targets — must equal M (balanced/overloaded: each target ≥ 1 time; SRP: exactly 1 time).
  2. Check constraint compliance for the specific model type.
  3. Check all targets in [0, M-1] are covered.
  If any check fails, fix the solution before outputting.
- Invalid outputs will be rejected. Double-check before responding.

## Output Quality
- Focus on generating HIGH-QUALITY solutions with low total cost.
- The "thought" field should explain your construction strategy.
- Output ONLY the JSON object. No markdown fences, no preamble.
"""


# =============================================================================
# Population Init — Scene-specific System Prompts
# =============================================================================

_POP_INIT_BALANCED = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a **balanced** UAV-target problem \
(N == M, one-to-one mapping). The solver will convert your assignments into an \
evolutionary algorithm's internal encoding, so you only need to produce \
**discrete assignments**.

{_json_format}
{{
    "thought": "<your construction strategy>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": 0, "targets": [2]}},
                {{"uav": 1, "targets": [0]}},
                {{"uav": 2, "targets": [1]}}
            ]
        }}
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete one-to-one assignment** covering ALL N UAVs.

## Rules (balanced: N == M)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once, each target appears exactly once.
- Each "targets" list has exactly 1 element.
- The set of all target IDs in a solution must be exactly [0, 1, ..., M-1].

{_reasoning}

## Construction Strategy
Use the following two-phase approach to generate high-quality, diverse solutions:

### Phase 1: Greedy Baseline (solution 1)
1. Identify **contested targets** (preferred by many UAVs). For each, assign the \
   lowest-cost UAV. If a UAV is already taken, use the next cheapest.
2. For remaining unassigned UAVs, greedily assign to the cheapest available target.
3. Compute total cost = sum(C_UT[uav][target]). This is your quality reference.

### Phase 2: Perturbations (solutions 2-K)
Generate variants by perturbing the baseline:
- **Swap**: exchange 2-3 UAV-target pairs and check if cost improves.
- **Contested reassignment**: pick one contested target, assign to a different UAV.
- **Greedy restart**: start the greedy construction from a different contested target.

{_validation}

## Guidelines
- Minimize total assignment cost = sum of C_UT[uav][target] for all assignments.
- Use TopKTargets_per_UAV and TopKUAVs_per_target to identify low-cost pairings.
- Pay attention to contested_targets — assigning them optimally is often the key differentiator.
- If a target has only 1-2 feasible UAVs (difficult_targets), lock those assignments early.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
)

_POP_INIT_OVERLOADED = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for an **overloaded** UAV-target problem \
(N > M, more UAVs than targets). The solver will convert your assignments into an \
evolutionary algorithm's internal encoding, so you only need to produce \
**discrete assignments**.

{_json_format}
{{
    "thought": "<your construction strategy>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": 0, "targets": [1]}},
                {{"uav": 1, "targets": [0]}},
                {{"uav": 2, "targets": [1]}}
            ]
        }}
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete assignment** covering ALL N UAVs.

## Rules (overloaded: N > M)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target **must appear at least once** across all assignments.
- Each "targets" list has exactly 1 element.
- Multiple UAVs may be assigned to the same target.

{_reasoning}

## Construction Strategy
### Phase 1: Coverage-First Baseline (solution 1)
1. Identify **difficult targets** (few feasible UAVs). Assign their best UAV first.
2. For each remaining uncovered target, assign the cheapest available UAV.
3. Distribute remaining surplus UAVs to their lowest-cost targets.
4. Compute total cost = sum(C_UT[uav][target]). This is your quality reference.

### Phase 2: Perturbations (solutions 2-K)
- **Surplus redistribution**: move 1-2 surplus UAVs to different targets.
- **Coverage swap**: swap assignments between two UAVs, maintain coverage.
- **Greedy restart**: prioritize a different target for the initial assignment.

{_validation}

## Guidelines
- Minimize total assignment cost.
- **Critical**: ensure every target is covered by at least one UAV.
- Use TopKUAVs_per_target to find the best UAV for each target.
- Use TopKTargets_per_UAV to distribute surplus UAVs to low-cost targets.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
)

_POP_INIT_SRP = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a **single-route patrol (SRP)** problem \
(N < M, each UAV visits multiple targets in sequence). The solver will convert your \
assignments into an evolutionary algorithm's internal encoding, so you only need to \
produce **discrete assignments**.

{_json_format}
{{
    "thought": "<your construction strategy>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": 0, "targets": [3, 1, 4]}},
                {{"uav": 1, "targets": [2, 0]}}
            ]
        }}
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete assignment** covering ALL N UAVs and ALL M targets.

## Rules (srp: N < M)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the **order represents the tour sequence**.
- Tour cost = C_UT[uav][first_target] + sum(C_TT[prev][next]) for transitions.

{_reasoning}

## Construction Strategy
### Phase 1: Partition + Nearest-Neighbor Tour (solution 1)
1. **Partition** targets among UAVs:
   - Identify difficult targets (few feasible UAVs), assign to their best UAV.
   - Group remaining targets by spatial proximity (use TopKTargets_per_UAV as anchor).
   - Each UAV should get roughly M/N targets.
2. **Order** each UAV's tour:
   - Start with the target that has lowest C_UT[uav][target].
   - Use nearest-neighbor: from current target, go to the closest unvisited target \
     (use C_TT or C_TT_sparse TopK adjacency).
3. Compute total cost = C_UT[uav][first] + sum(C_TT transitions).

### Phase 2: Tour Perturbations (solutions 2-K)
- **2-opt swap**: reverse a segment within one UAV's tour.
- **Target reassignment**: move one target from a heavily-loaded UAV to a lighter one.
- **Different starting target**: re-run nearest-neighbor from a different first target.
- **Partition variation**: try a different initial grouping of targets.

{_validation}

## Guidelines
- Minimize total cost = C_UT (UAV→first target) + C_TT (target→target transitions).
- Use TopKTargets_per_UAV to identify low-cost initial targets for each UAV.
- Use TopKNextTargets_per_target (or C_TT_sparse.top_next) to find efficient transitions.
- Avoid long tours for a single UAV when targets are geographically spread out.
- If C_TT is provided as C_TT_sparse (top-K adjacency), use it to guide tour ordering.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
)

# 通用回退（包含所有场景规则，用于 model_type 未知时）
_POP_INIT_GENERIC = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a UAV scheduling problem. \
The solver will convert your assignments into an evolutionary algorithm's \
internal encoding, so you only need to produce **discrete assignments**.

{_json_format}
{{
    "thought": "<your construction strategy>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": <int>, "targets": [<int>, ...]}},
                ...
            ]
        }},
        ...
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete assignment plan** covering ALL N UAVs.

## Assignment Rules by Model Type

### balanced (N == M, one-to-one)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once, each target appears exactly once.
- Each "targets" list has exactly 1 element.

### overloaded (N > M, UAVs outnumber targets)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target must appear at least once across all assignments.
- Each "targets" list has exactly 1 element.

### srp (N < M, UAVs visit multiple targets in sequence)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the order represents the **tour sequence**.

{_reasoning}
{_validation}

## Guidelines
- Focus on minimizing total cost while respecting constraints.
- Use the preference summary to identify low-cost assignments.
- For contested targets, assign to the best UAV first.
- For SRP, use nearest-neighbor ordering guided by C_TT transition costs.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
)

# Scene → Prompt 映射
_POP_INIT_SCENE_MAP: dict[str, str] = {
    "balanced": _POP_INIT_BALANCED,
    "overloaded": _POP_INIT_OVERLOADED,
    "srp": _POP_INIT_SRP,
}


# =============================================================================
# Population Init — User Prompt
# =============================================================================

POPULATION_INIT_USER_PROMPT = """\
## Problem (S_problem)
{problem_json}

## Task
Generate exactly {k} complete assignment solutions. \
Each solution must cover ALL {n_uavs} UAVs and satisfy all \
constraints for the '{model_type}' model type.

QUALITY PRIORITY:
- Use the cost matrix (C_UT, C_TT) and preference data to construct LOW-COST solutions.
- Think through your construction strategy in the "thought" field.
- Output ONLY the JSON object. No markdown fences, no preamble.
"""


# =============================================================================
# Search Controller — Scene-specific System Prompts
# =============================================================================

_SC_BASE = """\
You are an expert controller for a Differential Evolution (DE) solver on a \
combinatorial optimization problem (UAV-target assignment with discrete mapping).

You make sequential control decisions. You are consulted only when the search \
state changes (see trigger_reason) or when a fallback timer expires.

{actions_block}

## Reference Magnitudes (thresholds injected from solver config — authoritative)
- |delta_fitness_pct| < {df_noise_pct}%: effectively ZERO improvement. \
This is the solver's noise floor; do not treat smaller values as signal.
- |delta_diversity| < {dd_noise}: NOISE — treat as "diversity unchanged". \
This is the SAME threshold the trigger uses to decide whether to consult you, \
so a value below it is by construction not a diversity event.
- |df - df_shadow| < {shadow_contrast}%: your choices are INDISTINGUISHABLE from \
the fixed-CR baseline. See Shadow Control section.
- delta_fitness_pct is NOT normalized by stage_length. A long stage shows a \
larger df purely because it spans more generations — always divide by the \
Len column before comparing df across stages.
- stagnation_raw: true consecutive non-improving generations (uncapped). \
Higher values = deeper convergence. Several hundred = genuine long-term stagnation.
- acceptance_rate: fraction of offspring that survived selection. \
Lower = population is harder to improve. Combined with stagnation_raw, \
it diagnoses whether the search has converged or is still active.
- gens_since_last_improvement: how long since the last best-fitness update. \
A large value relative to stage_length confirms prolonged stagnation.
- feasible_ratio: fraction of the population satisfying all constraints \
(time windows, range). violation_mean / violation_max quantify how badly the \
rest violate them. GMR and F are the two levers that most affect feasibility: \
if feasible_ratio is low, prefer reducing F or holding GMR=off rather than \
increasing exploration.

## Diagnostic Reasoning Chain
Your decision must follow this sequence. Do NOT skip steps or jump to conclusions.

### Step 1: Search State Diagnosis
Synthesize ALL available signals into a coherent picture:
- **Convergence**: acceptance_rate, stagnation_raw, gens_since_last_improvement.
  Low acceptance + long stagnation = the population has converged.
- **Diversity structure**: diversity, diversity_p25, diversity_p75, delta_diversity.
  Is the population clustered? Are there outliers? Is diversity changing?
- **Improvement trajectory**: delta_fitness_pct, stage_best_curve.
  Is the search still finding better solutions? Is improvement decelerating?
- **Constraint pressure**: feasible_ratio, violation_mean, violation_max.
  Low feasible_ratio means the population is fighting the constraints — \
  prefer lowering F or holding GMR=off over increasing exploration.
- **Trigger context**: trigger_reason tells you why you were consulted. \
  "stagnation_deepen" = the search is stuck. "fitness_move" = something just changed. \
  "fallback_timer" = no event occurred, you may be consulted speculatively.
- **Cross-check**: do these signals agree? Disagreement is itself informative — \
  e.g., low acceptance + high diversity may indicate oscillation, not convergence. \
  Check stage_best_curve for instability in such cases.

### Step 2: Parameter Effectiveness Diagnosis
{param_effect_block}

### Step 3: Decision
{decision_block}

## Decision Policy
- Hold is the default. You were possibly consulted by a fallback timer rather than
  by a real event — being consulted is not evidence that a parameter should change.
- Changing a parameter requires evidence that (a) appeared since your last decision AND
  (b) distinguishes the candidate values from each other.
- The following are NOT valid reasons to change:
  * "I was consulted" / "it is time to act" / "the controller should respond".
  * The current value has been in place for several stages.
  * A generic wish to explore more, exploit more, or "try something different".
  * The state is unchanged — an unchanged state is evidence FOR holding.
"""

# ---- 动作空间块：解耦（三通道独立） ----
_SC_ACTIONS_DECOUPLED = """\
## Actions You Control
You independently control three DE parameters:

1. **CR** (crossover rate), one of {cr_choices}:
   Controls the per-gene probability of using DE/rand/1 (exploration) vs \
DE/best/2 (exploitation). Higher CR → more rand/1 → more exploration.
   Decision: cr_action = "hold" (keep current) or "set" (choose new value).

2. **F** (mutation scale factor), one of {f_choices}:
   Controls the step size of differential mutations. Higher F → larger steps. \
F is independent of CR — you can set them independently.
   "auto" = solver derives F from CR via formula (legacy coupled mode). \
A specific value overrides the formula.
   Decision: f_action = "hold" (keep current) or "set" (choose new value).

3. **GMR** (global mutation rate / extinction mode):
   Controls whether the solver applies extinction (resetting poor individuals).
   - "auto": extinction triggered by formula (CR < threshold → probabilistic).
   - "on": force extinction this stage (resets worst individuals).
   - "off": suppress extinction this stage.

## Decoupled Control
Previously, F was derived from CR (formula 3-11) and GMR was derived from CR \
(formula 3-12). You now control them independently. This means:
- You can increase exploration (high CR) without amplifying step size (keep F moderate).
- You can fine-tune (low F) without forcing exploitation (keep CR moderate).
- You can trigger extinction (GMR=on) without affecting CR or F.
Use this freedom to match the search dynamics, not to set all three every time. \
If the current values are working, hold them.
"""

# ---- 动作空间块：耦合（仅 CR，F/GMR 由公式 3-11/3-12 推导）----
# 这是原 DMDE 的手工耦合关系，用作解耦主张的对照条件。
# 关键：模型不得看到 F/GMR 通道，否则会产生"看得见却被忽略"的混淆变量。
_SC_ACTIONS_COUPLED = """\
## Actions You Control
You control exactly ONE DE parameter:

1. **CR** (crossover rate), one of {cr_choices}:
   Controls the per-gene probability of using DE/rand/1 (exploration) vs \
DE/best/2 (exploitation). Higher CR → more rand/1 → more exploration.
   Decision: cr_action = "hold" (keep current) or "set" (choose new value).

## Coupled Parameters (NOT under your control)
F (mutation scale factor) and GMR (global mutation rate / extinction) are \
derived from CR by the solver's built-in formulas:
- F is computed from CR (formula 3-11): F = 2*CR, or F = (2-CR)/2.
- GMR is computed from CR (formula 3-12): GMR = 0 if CR >= delta, \
else GMR = 1 - CR.

You CANNOT set F or GMR directly, and you must NOT output them. \
Your only lever is CR — but be aware that every CR change also moves F and GMR \
through the formulas above. Choosing a higher CR simultaneously enlarges the \
mutation step (F) and suppresses extinction (GMR). \
You must therefore pick a CR that is acceptable on all three dimensions at once; \
you cannot adjust them independently.
"""

# ---- 动作空间块：CR 锁定（部分解耦：只控 F / GMR）----
# 用于检验预测：CR 通道被 LLM 滥用（CR 设定次数与增益 Spearman ρ=+0.93），
# 锁住 CR、只把 F/GMR 交给 LLM，是否优于全解耦与全耦合。
_SC_ACTIONS_NOCR = """\
## Actions You Control
You control two DE parameters. **CR is locked by the solver and is NOT yours to
change** — do not output it.

1. **F** (mutation scale factor), one of {f_choices}:
   Controls the step size of differential mutations. Higher F → larger steps.
   Decision: f_action = "hold" (keep current) or "set" (choose new value).

2. **GMR** (global mutation rate / extinction mode):
   Controls whether the solver applies extinction (resetting poor individuals).
   - "auto": extinction triggered by formula.
   - "on": force extinction this stage (resets worst individuals).
   - "off": suppress extinction this stage.

## Why CR Is Locked
CR is held fixed by the solver at a value chosen offline. You cannot and must not
change it. Your levers are the mutation step size (F) and extinction (GMR) —
use them to respond to the search state.
"""

_SC_SHADOW = """\
## Shadow Control (Attribution)
A shadow population, evolved from the SAME starting point as the main population \
with FIXED CR={shadow_cr} (and F, GMR derived from that CR via formulas), \
is run in parallel over the same stage window.
The shadow represents the "no LLM intervention" baseline: \
all parameters are formula-derived from a fixed CR. \
The difference df - df_shadow = the combined effect of your CR+F+GMR choices.

### Attribution Gate (quantitative)
|df - df_shadow| must exceed **{shadow_contrast}%** to count as an effect of \
your choices. Below that threshold the difference is within measurement noise \
and is NOT evidence — neither that you helped nor that you hurt. \
When the gate is not passed, the correct action is to hold, not to experiment.

### Reading the Shadow Signal
- **df significantly better than df_shadow**: your strategy outperforms the fixed baseline. \
  Hold — do not change what is working. (This does not guarantee your parameters are optimal, \
  but changing without further evidence is risky.)
- **df ≈ df_shadow (both near zero)**: converged. Neither your strategy nor the baseline \
  can improve fitness. Consider GMR=on to force diversity injection.
- **df ≈ df_shadow (both significantly negative)**: both produce similar \
  improvement. Your choices are not differentiating — the improvement comes from \
  the search landscape, not your strategy. Hold.
- **df worse than df_shadow**: your current strategy is actively hurting. Strongest evidence \
  for changing parameters — consider adjusting CR, F, or GMR.

### Shadow Across Stages
If df tracks df_shadow across multiple stages, your parameter choices are not adding \
value over the fixed baseline. Reduce confidence in parameter adjustments; rely more \
on GMR=on to force diversity injection.
"""

# ---- Step 2 / Step 3 块：解耦（CR / F / GMR 分别诊断）----
_SC_PARAM_EFFECT_DECOUPLED = """\
Evaluate the current CR, F, and GMR settings:

**CR effectiveness**:
- df vs df_shadow is the PRIMARY attribution signal. See Shadow Control section.
- If acceptance_rate is very low, the population is not accepting offspring \
regardless of CR — CR is not the bottleneck.
- If all recent stages used the same CR, you have no comparative data. \
The absence of evidence is NOT evidence of absence.

**F effectiveness**:
- If F is too high: mutations overshoot, offspring are rejected (low acceptance). \
  stage_best_curve will show oscillation or flat+high-rejection.
- If F is too low: mutations are tiny, search crawls (slow df, long stagnation). \
  stage_best_curve will show gradual decline with no acceleration.
- If F is appropriate: steady improvement with reasonable acceptance.

**GMR effectiveness**:
- GMR=on forces extinction: diversity spikes, fitness may temporarily worsen. \
  Useful when stagnation is deep and CR/F changes have not helped.
- GMR=off suppresses extinction: preserves current population structure. \
  Useful when the search is actively improving and extinction would be disruptive.
- GMR=auto: the formula decides. Review stage_history to see if auto triggers \
  are well-timed or disruptive."""

_SC_DECISION_DECOUPLED = """\
- If the search is actively improving → **hold everything**.
- If stagnating but CR/F/GMR have not been varied → consider an **exploratory change**. \
  Pick the parameter most likely to address the diagnosed bottleneck.
- If converged and CR/F changes have not helped → consider GMR=on to force diversity injection.
- If uncertain → **hold**. A wrong change can harm search; holding cannot."""

# ---- Step 2 / Step 3 块：耦合（只能动 CR）----
_SC_PARAM_EFFECT_COUPLED = """\
Evaluate the current CR setting. Remember F and GMR are NOT yours to set — \
they follow CR through the formulas, so judge CR on all three effects at once:

**CR effectiveness**:
- df vs df_shadow is the PRIMARY attribution signal. See Shadow Control section.
- If acceptance_rate is very low, the population is not accepting offspring \
regardless of CR — CR is not the bottleneck.
- If all recent stages used the same CR, you have no comparative data. \
The absence of evidence is NOT evidence of absence.

**Coupled side effects to weigh before changing CR**:
- Raising CR also raises F (larger mutation steps) and pushes GMR toward 0 \
(fewer extinctions). Useful when the search is stagnant AND diversity is low.
- Lowering CR also lowers F (finer steps) and raises GMR (more extinctions). \
Useful when the search is close to a good solution but overshoots it.
- If the population is fighting constraints (low feasible_ratio), be careful: \
raising CR inflates F and may worsen feasibility."""

_SC_DECISION_COUPLED = """\
- If the search is actively improving → **hold**.
- If stagnating and CR has not been varied → consider an **exploratory change** to CR. \
  Weigh its coupled side effects on F and GMR before choosing.
- If uncertain → **hold**. A wrong change can harm search; holding cannot.
- You cannot trigger extinction directly. If you believe extinction is needed, \
the only way to influence it is through CR (lower CR raises GMR)."""

# ---- Step 2 / Step 3 块：CR 锁定（只评估 F / GMR）----
_SC_PARAM_EFFECT_NOCR = """\
Evaluate the current F and GMR settings. CR is locked — do not reason about \
changing it; treat it as part of the fixed environment:

**F effectiveness**:
- If F is too high: mutations overshoot, offspring are rejected (low acceptance). \
  stage_best_curve will show oscillation or flat+high-rejection.
- If F is too low: mutations are tiny, search crawls (slow df, long stagnation). \
  stage_best_curve will show gradual decline with no acceleration.
- If F is appropriate: steady improvement with reasonable acceptance.

**GMR effectiveness**:
- GMR=on forces extinction: diversity spikes, fitness may temporarily worsen. \
  Useful when stagnation is deep and F changes have not helped.
- GMR=off suppresses extinction: preserves current population structure. \
  Useful when the search is actively improving and extinction would be disruptive.
- GMR=auto: the formula decides. Review stage_history to see if auto triggers \
  are well-timed or disruptive."""

_SC_DECISION_NOCR = """\
- If the search is actively improving → **hold everything**.
- If stagnating and F/GMR have not been varied → consider an **exploratory change**. \
  Pick the parameter most likely to address the diagnosed bottleneck.
- If converged and F changes have not helped → consider GMR=on to force diversity injection.
- If uncertain → **hold**. A wrong change can harm search; holding cannot."""

_SC_FROZEN = """\
## CR Channel Is FROZEN This Round
The solver measured that over the last stage your parameters produced no improvement
(|df| within noise) and no difference from the fixed-CR shadow \
(|df - df_shadow| within noise). The CR channel is therefore currently
uninformative, and CR is FROZEN — you must set `cr_action: "hold"`.
{frozen_tail}
"""

# 冻结段结尾随模式变化：耦合模式下模型没有 F/GMR 通道，沿用解耦措辞会给出
# 自相矛盾的指令（"你可以调整你并不拥有的通道"）。
_SC_FROZEN_TAIL_DECOUPLED = """\
You may still adjust F and GMR. The CR channel being frozen does not \
mean the search is healthy — it means CR is not the lever to use right now."""

_SC_FROZEN_TAIL_COUPLED = """\
CR is your only lever and it is frozen this round, so the correct output is \
`cr_action: "hold"`. This does not mean the search is healthy — it means CR is \
not the lever to use right now."""

_SC_FROZEN_GMR = """\
## GMR Channel Is FROZEN This Round
You have actively used GMR (on/off) for several consecutive stages and the search
produced no meaningful improvement in any of them. Repeating the same lever without
effect is not a strategy — GMR is FROZEN this round: you must set `gmr_mode: "auto"`
(let the formula decide). You may still adjust CR and F.
GMR unlocks automatically once the search shows real improvement again.
"""

_SC_FROZEN_F = """\
## F Channel Is FROZEN This Round
You have actively changed F for several consecutive stages and the search produced
no meaningful improvement in any of them. F is FROZEN this round: you must set
`f_action: "hold"` and `f: null`. You may still adjust CR and GMR.
F unlocks automatically once the search shows real improvement again.
"""

_SC_SIGNAL_GUIDE = """\
## Signal Interpretation

### Stage Best Curve (stage_best_curve)
A time series [gen_offset, delta_pct] within the current stage. \
delta_pct = (stage_start_fitness - current_best) / stage_start_fitness * 100.
- Steep initial drop → rapid early improvement (current dynamics are healthy).
- Flat throughout → no improvement this stage.
- Drop then plateau → improvement stalled mid-stage (possible local optimum).
- Gradual steady decline → slow but consistent progress.
The curve shape reflects search dynamics, not a specific CR value.

### Stage History Table
Each row is one of your previous decision stages. Read it for:
- **CR-response**: did df change when CR changed across stages? \
  Consistent correlation → CR is effective. No correlation → insensitive.
- **F-response**: did changing F affect acceptance_rate or df? \
  High F + low acceptance → F may be too high (mutations overshoot). \
  Low F + slow improvement → F may be too low.
- **GMR-response**: did GMR=on cause diversity spike + fitness reset? \
  Did GMR=off preserve improvement momentum?
- **Shadow tracking**: does df follow df_shadow? If always similar, \
  your parameter choices are not adding value over the fixed baseline.
- **Diminishing returns**: are improvements shrinking stage over stage?
- **Acceptance trend**: is acceptance_rate declining? \
  Declining = the population is hardening against new solutions.

### Evidence Budget (how much history you actually get)
You receive at most the last 5 stages, and channels are frequently FROZEN, so
the number of stages in which a given parameter actually varied is usually 1-3.
Calibrate your confidence to what you actually have:
- **0-1 comparable stages for a parameter** → you have NO evidence about it. \
  Hold that parameter. Never infer a trend from a single observation.
- **2-3 comparable stages** → weak evidence. Act only if the direction is \
  consistent across all of them AND the shadow attribution gate is passed.
- **4+ comparable stages** → a trend may be read, still subject to the gate.
A stage counts as "comparable" for a parameter only if that parameter actually
varied in it. Stages where it was frozen or held are NOT evidence.
"""

_SC_FORMAT = """\
## Decision Format
Respond with a JSON object only (no markdown), filling the fields IN THIS ORDER:
{{
    "evidence_read": "<state: ...> | <param_effect: ...> | <history: ...>",
    "cr_action": "hold" | "set",
    "cr": {cr_null_or_value},
    "f_action": "hold" | "set",
    "f": {f_null_or_value},
    "gmr_mode": {gmr_value},
    "reasoning": "<one sentence>"
}}

Rules:
- `evidence_read`: three parts — state (converged/exploring/stagnating), \
  param_effect (are current CR/F/GMR helping?), history (stage_history trend).
  If no stage history exists yet, write "history: first decision".
- `cr_action`: "hold" unless evidence supports a different CR.
  If "hold", `cr` MUST be null.
  If "set", `cr` must be one of {cr_choices}.
- `f_action`: "hold" unless evidence supports a different F. \
  If "hold", `f` MUST be null. \
  If "set", `f` must be one of {f_choices}.
- `gmr_mode`: "auto" = formula decides (default). \
  "on" = force extinction (deep stagnation, other levers exhausted). \
  "off" = suppress extinction (active improvement, don't disrupt).
- Do not alternate values without evidence — a hold is not a failure to act.
"""

# ---- 输出格式：耦合模式（无 F / GMR 字段）----
# ---- 输出格式：CR 锁定（只有 F / GMR 字段）----
_SC_FORMAT_NOCR = """\
## Decision Format
Respond with a JSON object only (no markdown), filling the fields IN THIS ORDER:
{{
    "evidence_read": "<state: ...> | <param_effect: ...> | <history: ...>",
    "f_action": "hold" | "set",
    "f": {f_null_or_value},
    "gmr_mode": {gmr_value},
    "reasoning": "<one sentence>"
}}

Rules:
- `evidence_read`: three parts — state (converged/exploring/stagnating), \
  param_effect (are current F / GMR helping?), history (stage_history trend).
  If no stage history exists yet, write "history: first decision".
- `f_action`: "hold" unless evidence supports a different F. \
  If "hold", `f` MUST be null. If "set", `f` must be one of {f_choices}.
- `gmr_mode`: "auto" = formula decides (default). \
  "on" = force extinction (deep stagnation, other levers exhausted). \
  "off" = suppress extinction (active improvement, don't disrupt).
- Do NOT output `cr` or `cr_action` — the CR channel is locked and will be rejected.
- Do not alternate values without evidence — a hold is not a failure to act.
"""

_SC_FORMAT_COUPLED = """\
## Decision Format
Respond with a JSON object only (no markdown), filling the fields IN THIS ORDER:
{{
    "evidence_read": "<state: ...> | <param_effect: ...> | <history: ...>",
    "cr_action": "hold" | "set",
    "cr": {cr_null_or_value},
    "reasoning": "<one sentence>"
}}

Rules:
- `evidence_read`: three parts — state (converged/exploring/stagnating), \
  param_effect (is the current CR helping, and are its coupled F/GMR side effects \
  favourable?), history (stage_history trend).
  If no stage history exists yet, write "history: first decision".
- `cr_action`: "hold" unless evidence supports a different CR.
  If "hold", `cr` MUST be null.
  If "set", `cr` must be one of {cr_choices}.
- Do NOT output `f`, `f_action` or `gmr_mode` — those channels are not available \
  to you and will be rejected.
- Do not alternate values without evidence — a hold is not a failure to act.
"""

_SC_BALANCED_EXTRA = """\
## Scene: balanced (N == M)
- One-to-one assignment: each UAV maps to exactly one target.
- The search space is a **permutation space** — diversity tends to drop quickly \
because swapping assignments between UAVs can lead to rapid convergence.
- Consider the trade-off between maintaining population diversity \
and converging toward low-cost permutations.
- Use the observed optimization state and recent trajectory \
to determine the appropriate CR.
"""

_SC_OVERLOADED_EXTRA = """\
## Scene: overloaded (N > M)
- Multiple UAVs share targets — the search space has **redundancy** \
(different UAV combinations can achieve similar coverage).
- Diversity is naturally higher due to target sharing, \
but this also means many solutions are near-equivalent.
- Consider whether the solver is exploring enough distinct assignment patterns \
or getting stuck in redundant regions.
- Use the observed optimization state and recent trajectory \
to determine the appropriate CR.
"""

_SC_SRP_EXTRA = """\
## Scene: srp (N < M)
- Each UAV visits multiple targets in sequence — **tour ordering** matters.
- The search space is larger (assignment + ordering combined), \
and the interaction between assignment and ordering creates complex fitness landscapes.
- Consider whether the solver needs to explore different assignment-to-ordering combinations \
or refine existing tour segments.
- Use the observed optimization state and recent trajectory \
to determine the appropriate CR.
"""

_SC_SCENE_MAP: dict[str, str] = {
    "balanced": _SC_BALANCED_EXTRA,
    "overloaded": _SC_OVERLOADED_EXTRA,
    "srp": _SC_SRP_EXTRA,
}


def get_search_controller_prompt(
    cr_choices: list[float] | None = None,
    model_type: str | None = None,
    f_choices: list[float] | None = None,
    shadow_cr: float | None = None,
    cr_frozen: bool = False,
    gmr_frozen: bool = False,
    f_frozen: bool = False,
    df_noise_pct: float = 0.05,
    dd_noise: float = 0.05,
    shadow_contrast: float = 0.05,
    coupled: bool = False,
    no_cr: bool = False,
) -> str:
    """获取搜索控制器 system prompt。

    Args（新增）:
        coupled: True = 耦合对照条件。模型只控制 CR，F/GMR 由公式 3-11/3-12 推导。
                 用于检验核心主张"解耦是否优于手工耦合"——必须与解耦条件使用
                 完全相同的观测集、触发机制与守卫，仅动作空间不同。
        no_cr:   True = CR 锁定条件（部分解耦）。CR 由 solver 锁定为离线选定值，
                 模型只控制 F / GMR。用于检验预测：CR 通道被模型滥用
                 （CR 设定次数与增益 Spearman ρ=+0.93），移除它是否更优。
                 与 coupled 互斥；同时为 True 时以 no_cr 优先。

    Args:
        cr_choices: CR 候选值列表
        model_type: 场景类型 ("balanced"/"overloaded"/"srp")，
                     非空时追加场景搜索特性描述（供 LLM 参考，非决策规则）
        shadow_cr: 影子对照种群使用的固定 CR（None = 无影子对照）
        cr_frozen: CR 通道是否被冻结（无证据守卫触发）。
                   True 时注入冻结段并把输出格式的 cr 固定为 null
        gmr_frozen: GMR 通道是否被冻结（连续无效动作守卫）。
                    True 时注入冻结段并把 gmr_mode 固定为 "auto"
        f_frozen: F 通道是否被冻结（连续无效动作守卫）。
                  True 时注入冻结段并把 f_action 固定为 "hold"
        df_noise_pct: Δf 噪声阈值(%)，与 solver freeze.df_noise 同源
        dd_noise: ΔD 噪声阈值，与 solver trigger.dd_threshold 同源
        shadow_contrast: 影子归因阈值(%)，与 solver freeze.shadow_contrast 同源

    三个阈值必须从 solver 配置传入，避免 prompt 与 solver 判据口径漂移
    （历史上 prompt 写死 0.02 而 solver 用 0.05，导致读数标准与采样标准不一致）。
    """
    if cr_choices is None:
        cr_choices = [0.1, 0.3, 0.5, 0.7, 0.9]
    if f_choices is None:
        f_choices = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5]

    # 动作空间 / 参数诊断 / 决策策略三块：解耦与耦合共用同一观测集与诊断链，
    # 只有"能动什么"不同——这是本对照实验要隔离的唯一变量。
    if no_cr:
        actions_block = _SC_ACTIONS_NOCR.format(f_choices=f_choices)
        param_effect_block = _SC_PARAM_EFFECT_NOCR
        decision_block = _SC_DECISION_NOCR
    elif coupled:
        actions_block = _SC_ACTIONS_COUPLED.format(cr_choices=cr_choices)
        param_effect_block = _SC_PARAM_EFFECT_COUPLED
        decision_block = _SC_DECISION_COUPLED
    else:
        actions_block = _SC_ACTIONS_DECOUPLED.format(
            cr_choices=cr_choices, f_choices=f_choices)
        param_effect_block = _SC_PARAM_EFFECT_DECOUPLED
        decision_block = _SC_DECISION_DECOUPLED

    base = _SC_BASE.format(
        actions_block=actions_block,
        param_effect_block=param_effect_block,
        decision_block=decision_block,
        cr_choices=cr_choices,
        f_choices=f_choices,
        df_noise_pct=df_noise_pct,
        dd_noise=dd_noise,
        shadow_contrast=shadow_contrast,
    )
    # 影子对照段：仅在 solver 实际启用影子种群时注入
    if shadow_cr is not None:
        base += _SC_SHADOW.format(
            shadow_cr=shadow_cr,
            shadow_contrast=shadow_contrast,
        )
    # 冻结段：CR 通道被判定为无信息时，明确禁止本轮改动 CR
    # no_cr 模式下 CR 本就锁定，注入"CR 被冻结"是冗余且可能误导，故跳过。
    if cr_frozen and not no_cr:
        base += _SC_FROZEN.format(
            frozen_tail=(_SC_FROZEN_TAIL_COUPLED if coupled
                         else _SC_FROZEN_TAIL_DECOUPLED)
        )
    # 失败冻结段：GMR / F 连续多次无效后禁止本轮继续操作该通道
    # 耦合模式下这两段不适用——模型本就没有 F/GMR 通道，注入会自相矛盾。
    if gmr_frozen and not coupled:
        base += _SC_FROZEN_GMR
    if f_frozen and not coupled:
        base += _SC_FROZEN_F
    scene_extra = _SC_SCENE_MAP.get(model_type, "") if model_type else ""
    if no_cr:
        return base + _SC_SIGNAL_GUIDE + scene_extra + _SC_FORMAT_NOCR.format(
            f_choices=f_choices,
            f_null_or_value=(
                "null (F is FROZEN — you must hold)" if f_frozen
                else "null if f_action is \"hold\", else one of {f_choices}".format(
                    f_choices=f_choices)
            ),
            gmr_value=(
                "\"auto\" (GMR is FROZEN — you must use auto)" if gmr_frozen
                else "\"auto\" | \"on\" | \"off\""
            ),
        )
    if coupled:
        return base + _SC_SIGNAL_GUIDE + scene_extra + _SC_FORMAT_COUPLED.format(
            cr_choices=cr_choices,
            cr_null_or_value="null (CR is FROZEN — you must hold)" if cr_frozen
            else "null if cr_action is \"hold\", else one of {cr_choices}".format(
                cr_choices=cr_choices),
        )
    fmt = _SC_FORMAT.format(
        cr_choices=cr_choices,
        f_choices=f_choices,
        cr_null_or_value="null (CR is FROZEN — you must hold)" if cr_frozen
        else "null if cr_action is \"hold\", else one of {cr_choices}".format(
            cr_choices=cr_choices),
        f_null_or_value=(
            "null (F is FROZEN — you must hold)" if f_frozen
            else "null if f_action is \"hold\", else one of {f_choices}".format(
                f_choices=f_choices)
        ),
        gmr_value=(
            "\"auto\" (GMR is FROZEN — you must use auto)" if gmr_frozen
            else "\"auto\" | \"on\" | \"off\""
        ),
    )

    return base + _SC_SIGNAL_GUIDE + scene_extra + fmt


# =============================================================================
# Search Controller — User Prompt
# =============================================================================

SEARCH_CONTROLLER_USER_PROMPT = """\
## Current Search State
{state_json}

## Stage History (each row = one of your previous decisions)
{trajectory_text}

## Task
Decide CR, F, and GMR for the next stage. \
For CR and F: answer two questions in order — (1) should the parameter change at all \
(the default is "hold"); (2) only if yes, what value. \
Use the evidence: current state, stage history, shadow contrast, and the \
trigger_reason (why you are being consulted now). \
Note that being consulted is not by itself evidence: the controller also consults \
you on a fallback timer. Respond with JSON only.
"""


# =============================================================================
# CR Control (deprecated) — Prompts
# =============================================================================

CR_CONTROL_SYSTEM_PROMPT = """\
You are an expert in Differential Evolution parameter control.

Your task: Adjust the crossover rate (CR) and scale factor (F) offsets \
based on the current search state.

## Decision Format
Respond with a JSON object only (no markdown):
{
    "cr_offset": <float in [-0.3, 0.3], or null to keep default>,
    "f_offset": <float in [-0.3, 0.3], or null to keep default>,
    "reasoning": "<brief explanation>"
}

## Guidelines
- High stagnation + low diversity → increase CR (more exploration)
- Converging well → decrease CR slightly (more exploitation)
- Large fitness variance → increase F (bigger steps)
- Near convergence → decrease F (fine-tuning)
- Only adjust when the state clearly warrants a change (prefer null)
"""

CR_CONTROL_USER_PROMPT = """\
## Current Search State
{state_json}

The `previous_decision_feedback` section shows the effect of your last decision.
Use this feedback to evaluate whether your previous adjustment helped or hurt,
and adapt your next decision accordingly.

## Task
Adjust CR and F offsets for the next generation.
Base your decision on the **current state and feedback trajectory**, \
not on preset rules for the scenario.
Respond with JSON only.
"""


# =============================================================================
# Operator Selection (deprecated) — Prompts
# =============================================================================

def get_operator_selection_prompt(strategies: list[str] | None = None) -> str:
    if strategies is None:
        strategies = ["rand/1", "best/1", "best/2", "current-to-pbest/1", "rand/2", "rand-to-best/1"]
    strategies_str = ", ".join(strategies)
    return f"""\
You are an expert in Differential Evolution (DE) for combinatorial optimization \
(UAV-target assignment with discrete mapping).

Your task: Select the most suitable DE operator strategy for the current search state.

## Available Strategies
{strategies_str}

## Decision Format
Respond with a JSON object only (no markdown):
{{
    "strategy": "<strategy name>",
    "reasoning": "<brief explanation>"
}}

## Guidelines
- Low diversity + stagnation → exploratory (rand/1, rand/2)
- High diversity + slow convergence → exploitative (best/1, best/2)
- Balanced state → balanced (current-to-pbest/1)
"""

OPERATOR_SELECTION_USER_PROMPT = """\
## Current Search State
{state_json}

## Recent Trajectory
{trajectory_text}

## Task
Select the best DE strategy for the next interval. Respond with JSON only.
"""


# =============================================================================
# Scene-specific System Prompt Dispatcher
# =============================================================================

def get_scene_specific_system_prompt(
    module_name: str,
    model_type: str,
    **kwargs: Any,
) -> str:
    """获取针对特定场景（model_type）的系统提示词。

    根据 N,M 关系自动选择对应的 prompt，只包含该场景的规则和示例，
    去除无关场景的信息噪声，节省 token。

    Args:
        module_name: 模块名称 ("population_init", "search_controller")
        model_type: 场景类型 ("balanced", "overloaded", "srp")
        **kwargs: 额外参数（如 cr_choices）

    Returns:
        针对该场景的系统提示词
    """
    if module_name == "population_init":
        prompt = _POP_INIT_SCENE_MAP.get(model_type)
        if prompt is None:
            logger.warning(
                "[prompts] Unknown model_type %r for population_init, using generic prompt",
                model_type,
            )
            return _POP_INIT_GENERIC
        return prompt

    if module_name == "search_controller":
        return get_search_controller_prompt(model_type=model_type, **kwargs)

    raise ValueError(
        f"Unknown module for scene-specific prompt: {module_name!r}. "
        f"Available: ['population_init', 'search_controller']"
    )


# =============================================================================
# Registries
# =============================================================================

PROMPT_REGISTRY: dict[str, str] = {
    "population_init": _POP_INIT_GENERIC,
    "cr_control": CR_CONTROL_SYSTEM_PROMPT,
}

USER_PROMPT_REGISTRY: dict[str, str] = {
    "population_init": POPULATION_INIT_USER_PROMPT,
    "cr_control": CR_CONTROL_USER_PROMPT,
    "search_controller": SEARCH_CONTROLLER_USER_PROMPT,
    "operator_selection": OPERATOR_SELECTION_USER_PROMPT,
}


# =============================================================================
# Unified Interface
# =============================================================================

def get_prompt(
    module_name: str,
    prompt_type: str = "system",
    prompt_path: str | Path | None = None,
    model_type: str | None = None,
    **kwargs: Any,
) -> str:
    """获取指定模块的提示词。

    优先级：外部文件 > 场景化 prompt > 通用默认 prompt

    Args:
        module_name: 模块名称
        prompt_type: "system" 或 "user"
        prompt_path: 外部文件路径（覆盖默认）
        model_type: 场景类型 ("balanced"/"overloaded"/"srp")，用于场景化 system prompt
        **kwargs: 动态参数（用于 user prompt 插值或 search_controller 的 cr_choices）

    Returns:
        提示词字符串
    """
    # 优先从外部文件加载
    if prompt_path is not None:
        p = Path(prompt_path)
        if p.exists():
            logger.info("[prompts] Loading prompt from external file: %s", prompt_path)
            return p.read_text(encoding="utf-8")
        else:
            logger.warning("[prompts] Prompt file not found: %s, using default", prompt_path)

    # System prompt
    if prompt_type == "system":
        # 有 model_type 时优先用场景化 prompt
        if model_type is not None:
            return get_scene_specific_system_prompt(module_name, model_type, **kwargs)

        if module_name in PROMPT_REGISTRY:
            return PROMPT_REGISTRY[module_name]

        if module_name == "search_controller":
            return get_search_controller_prompt(**kwargs)

        if module_name == "operator_selection":
            return get_operator_selection_prompt(**kwargs)

        raise ValueError(
            f"Unknown system prompt module: {module_name!r}. "
            f"Available: {list(PROMPT_REGISTRY.keys()) + ['search_controller', 'operator_selection']}"
        )

    # User prompt
    elif prompt_type == "user":
        if module_name not in USER_PROMPT_REGISTRY:
            raise ValueError(
                f"Unknown user prompt module: {module_name!r}. "
                f"Available: {list(USER_PROMPT_REGISTRY.keys())}"
            )

        template = USER_PROMPT_REGISTRY[module_name]

        # model_type 是 get_prompt 的命名参数，不会自动进入 **kwargs，
        # 但 user prompt 模板中可能需要它，因此手动注入。
        if model_type is not None and "model_type" not in kwargs:
            kwargs = {**kwargs, "model_type": model_type}

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(
                "[prompts] Missing parameter %s for user prompt '%s', using raw template",
                e, module_name,
            )
            return template

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type!r}. Use 'system' or 'user'.")


def load_prompt_from_file(prompt_path: str | Path, fallback: str | None = None) -> str:
    """从文件加载提示词，失败时返回 fallback。"""
    p = Path(prompt_path)
    if not p.exists():
        logger.warning("[prompts] File not found: %s", prompt_path)
        if fallback is not None:
            return fallback
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return p.read_text(encoding="utf-8")


__all__ = [
    "get_prompt",
    "load_prompt_from_file",
    "get_scene_specific_system_prompt",
    "get_search_controller_prompt",
    "get_operator_selection_prompt",
    "POPULATION_INIT_USER_PROMPT",
    "CR_CONTROL_SYSTEM_PROMPT",
    "CR_CONTROL_USER_PROMPT",
    "SEARCH_CONTROLLER_USER_PROMPT",
    "OPERATOR_SELECTION_USER_PROMPT",
    "PROMPT_REGISTRY",
    "USER_PROMPT_REGISTRY",
]