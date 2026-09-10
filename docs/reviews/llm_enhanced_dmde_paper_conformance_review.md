# Paper Conformance Review: LLM-Enhanced DMDE Module

**Reviewer:** Automated Code Analysis  
**Date:** 2026-09-10  
**Module:** `src/algorithms/algorithm_llm_enhanced_dmde/`  
**References:**
- Ming (2017): "Improved discrete mapping differential evolution for multi-UAVs" — IJ Mach. Learn. & Cyber. 8:765–780
- Zhang (2025): "Leveraging Large Language Models for Dynamic Multi-Objective Optimization in UAV Sensor-Target Assignment" — IFAC PapersOnLine 59-20 (2025) 2314–2319

---

## DMDE Core (Ming 2017)

| Component | Paper Reference | Implementation File | Conformance | Notes |
|-----------|----------------|---------------------|-------------|-------|
| Encoding (N=M) | Rule 3.1: U and T both non-repetitive, one-to-one | `encoder.py::_generate_balanced()` | ✅ Conforms | Shuffled permutation of targets → one-to-one mapping. Correct. |
| Encoding (N>M) | Rule 3.2: U non-repetitive, T may repeat, each T at least once | `encoder.py::_generate_overloaded()` | ✅ Conforms | First m UAVs get distinct targets, remaining randomly assigned. Correct. |
| Encoding (N<M) | Rule 3.3: T non-repetitive, U may repeat (cruise), each U at least once | `encoder.py::_generate_srp()` | ✅ Conforms | Targets distributed to UAVs with greedy nearest-neighbor tour ordering. Matches paper's shortest-path principle for cruise sequences. |
| Gene structure | Triple vector (U_i, T_j, C(i,j)) or (T_j, T_{j+1}, Tc(j,j+1)) | `encoder.py::Gene` dataclass | ✅ Conforms | Stores (uav_id, target_id, cost). For SRP cruise genes, uav_id=-1 with target-to-target cost. Correct. |
| Mapping φ | Formula 3-5 (Eq. 12): φ: discrete → continuous via C_cost | `mapper.py::phi()` | ✅ Conforms | Simply extracts cost values from genes into a continuous vector. Exactly matches the paper's description: "calculate the differential of individuals in DMDE with the corresponding cost." |
| Inverse Mapping φ' | Formula 3-7 (Eq. 14): R^c → {(U_i,T_j)} via Match rules | `inverse_mapper.py::inverse_phi()` | ⚠️ Conforms with extensions | Core three-rule pipeline is correct, but adds temperature-adaptive softmax sampling and post-match perturbation not in the original paper. |
| Rule 3.4: Nearest Match | Eq. 15: min\|x'_j(t) - C(i,j)\| over valid (i,j) | `nearest_match.py::nearest_match()` | ✅ Conforms (base version) | Pure greedy nearest-distance matching. The adaptive variant adds softmax sampling — an enhancement beyond the paper. |
| Rule 3.5: Unique Match | 3.5a (N=M): delete row+col; 3.5b (N>M): delete row; 3.5c (N<M): replace row with Tc | `unique_filter.py` | ✅ Conforms | Uses boolean mask (equivalent to Inf reset as paper suggests). `mask_balanced` masks row+col, `mask_overloaded` masks row only, `mask_srp_upper` masks col and replaces row with target-to-target costs. Correct for all three cases. |
| Rule 3.6: Invalid Mutation | Random assignment from unmapped matrix positions for invalid values | `invalid_mutator.py::repair_invalid()` | ✅ Conforms | Randomly selects from available (unmasked) positions. Follows paper's description. |
| Dynamic CR | Formula 3-9 (Eq. 16): CR = 1 - (log(x)/log(S))^h | `crossover.py::dynamic_crossover_rate()` | ✅ Conforms | Exact formula implementation with x=current_gen, S=total_gens, h=zeta. Default h=3 matches paper's recommended value. |
| Hybrid Differential | Formula 3-10 (Eq. 17): CR≥rand → DE/rand/1; CR<rand → DE/best/2 | `crossover.py::hybrid_differential()` | ✅ Conforms | Per-gene strategy switching based on CR threshold. rand/1 = x_r1 + F*(x_r2-x_r3), best/2 = x_best + F*(x_r1+x_r2-x_r3-x_r4). Correct. |
| Dynamic F | Formula 3-11 (Eq. 18): F=2*CR if rand≤CR, else F=(2-CR)/2 | `scale_factor.py::dynamic_scale_factor()` | ✅ Formula correct / ❌ **Not used in solver** | Formula implementation is correct. **BUG: The solver hardcodes `base_f = 0.5` instead of calling this function.** See Discrepancy #1. |
| GMR Extinction | Formula 3-12: GMR=0 if CR≥δ, else GMR=1-CR | `extinction.py::gmr_rate()` | ⚠️ Formula correct / ⚠️ **Semantic mismatch** | Formula is correct. However, the paper treats GMR as a deterministic replacement ratio (proportion of population replaced each generation), while the code treats it as a probabilistic trigger. See Discrepancy #2. |
| Algorithm 3.1 Flow | Init → Evaluate → [Mutate → InvMap → Evaluate → Select] per generation | `llm_enhanced_dmde_solver.py::solve()` | ✅ Conforms | Main loop follows: init → eval → for each gen: mutate → inverse_phi → eval → greedy selection → extinction check. Correct flow. |

### LLM Enhancement (Zhang 2025)

| Component | Paper Reference | Implementation File | Conformance | Notes |
|-----------|----------------|---------------------|-------------|-------|
| Adaptive Operator Selection | Zhang §3 (Algorithm 1): LLM selects from operator pool every p iterations | `operator_selection.py` | ✅ Conforms | LLM selects from {rand/1, best/1, best/2, current-to-pbest/1, rand/2, rand-to-best/1}. Correct. |
| Feature Abstraction | Zhang §3.2 Table 2: f1-f6 search-dependent, f7-f8 solution-dependent, NO instance-dependent | `features/*.py` | ⚠️ Partially conforms | Includes search-dependent features (diversity, convergence speed, stagnation, feasible ratio). **Does not include** the paper's specific f1 (avg fitness), f2 (avg crowding distance), f3 (progress 1-i/T), f4 (improvement rate), f5 (avg std of objectives). Uses different but related features. Instance-dependent features correctly excluded. |
| Prompt Design | Zhang §3.2: Combine optimization trajectory + state features as input | `operator_selection.py::build_prompt()` | ✅ Conforms | Prompt includes state features (JSON) + recent trajectory (last 5 entries) + task instruction. Matches paper's two-component log design. |
| Optimization Trajectory | Zhang Algorithm 1 line 10: Record trajectory each iteration | `trajectory/optimization_trajectory.py` | ✅ Conforms | Records fitness, diversity, strategy, CR, F per generation. Trajectory fed to LLM. Correct. |
| Adaptive Cycle | Zhang Algorithm 1 line 4: every p iterations, trigger LLM | `solver.py` `gen % interval == 0` | ✅ Conforms | Default interval=50 for operator_selection, 10 for cr_control. Matches paper's p=10 design. |
| Operator Pool | Zhang §4.1: 3 crossover + 3 mutation operators | `operator_selection.py::AVAILABLE_STRATEGIES` | ⚠️ Adapted | Paper uses NSGA-II with crossover/mutation operators (single-point, multi-point, uniform crossover + swap, reverse, shift mutation). Implementation adapts to DE strategies (rand/1, best/1, etc.). Reasonable adaptation for DMDE context, but not a direct port. |
| Ablation Support | Zhang §4.3: LLM-MOEA\T (no trajectory), LLM-MOEA\F (no features) | `solver.py::modules` config | ✅ Conforms | Modules can be independently enabled/disabled via config dict. Supports full ablation. |

### DMDE-specific LLM Enhancements (AI Guide)

| Enhancement | Guide Reference | Implementation | Status |
|-------------|----------------|----------------|--------|
| Strategy Selection | §3.1: LLM selects DE strategy based on diversity/stagnation | `operator_selection.py` | ✅ Implemented |
| Temperature Control | §3.2: LLM controls inverse-mapping temperature decay | Not implemented as LLM module | ❌ **Missing** — Temperature uses fixed linear decay `1.0 - gen/max_gen`, not LLM-adaptive. |
| Extinction Control | §3.3: LLM decides whether to trigger extinction | Not implemented as LLM module | ❌ **Missing** — Extinction uses fixed δ threshold from paper, not LLM-driven. |
| Constraint Handling | §3.4: LLM adjusts α/β penalty weights | Not implemented | ❌ **Missing** — Penalty coefficients are external to this module. |
| SRP Tour Repair | §3.5: LLM gives specific gene reorderings for seq_violation | Not implemented as LLM module | ❌ **Missing** — SRP tour uses greedy + random perturbation, no LLM-guided repair. |

---

## Discrepancies & Issues

### ❌ CRITICAL: Dynamic Scale Factor F Not Used (Formula 3-11)

**Paper (Eq. 18):** F(t) = 2·CR(t) if rand[0,1] ≤ CR(t), else F(t) = (2-CR(t))/2

**Implementation (`solver.py` line):**
```python
base_f = 0.5  # 默认缩放因子
```

The formula is correctly implemented in `scale_factor.py::dynamic_scale_factor()` and `dynamic_scale_factor_batch()`, but **the solver never calls these functions**. Instead, it hardcodes `base_f = 0.5`. This means:
- F is constant at 0.5 throughout evolution
- The paper's dynamic F-CR coupling (where F varies with CR) is completely absent
- When LLM `f_offset` is applied, it's offset from 0.5 rather than from the dynamic F value

**Impact:** High. The paper explicitly states: "the value of the scale factor F is related to CR so that it is also the value of dynamic change" (Section 4.2). This coupling is a key design feature of DMDE.

**Fix:** Replace `base_f = 0.5` with `base_f = dynamic_scale_factor(cr, rng)` or use `dynamic_scale_factor_batch`.

---

### ⚠️ MODERATE: GMR Extinction Semantic Mismatch (Formula 3-12)

**Paper:** "GMR = 1 - CR, which gradually increase with the reduction of CR" — GMR is described as a **replacement rate** (proportion of population replaced), applied as a **generational mutation mechanism** each generation.

**Implementation (`extinction.py::should_extinct()`):**
```python
def should_extinct(cr, delta=0.3, rng=None):
    rate = gmr_rate(cr, delta)
    if rate <= 0:
        return False
    return rng.random() < rate  # Probabilistic trigger
```

The code interprets GMR as a **probability of triggering extinction at all**, rather than as the **fraction of population to replace**. Two issues:
1. **Probabilistic vs. deterministic:** The paper applies GMR deterministically every generation; the code applies it probabilistically (may not trigger even when GMR > 0).
2. **Replacement ratio vs. trigger probability:** When GMR = 0.7, the paper means "replace 70% of the population"; the code means "70% chance of replacing some population."

**Impact:** Moderate. This changes the extinction dynamics significantly. In early generations (high CR, low GMR), the paper still applies small-scale extinction; the code does nothing.

**Fix:** Consider making extinction deterministic (always apply when GMR > 0) and using GMR as the replacement ratio in `apply_extinction()`.

---

### ⚠️ MODERATE: Inverse Mapper Extensions Not in Paper

The `inverse_mapper.py` adds several features not described in Ming (2017):
1. **Temperature-adaptive matching:** `nearest_match_adaptive()` uses softmax sampling with temperature, while the paper specifies pure nearest-distance matching (Eq. 15).
2. **Post-match perturbation:** `_perturb_genes()` swaps/reassigns genes after matching, not described in the paper.
3. **SRP tour perturbation:** `_perturb_srp_tour()` applies swap/insert/reverse to cruise sequences.

These are reasonable enhancements for the LLM integration context (the temperature parameter enables LLM-controlled exploration), but they should be clearly documented as deviations from the base DMDE algorithm.

---

### ⚠️ MINOR: Operator Selection Interval at Gen=0

In `solver.py`:
```python
if op_module and op_module.enabled and gen % op_module.interval == 0:
```

When `gen = interval` (e.g., 50), this triggers the first LLM call. However, the trajectory only contains entries from gen 1 to gen 49 at that point. The paper's Algorithm 1 (line 4) says "for every p iterations", starting after p iterations have completed. The code correctly waits for p iterations before the first LLM call. ✅ Correct.

---

### ⚠️ MINOR: CR Control Interval (every 10 generations) vs. Paper

The paper triggers LLM operator selection every p=10 iterations. The code has two separate intervals:
- `operator_selection`: interval=50 (every 50 generations)
- `cr_control`: interval=10 (every 10 generations)

The `cr_control` module at interval=10 is more frequent than the paper's design for operator selection. This is a reasonable design choice (CR control needs more frequent adjustment than strategy selection) but is a deviation from the paper's single-interval design.

---

### ⚠️ MINOR: Feature Set Differences from Zhang Table 2

Zhang (2025) Table 2 defines specific features:
| Paper Feature | Code Equivalent | Status |
|--------------|-----------------|--------|
| f1: Average fitness | `mean_fitness` in ModuleState | ✅ Present |
| f2: Average crowding distance | Not implemented | ❌ Missing (DMDE is single-objective, no crowding distance) |
| f3: Progress 1-i/T | `generation/max_generations` (computed from state) | ✅ Derivable |
| f4: Improvement rate | `convergence_speed` | ✅ Equivalent |
| f5: Avg std of objectives | `gene_variance` | ✅ Approximate |
| f6: Current operators | `extra["current_strategy"]` | ✅ Present |
| f7: Best solution | `best_fitness` | ✅ Present |

The missing f2 (crowding distance) is justified because DMDE is single-objective while Zhang's paper addresses multi-objective NSGA-II. The code substitutes with diversity metrics, which is reasonable.

---

## Missing Features

1. **LLM Temperature Control (AI Guide §3.2):** The temperature parameter exists in `inverse_mapper.py` but is controlled by a fixed linear formula (`1.0 - gen/max_gen`), not by LLM decision. The AI guide suggests LLM should adaptively control temperature based on feasible ratio, constraint violation trends, etc.

2. **LLM Extinction Control (AI Guide §3.3):** No LLM module decides whether to trigger extinction or how much population to replace. The current implementation uses a fixed δ threshold.

3. **LLM Constraint Handling (AI Guide §3.4):** No adaptive penalty coefficient adjustment. This is partially outside the module's scope (penalty is in the fitness evaluator), but the LLM could influence it.

4. **LLM SRP Tour Repair (AI Guide §3.5):** The SRP tour uses greedy nearest-neighbor + random perturbation. The AI guide suggests LLM could provide specific reorderings based on understanding of sequence constraints.

5. **Dynamic F Integration:** The `dynamic_scale_factor` function exists but is never called from the solver loop. This is a bug, not a missing feature.

---

## Summary of Findings

| Category | Count | Severity |
|----------|-------|----------|
| Critical bugs | 1 | Dynamic F formula (3-11) implemented but unused |
| Semantic mismatches | 1 | GMR extinction interpretation |
| Paper deviations (enhancements) | 3 | Temperature adaptation, post-match perturbation, SRP perturbation |
| Missing LLM enhancements | 4 | Temperature control, extinction control, constraint handling, SRP repair |
| Minor discrepancies | 2 | Feature set adaptation, interval differences |

### Overall Assessment

The DMDE core algorithm is **largely faithful** to Ming (2017). The encoding rules (3.1/3.2/3.3), mapping (Eq. 12/14), matching rules (3.4/3.5/3.6), dynamic CR (Eq. 16), hybrid differential (Eq. 17), and GMR formula (Eq. 19/3-12) are all correctly implemented at the formula level.

The **most critical issue** is that Formula 3-11 (dynamic F) is correctly coded in `scale_factor.py` but **never called from the solver**, which breaks the paper's CR-F coupling — a key design feature of DMDE.

The LLM enhancement follows Zhang (2025)'s framework (adaptive operator selection, trajectory-based prompting, feature abstraction) with reasonable adaptations for the single-objective DMDE context. However, only 1 of 5 DMDE-specific enhancement opportunities identified in the AI guide (§3.1-3.5) has been implemented.

### Recommended Actions

1. **[CRITICAL]** Fix `solver.py` to call `dynamic_scale_factor(cr, rng)` instead of hardcoding `base_f = 0.5`
2. **[MODERATE]** Clarify GMR extinction semantics — document whether probabilistic or deterministic interpretation is intended
3. **[MODERATE]** Document temperature adaptation and perturbation as paper extensions, not base DMDE
4. **[LOW]** Consider implementing LLM temperature control and extinction control modules as identified in the AI guide
5. **[LOW]** Add `f_scale` override support in solver to allow LLM CR control module to also influence F dynamically
