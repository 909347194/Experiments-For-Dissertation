# S2 验证：search_controller 决策协议修复

本目录下的脚本用于复现论文 S2 消融中 **search_controller 决策协议修复** 的验证结果。

## 修复内容（commit `2d8cca8`）

原协议把约 2/3 的决策浪费在 `recover`（强制全局灭绝，保留最优 30%、重置 70% 种群）上，
且 `explore` 因 `presets.py` 与 `search_controller.py` 定义不一致（`gmr` 实际为 `on`）也在强制重置，
反复打断收敛，既拖慢速度也停在次优盆地。

修复（3 个文件，+71/−8 行）：

1. `search_controller.py`：`STRATEGIES['explore'].gmr` 由 `on` 改回 `off`（与 `presets.py` 一致）。
2. `search_controller.py`：新增 `screen_strategy` 护栏——`recover` 仅在**真正绝境**
   （`delta_diversity < -0.06` 且 `stagnation_raw ≥ 60` 且近 3 stage 未用过）才放行，否则降级为 `exploit`。
3. `prompts/__init__.py`：重写 “When to Switch”，收敛态默认 `exploit`/`hold`，`recover` 仅最后手段。
4. `solvers/...py`：决策应用前调用 `screen_strategy`（3 行）。

## 复现步骤

### 1. 跑实验

在仓库根目录执行（Python 3.12 环境用 `uv run`）：

```bash
# 沙箱（缺 3.12）用系统 3.11 直接跑：
export LLM_MODEL=Qwen/Qwen3.5-27B
python3.11 experiments/exp_ablation_study/run_ablation.py \
    --scenario S2 --model Qwen/Qwen3.5-27B --runs 3

# 本地标准方式：
uv run python experiments/exp_ablation_study/run_ablation.py \
    --scenario S2 --model Qwen/Qwen3.5-27B --runs 3
```

> **关于模型名**：`Qwen/Qwen3.8-27B` 在 siliconflow 上**不存在**，这是 `Qwen/Qwen3.5-27B` 的笔误。
> Qwen3.x 系列中唯一的 27B 模型是 `Qwen/Qwen3.5-27B`（另有 14B/32B/8B，以及 3.5 的 4B/35B-A3B/122B-A10B）。
> 若你确有别的意图，改成对应可用模型名即可。

> **注意**：`run_ablation.py` 会把结果写回各配置的 `results/ablation_results.json`，**覆盖**已有结果。
> 切换模型重跑前请先备份旧 JSON（它们在 `.gitignore` 中，不入库）。

### 2. 跑验证分析

```bash
python3.11 experiments/exp_ablation_study/validation/validate_fix.py
python3.11 experiments/exp_ablation_study/validation/make_figures.py
```

- `validate_fix.py` 输出 A0（基线）vs A1（`A1_cr_control`，修复后）的终值统计、配对种子胜率、
  收敛代数、以及 A1 决策画像（策略分布 / 每代 gmr_mode 分布，用于证明全局重置已消除），
  并保存 `summary.json`。
- `make_figures.py` 输出 `s2_fix_validation.png`（收敛曲线 / 终值箱线图 / 决策画像）。

可用参数（默认指向 `S2_srp_N10_M20`）：

```
--scenario-dir <dir>   # 含 A0_dmde / A1_cr_control 子目录的场景目录
--a0 A0_dmde           # 基线配置目录名
--a1 A1_cr_control     # 修复后配置目录名
--out <dir>            # 输出目录
```

## 结果解读（截至 runs=3, Qwen3-14B）

- **全局重置被彻底消除**：修复后全 500 代 0 次强制 `gmr=on`；`recover` 被护栏降级。
- **稳定性**：A1 std 较 A0 降低约一个数量级。
- **终值 / 收敛**：典型种子上 A1 优于 A0（median 略低），但 A0 的 seed42 是幸运离群点。

## ⚠️ 统计功效警示

`runs=3` 且 LLM `temperature=0.7` **不足以得出确定性结论**——同种子两次独立运行可相差数万。
要做论文级检验，请用 `--runs 30` 同时重跑 A0 与 A1，并用配对 t 检验 / Wilcoxon 检验：

```bash
python3.11 experiments/exp_ablation_study/run_ablation.py \
    --scenario S2 --model Qwen/Qwen3.5-27B --runs 30
```
