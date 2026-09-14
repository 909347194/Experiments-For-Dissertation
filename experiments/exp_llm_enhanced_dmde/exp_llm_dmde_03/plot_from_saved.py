# -*- coding: utf-8 -*-
"""plot_from_saved.py — 读取已保存的 exp_llm_dmde_03 数据重新绘图

背景：
    ``run.py`` 每次运行需完成 LLM 增强 DMDE 求解，并把绘图所需的全部数据
    （含 LLM 决策日志）保存为 ``results/exp_llm_dmde_03_data.json``。
    本脚本只负责：加载该 JSON → 重建可视化对象 → 调用 visualizer 出图，
    因此调整绘图样式时无需重跑实验。

用法：
    # 默认：加载 results/exp_llm_dmde_03_data.json
    python plot_from_saved.py

    # 指定数据 / 输出目录
    python plot_from_saved.py --data path/to/data.json --out path/to/figures

    # 跳过 DEM 加载（最快，仅二维图）
    python plot_from_saved.py --no-3d
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── 路径设置（与 run.py 保持一致）───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_DATA_FILE = RESULTS_DIR / "exp_llm_dmde_03_data.json"
DEFAULT_OUT_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(SRC_ROOT))
# 复用 exp_llm_dmde_01 的 visualization / data_store
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01"))

# Windows 下控制台/重定向管道默认使用 GBK
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(errors="replace")

DEM_FILE = DATA_DIR / "chengguan_district_dem.tif"
if not DEM_FILE.exists():
    # 回退：复用对应 DMDE 基线实验（exp_dmde_03）的 DEM 栅格；
    # 若本目录存在 data/*.tif 则优先使用它。
    _fallback_dem = (PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_03"
                     / "data" / "chengguan_district_dem.tif")
    if _fallback_dem.exists():
        DEM_FILE = _fallback_dem


def _load_dem():
    """重载 DEM 地形（开销小，不涉及求解）。"""
    from environments.environment_dmde import DEMTerrain

    t0 = time.time()
    dem = DEMTerrain.from_file(DEM_FILE)
    print(f"    DEM: {dem.meta.width}×{dem.meta.height}, "
          f"bounds={[f'{b:.3f}' for b in dem.bounds]} ({time.time() - t0:.2f}s)")
    return dem


def main() -> None:
    ap = argparse.ArgumentParser(
        description="读取已保存 LLM 增强实验数据并重新生成图表（不重跑实验）"
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA_FILE),
                    help="全量实验数据 JSON 路径")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                    help="图表输出目录")
    ap.add_argument("--no-3d", action="store_true",
                    help="跳过 DEM 加载与三维图（最快，仅二维图）")
    args = ap.parse_args()

    t_start = time.time()

    # 1. 加载已保存数据
    print(f"[1] 加载已保存实验数据: {args.data}")
    from data_store import load_experiment

    payload = load_experiment(args.data)
    scenarios = payload["scenarios"]
    uavs_dict = payload["uavs_dict"]
    targets_dict = payload["targets_dict"]
    meta = payload.get("meta", {})
    llm_decisions = payload.get("llm_decisions")
    print(f"    场景: {[s['name'] for s in scenarios]}")
    print(f"    元信息: solver_params={meta.get('solver_params', {})}, "
          f"n_runs={meta.get('n_runs', '?')}, constraint={meta.get('constraint_desc', '—')}")
    if llm_decisions:
        total = sum(len(v) for v in llm_decisions.values())
        print(f"    LLM 决策记录: {total} 条（{len(llm_decisions)} 次运行）")

    # 2. 加载 DEM（默认，用于三维图）
    dem = None
    if not args.no_3d:
        print("[2] 加载 DEM 地形（用于三维图）...")
        dem = _load_dem()
    else:
        print("[2] 已跳过 DEM 加载（--no-3d）")

    # 3. 生成全部图表
    print("[3] 生成图表 ...")
    from visualization.visualizer import ExperimentVisualizer

    viz = ExperimentVisualizer(output_dir=args.out, algo_name="LLM-DMDE")
    saved_files = viz.plot_all(scenarios, uavs_dict, targets_dict, dem_terrain=dem,
                               llm_decisions=llm_decisions)

    print(f"\n[OK] 共生成 {len(saved_files)} 张图表")
    print(f"   图表输出目录: {args.out}")
    print(f"   本次绘图总耗时: {time.time() - t_start:.2f}s（未运行求解器）")
    for p in saved_files:
        print(f"   - {p.name}")


if __name__ == "__main__":
    main()
