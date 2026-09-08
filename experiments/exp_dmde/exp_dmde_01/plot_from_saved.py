# -*- coding: utf-8 -*-
"""plot_from_saved.py — 读取已保存的实验数据重新绘图（无需重跑实验）

背景：
    ``run.py`` 每次运行需要完成 3 个场景 × N_RUNS 次 DMDE 求解（默认 5 次，
    全量约 20 分钟）。它会把绘图所需的全部数据保存为
    ``results/exp_dmde_01_data.json``（见 data_store.py）。
    本脚本只负责：加载该 JSON → 重建可视化所需对象 → 调用 visualizer 出图。

    因此调整绘图样式时，只需修改 ``visualization/visualizer.py`` 后重新运行
    本脚本（秒级），不必再重跑实验。

用法：
    # 默认：加载 results/exp_dmde_01_data.json，含 DEM 三维图
    python experiments/exp_dmde/exp_dmde_01/plot_from_saved.py

    # 指定数据 / 输出目录
    python experiments/exp_dmde/exp_dmde_01/plot_from_saved.py \
        --data path/to/exp_dmde_01_data.json --out path/to/figures

    # 跳过 DEM 加载（最快，仅二维图）
    python experiments/exp_dmde/exp_dmde_01/plot_from_saved.py --no-3d
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── 路径设置（与 run.py 保持一致）───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Experiments-For-Dissertation
SRC_ROOT = PROJECT_ROOT / "src"
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_DATA_FILE = RESULTS_DIR / "exp_dmde_01_data.json"
DEFAULT_OUT_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(EXP_DIR))

# DEM 文件（仅 3D 图需要；与 run.py 中常量一致）
DEM_FILE = DATA_DIR / "ASTGTMV003_N29E091" / "ASTGTMV003_N29E091_dem.tif"


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
        description="读取已保存实验数据并重新生成图表（不重跑实验）"
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
    n_runs = meta.get("n_runs", "?")
    print(f"    场景: {[s['name'] for s in scenarios]}")
    print(f"    元信息: solver_params={meta.get('solver_params', {})}, n_runs={n_runs} "
          f"({time.time() - t_start:.2f}s)")

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

    viz = ExperimentVisualizer(output_dir=args.out)
    saved_files = viz.plot_all(scenarios, uavs_dict, targets_dict, dem_terrain=dem)

    print(f"\n✅ 共生成 {len(saved_files)} 张图表")
    print(f"   图表输出目录: {args.out}")
    print(f"   本次绘图总耗时: {time.time() - t_start:.2f}s（未运行求解器）")
    for p in saved_files:
        print(f"   - {p.name}")


if __name__ == "__main__":
    main()
