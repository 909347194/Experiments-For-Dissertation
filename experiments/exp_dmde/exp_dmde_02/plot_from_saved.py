# -*- coding: utf-8 -*-
"""plot_from_saved.py — 读取已保存的 exp_dmde_02 数据重新绘图

用法：
    python plot_from_saved.py                        # 默认
    python plot_from_saved.py --no-3d                # 跳过 DEM 三维图
    python plot_from_saved.py --data path/to/file.json --out path/to/figures
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_DATA_FILE = RESULTS_DIR / "exp_dmde_02_data.json"
DEFAULT_OUT_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_01"))

DEM_FILE = DATA_DIR / "chengguan_district_dem.tif"


def _load_dem():
    from environments.environment_dmde import DEMTerrain
    t0 = time.time()
    dem = DEMTerrain.from_file(DEM_FILE)
    print(f"    DEM: {dem.meta.width}×{dem.meta.height}, "
          f"bounds={[f'{b:.3f}' for b in dem.bounds]} ({time.time()-t0:.2f}s)")
    return dem


def main():
    ap = argparse.ArgumentParser(description="exp_dmde_02 重新绘图")
    ap.add_argument("--data", default=str(DEFAULT_DATA_FILE))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--no-3d", action="store_true")
    args = ap.parse_args()

    t_start = time.time()

    print(f"[1] 加载数据: {args.data}")
    from data_store import load_experiment
    payload = load_experiment(args.data)
    scenarios = payload["scenarios"]
    uavs_dict = payload["uavs_dict"]
    targets_dict = payload["targets_dict"]
    print(f"    场景: {[s['name'] for s in scenarios]}")

    dem = None
    if not args.no_3d:
        print("[2] 加载 DEM...")
        dem = _load_dem()
    else:
        print("[2] 跳过 DEM (--no-3d)")

    print("[3] 生成图表...")
    from visualization.visualizer import ExperimentVisualizer
    viz = ExperimentVisualizer(output_dir=args.out)
    saved = viz.plot_all(scenarios, uavs_dict, targets_dict, dem_terrain=dem)

    print(f"\n完成: {len(saved)} 张图表, 耗时 {time.time()-t_start:.2f}s")
    for p in saved:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
