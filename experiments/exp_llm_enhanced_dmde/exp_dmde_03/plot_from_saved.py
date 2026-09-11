# -*- coding: utf-8 -*-
"""plot_from_saved.py — 读取已保存的 LLM 增强实验数据重新绘图 (N<M SRP)

用法：
    python plot_from_saved.py
    python plot_from_saved.py --data path/to/data.json --out path/to/figures
    python plot_from_saved.py --no-3d
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
DEFAULT_DATA_FILE = RESULTS_DIR / "exp_llm_dmde_03_data.json"
DEFAULT_OUT_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(EXP_DIR))

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(errors="replace")

DEM_FILE = DATA_DIR / "chengguan_district_dem.tif"


def _load_dem():
    from environments.environment_dmde import DEMTerrain

    t0 = time.time()
    dem = DEMTerrain.from_file(DEM_FILE)
    print(f"    DEM: {dem.meta.width}×{dem.meta.height}, "
          f"bounds={[f'{b:.3f}' for b in dem.bounds]} ({time.time() - t0:.2f}s)")
    return dem


def main() -> None:
    ap = argparse.ArgumentParser(
        description="读取已保存 LLM 增强实验数据并重新生成图表 (N<M SRP)"
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA_FILE),
                    help="全量实验数据 JSON 路径")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                    help="图表输出目录")
    ap.add_argument("--no-3d", action="store_true",
                    help="跳过 DEM 加载与三维图")
    args = ap.parse_args()

    t_start = time.time()

    print(f"[1] 加载已保存实验数据: {args.data}")
    from data_store import load_experiment

    payload = load_experiment(args.data)
    scenarios = payload["scenarios"]
    uavs_dict = payload["uavs_dict"]
    targets_dict = payload["targets_dict"]
    meta = payload.get("meta", {})
    llm_decisions = payload.get("llm_decisions")
    n_runs = meta.get("n_runs", "?")
    print(f"    场景: {[s['name'] for s in scenarios]}")
    print(f"    元信息: solver_params={meta.get('solver_params', {})}, n_runs={n_runs}")
    if llm_decisions:
        print(f"    LLM 决策记录: {len(llm_decisions)} 条")
    print(f"    ({time.time() - t_start:.2f}s)")

    dem = None
    if not args.no_3d:
        print("[2] 加载 DEM 地形（用于三维图）...")
        dem = _load_dem()
    else:
        print("[2] 已跳过 DEM 加载（--no-3d）")

    print("[3] 生成图表 ...")
    from visualization.visualizer import ExperimentVisualizer

    viz = ExperimentVisualizer(output_dir=args.out)
    saved_files = viz.plot_all(scenarios, uavs_dict, targets_dict, dem_terrain=dem)

    if llm_decisions:
        print("    [INFO] LLM 决策日志已加载，可扩展专用图表")

    print(f"\n[OK] 共生成 {len(saved_files)} 张图表")
    print(f"   图表输出目录: {args.out}")
    print(f"   本次绘图总耗时: {time.time() - t_start:.2f}s（未运行求解器）")
    for p in saved_files:
        print(f"   - {p.name}")


if __name__ == "__main__":
    main()
