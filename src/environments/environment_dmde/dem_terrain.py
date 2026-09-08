# -*- coding: utf-8 -*-
"""dem_terrain.py — DEM 高程地图加载与剖面提取

职责：
    本模块负责加载 DEM (Digital Elevation Model) 栅格高程数据，
    并提供沿任意两点连线提取地形剖面的功能，为垂直切面航程代价
    估算提供地形数据支撑。

对应论文：
    第 2 章 2.4.1 节 —— 基于垂直切面的航程代价估算方法。
    DEM 数据用于描述仿真环境的地貌形态，是三维航程代价估算
    的基础数据来源。

主要功能：
    1. 加载 GeoTIFF 格式的 DEM 文件，获取高程矩阵、地理范围、
       分辨率、坐标参考系等元信息。
    2. 在 DEM 上沿两点连线进行双线性插值采样，提取地形剖面。
    3. 提供坐标（经纬度 / 平面坐标）到像素索引的转换。
    4. 获取指定点的高程值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import rasterio
    from rasterio.transform import rowcol
except ImportError:
    rasterio = None  # type: ignore[assignment]
    rowcol = None  # type: ignore[assignment]

try:
    from utils.utils_dmde.coord_transform import degree_to_meters_at_lat
except ImportError:
    degree_to_meters_at_lat = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DEMProfile:
    """沿两点连线提取的地形剖面结果。

    Attributes:
        distances:  沿连线方向的累计距离序列（单位：与 DEM 坐标一致，通常为米）。
        elevations: 对应位置的高程序列。
        coords_xy:  剖面采样点的 (x, y) 坐标，shape = (n, 2)。
    """

    distances: np.ndarray   # shape (n,)
    elevations: np.ndarray  # shape (n,)
    coords_xy: np.ndarray   # shape (n, 2)


@dataclass(frozen=True)
class DEMMeta:
    """DEM 元信息快照。

    Attributes:
        width:       栅格列数。
        height:      栅格行数。
        resolution:  (x 方向分辨率, y 方向分辨率)。
        bounds:      (left, bottom, right, top) 地理范围。
        crs:         坐标参考系（字符串表示）。
        nodata:      无效值标记。
        transform:   仿射变换矩阵。
    """

    width: int
    height: int
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    crs: str
    nodata: float | None
    transform: object  # rasterio.Affine


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class DEMTerrain:
    """DEM 地形管理器。

    支持从 GeoTIFF 文件加载 DEM 数据，并提供地形剖面提取、
    高程查询等接口。

    使用方式::

        dem = DEMTerrain.from_file("path/to/dem.tif")
        profile = dem.extract_profile(start=(x1, y1), end=(x2, y2), num_samples=200)
        elev = dem.query_elevation(x=91.0, y=29.6)
    """

    def __init__(
        self,
        elevation: np.ndarray,
        meta: DEMMeta,
    ) -> None:
        self._elevation = elevation
        self._meta = meta

    # ---- 工厂方法 ----------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "DEMTerrain":
        """从 GeoTIFF 文件加载 DEM。

        Args:
            path: DEM 文件路径（支持 .tif / .tiff）。

        Returns:
            DEMTerrain 实例。

        Raises:
            ImportError:  未安装 rasterio。
            FileNotFoundError: 文件不存在。
            ValueError: 文件无法正确读取。
        """
        if rasterio is None:
            raise ImportError(
                "rasterio is required to load DEM files. "
                "Install it with: pip install rasterio"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DEM file not found: {path}")

        with rasterio.open(str(path)) as src:
            elevation = src.read(1).astype(np.float64)
            nodata = src.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan

            meta = DEMMeta(
                width=src.width,
                height=src.height,
                resolution=(src.res[0], src.res[1]),
                bounds=(
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                ),
                crs=str(src.crs) if src.crs else "unknown",
                nodata=nodata,
                transform=src.transform,
            )

        return cls(elevation=elevation, meta=meta)

    @classmethod
    def from_array(
        cls,
        elevation: np.ndarray,
        bounds: tuple[float, float, float, float],
        crs: str = "unknown",
        nodata: float | None = None,
    ) -> "DEMTerrain":
        """从 NumPy 数组和地理范围直接构造（无需 rasterio）。

        Args:
            elevation: 二维高程数组，shape = (height, width)。
            bounds: (left, bottom, right, top) 地理范围。
            crs: 坐标参考系字符串。
            nodata: 无效值标记。

        Returns:
            DEMTerrain 实例。
        """
        if elevation.ndim != 2:
            raise ValueError(f"elevation must be 2-D, got shape {elevation.shape}")

        height, width = elevation.shape
        left, bottom, right, top = bounds
        res_x = (right - left) / width
        res_y = (top - bottom) / height

        if rasterio is not None:
            from rasterio.transform import from_bounds
            transform = from_bounds(left, bottom, right, top, width, height)
        else:
            transform = None

        elev = elevation.astype(np.float64).copy()
        if nodata is not None:
            elev[elev == nodata] = np.nan

        meta = DEMMeta(
            width=width,
            height=height,
            resolution=(res_x, res_y),
            bounds=bounds,
            crs=crs,
            nodata=nodata,
            transform=transform,
        )
        return cls(elevation=elev, meta=meta)

    # ---- 属性 --------------------------------------------------------------

    @property
    def elevation(self) -> np.ndarray:
        """高程矩阵，shape = (height, width)，无效值为 NaN。"""
        return self._elevation

    @property
    def meta(self) -> DEMMeta:
        """DEM 元信息。"""
        return self._meta

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) 地理范围。"""
        return self._meta.bounds

    # ---- 高程查询 ----------------------------------------------------------

    def query_elevation(self, x: float, y: float) -> float:
        """查询指定坐标处的高程（双线性插值）。

        Args:
            x: 横坐标（经度 / 东向）。
            y: 纵坐标（纬度 / 北向）。

        Returns:
            高程值。若坐标越界或对应像元为 NaN，返回 NaN。
        """
        left, bottom, right, top = self._meta.bounds
        res_x, res_y = self._meta.resolution

        # 转换为浮点像素坐标（行列号）
        col_f = (x - left) / res_x
        row_f = (top - y) / res_y  # 注意：影像行号向下递增

        if col_f < 0 or col_f >= self._meta.width - 1:
            return np.nan
        if row_f < 0 or row_f >= self._meta.height - 1:
            return np.nan

        # 双线性插值
        row0 = int(np.floor(row_f))
        col0 = int(np.floor(col_f))
        dy = row_f - row0
        dx = col_f - col0

        z00 = self._elevation[row0, col0]
        z01 = self._elevation[row0, col0 + 1]
        z10 = self._elevation[row0 + 1, col0]
        z11 = self._elevation[row0 + 1, col0 + 1]

        vals = np.array([z00, z01, z10, z11])
        if np.any(np.isnan(vals)):
            return np.nan

        return float(
            z00 * (1 - dx) * (1 - dy)
            + z01 * dx * (1 - dy)
            + z10 * (1 - dx) * dy
            + z11 * dx * dy
        )

    def query_elevations_batch(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> np.ndarray:
        """批量查询高程（矢量化双线性插值）。

        Args:
            xs: 横坐标数组。
            ys: 纵坐标数组。

        Returns:
            高程数组，shape 与 xs 相同。越界或无效位置为 NaN。
        """
        left, bottom, right, top = self._meta.bounds
        res_x, res_y = self._meta.resolution

        col_f = (xs - left) / res_x
        row_f = (top - ys) / res_y

        h, w = self._elevation.shape
        mask = (col_f < 0) | (col_f >= w - 1) | (row_f < 0) | (row_f >= h - 1)

        col_f = np.clip(col_f, 0, w - 2)
        row_f = np.clip(row_f, 0, h - 2)

        row0 = np.floor(row_f).astype(int)
        col0 = np.floor(col_f).astype(int)
        dy = row_f - row0
        dx = col_f - col0

        z00 = self._elevation[row0, col0]
        z01 = self._elevation[row0, col0 + 1]
        z10 = self._elevation[row0 + 1, col0]
        z11 = self._elevation[row0 + 1, col0 + 1]

        result = (
            z00 * (1 - dx) * (1 - dy)
            + z01 * dx * (1 - dy)
            + z10 * (1 - dx) * dy
            + z11 * dx * dy
        )

        any_nan = np.isnan(z00) | np.isnan(z01) | np.isnan(z10) | np.isnan(z11)
        result[any_nan | mask] = np.nan
        return result

    # ---- 地形剖面提取 -------------------------------------------------------

    def extract_profile(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        num_samples: int = 200,
    ) -> DEMProfile:
        """沿两点连线提取地形剖面。

        对应论文 Step1 ~ Step3：取初始点 S 和目标点 T 的连线，
        在连线上等间距采样，获取各采样点的高程值。

        Args:
            start:       起点 (x, y)。
            end:         终点 (x, y)。
            num_samples: 采样点数量（≥ 2）。

        Returns:
            DEMProfile 实例。
        """
        if num_samples < 2:
            raise ValueError("num_samples must be >= 2")

        sx, sy = start
        ex, ey = end

        ts = np.linspace(0.0, 1.0, num_samples)
        xs = sx + (ex - sx) * ts
        ys = sy + (ey - sy) * ts

        elevations = self.query_elevations_batch(xs, ys)

        # 计算累计距离（三维距离，经纬度转米）
        dx_deg = np.diff(xs)   # 经度差 (度)
        dy_deg = np.diff(ys)   # 纬度差 (度)
        dz = np.diff(elevations)  # 高程差 (米)
        dz = np.where(np.isnan(dz), 0.0, dz)

        # 经纬度差 → 米（用剖面平均纬度处的换算系数）
        if degree_to_meters_at_lat is not None:
            mean_lat = float(np.nanmean(ys))
            m_per_deg_lon, m_per_deg_lat = degree_to_meters_at_lat(mean_lat)
        else:
            # 降级：近似 1° ≈ 111km
            m_per_deg_lon = 111320.0
            m_per_deg_lat = 110540.0

        dx_m = dx_deg * m_per_deg_lon
        dy_m = dy_deg * m_per_deg_lat
        seg_lengths = np.sqrt(dx_m**2 + dy_m**2 + dz**2)
        distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        coords_xy = np.column_stack([xs, ys])

        return DEMProfile(
            distances=distances,
            elevations=elevations,
            coords_xy=coords_xy,
        )

    # ---- 可视化辅助 ---------------------------------------------------------

    def plot_terrain(
        self,
        ax=None,
        cmap: str = "terrain",
        alpha: float = 1.0,
    ):
        """绘制 DEM 高程热力图（需要 matplotlib）。

        Args:
            ax:   matplotlib Axes 对象，为 None 时自动创建。
            cmap: 颜色映射名称。
            alpha: 透明度。

        Returns:
            (fig, ax) 元组。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        left, bottom, right, top = self._meta.bounds
        extent = [left, right, bottom, top]

        elev = self._elevation.copy()
        elev[np.isnan(elev)] = -9999

        im = ax.imshow(
            elev,
            extent=extent,
            origin="upper",
            cmap=cmap,
            alpha=alpha,
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        ax.set_xlabel("X (lon / east)")
        ax.set_ylabel("Y (lat / north)")
        ax.set_title("DEM Terrain")

        return fig, ax
