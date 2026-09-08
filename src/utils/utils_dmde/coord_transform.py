# -*- coding: utf-8 -*-
"""coord_transform.py — 坐标转换工具模块

职责：
    提供 WGS84 经纬度与投影平面坐标之间的双向转换，
    解决 DEM 地形模块中"度 vs 米"单位混用问题。

投影方案：
    UTM (Universal Transverse Mercator) —— 国际标准横轴墨卡托投影。
    按经度自动选择 UTM 分带（6°带），确保小范围（<1000km）内
    距离误差 < 0.1%。

依赖：
    pyproj (优先): 专业 GIS 投影库，精度最高。
    手动实现 (备选): 无外部依赖，UTM 投影精度足够实验使用。

使用方式：
    from utils.utils_dmde.coord_transform import WGS84Transformer

    transformer = WGS84Transformer.from_lonlat(91.1, 29.6)
    x, y = transformer.to_xy(91.3, 29.7)          # 经纬度 → 平面米
    lon, lat = transformer.to_lonlat(x, y)          # 平面米 → 经纬度
    d = transformer.distance(91.0, 29.5, 91.3, 29.7)  # 两点间距离(米)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── WGS84 椭球常量 ─────────────────────────────────────────
WGS84_A = 6378137.0           # 半长轴 (m)
WGS84_F = 1 / 298.257223563   # 扁率
WGS84_B = WGS84_A * (1 - WGS84_F)  # 半短轴 (m)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # 第一偏心率平方

# ── UTM 投影参数 ────────────────────────────────────────────
UTM_K0 = 0.9996               # 比例因子
UTM_E0 = 500000.0             # 东偏 (m)
UTM_N0_NORTH = 0.0            # 北半球北偏 (m)
UTM_N0_SOUTH = 10000000.0     # 南半球北偏 (m)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主类
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class UTMZone:
    """UTM 分带信息。"""
    zone: int        # 带号 (1-60)
    letter: str      # 纬度带字母 (C-X)
    central_meridian: float  # 中央经线 (度)
    is_north: bool   # 是否北半球


class WGS84Transformer:
    """WGS84 经纬度 ↔ UTM 平面坐标转换器。

    以某点为参考原点建立局部坐标系，支持：
    - 经纬度 → 局部平面坐标 (米)
    - 局部平面坐标 → 经纬度
    - 两点间距离计算
    - 批量坐标转换

    使用方式::

        # 方法 1: 自动检测 UTM 带
        transformer = WGS84Transformer.from_lonlat(91.1, 29.6)

        # 方法 2: 指定 UTM 带
        transformer = WGS84Transformer(utm_zone=46, is_north=True)

        # 转换
        x, y = transformer.to_xy(91.3, 29.7)
        lon, lat = transformer.to_lonlat(x, y)
        d = transformer.distance(91.0, 29.5, 91.3, 29.7)
    """

    def __init__(
        self,
        utm_zone: int | None = None,
        is_north: bool = True,
        ref_lon: float | None = None,
        ref_lat: float | None = None,
    ):
        """初始化转换器。

        Args:
            utm_zone: UTM 带号 (1-60)，为 None 时从 ref_lon 自动推算。
            is_north: 是否北半球。
            ref_lon: 参考原点经度（局部坐标系原点）。
            ref_lat: 参考原点纬度（局部坐标系原点）。
        """
        if utm_zone is None:
            if ref_lon is not None:
                utm_zone = _lon_to_utm_zone(ref_lon)
            else:
                utm_zone = 46  # 默认覆盖中国西部

        central_meridian = (utm_zone - 1) * 6 - 180 + 3

        object.__setattr__(self, '_utm_zone', UTMZone(
            zone=utm_zone,
            letter=_lat_to_utm_letter(ref_lat) if ref_lat is not None else 'N',
            central_meridian=central_meridian,
            is_north=is_north,
        ))
        object.__setattr__(self, '_ref_lon', ref_lon)
        object.__setattr__(self, '_ref_lat', ref_lat)

        # 参考点的投影坐标（作为局部原点）
        if ref_lon is not None and ref_lat is not None:
            e0, n0 = _wgs84_to_utm_raw(ref_lon, ref_lat, utm_zone, is_north)
            object.__setattr__(self, '_ref_e', e0)
            object.__setattr__(self, '_ref_n', n0)
        else:
            object.__setattr__(self, '_ref_e', 0.0)
            object.__setattr__(self, '_ref_n', 0.0)

        # 尝试加载 pyproj（精度更高）
        try:
            from pyproj import Transformer as ProjTransformer
            crs_from = "EPSG:4326"  # WGS84
            crs_to = f"EPSG:{32600 + utm_zone if is_north else 32700 + utm_zone}"
            object.__setattr__(self, '_proj_transformer',
                               ProjTransformer.from_crs(crs_from, crs_to, always_xy=True))
        except ImportError:
            object.__setattr__(self, '_proj_transformer', None)

    @classmethod
    def from_lonlat(cls, lon: float, lat: float) -> "WGS84Transformer":
        """从经纬度自动创建转换器（自动检测 UTM 带）。"""
        zone = _lon_to_utm_zone(lon)
        return cls(utm_zone=zone, is_north=lat >= 0, ref_lon=lon, ref_lat=lat)

    @classmethod
    def from_utm_zone(cls, zone: int, is_north: bool = True) -> "WGS84Transformer":
        """从指定 UTM 带创建转换器（无参考原点）。"""
        return cls(utm_zone=zone, is_north=is_north)

    # ── 属性 ──────────────────────────────────────────────

    @property
    def utm_zone(self) -> UTMZone:
        return self._utm_zone

    @property
    def has_reference(self) -> bool:
        return self._ref_lon is not None

    # ── 经纬度 → 平面坐标 ────────────────────────────────

    def to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        """经纬度 → 局部平面坐标 (米)。

        以参考点为原点，东向为 x，北向为 y。

        Args:
            lon: 经度 (度)。
            lat: 纬度 (度)。

        Returns:
            (x, y) 平面坐标 (米)。
        """
        if self._proj_transformer is not None:
            e, n = self._proj_transformer.transform(lon, lat)
        else:
            e, n = _wgs84_to_utm_raw(lon, lat, self._utm_zone.zone, self._utm_zone.is_north)
        return e - self._ref_e, n - self._ref_n

    def to_xy_batch(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量经纬度 → 局部平面坐标。

        Args:
            lons: 经度数组。
            lats: 纬度数组。

        Returns:
            (xs, ys) 平面坐标数组 (米)。
        """
        if self._proj_transformer is not None:
            e, n = self._proj_transformer.transform(lons, lats)
        else:
            e, n = _wgs84_to_utm_raw_batch(lons, lats, self._utm_zone.zone, self._utm_zone.is_north)
        return e - self._ref_e, n - self._ref_n

    # ── 平面坐标 → 经纬度 ────────────────────────────────

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """局部平面坐标 → 经纬度。

        Args:
            x: 东向坐标 (米)。
            y: 北向坐标 (米)。

        Returns:
            (lon, lat) 经纬度 (度)。
        """
        e = x + self._ref_e
        n = y + self._ref_n
        if self._proj_transformer is not None:
            lon, lat = self._proj_transformer.transform(e, n, direction='INVERSE')
        else:
            lon, lat = _utm_to_wgs84_raw(e, n, self._utm_zone.zone, self._utm_zone.is_north)
        return lon, lat

    def to_lonlat_batch(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量平面坐标 → 经纬度。"""
        e = xs + self._ref_e
        n = ys + self._ref_n
        if self._proj_transformer is not None:
            lon, lat = self._proj_transformer.transform(e, n, direction='INVERSE')
        else:
            lon, lat = _utm_to_wgs84_raw_batch(e, n, self._utm_zone.zone, self._utm_zone.is_north)
        return lon, lat

    # ── 距离计算 ─────────────────────────────────────────

    def distance(
        self, lon1: float, lat1: float, lon2: float, lat2: float
    ) -> float:
        """两点间投影距离 (米)。

        Args:
            lon1, lat1: 点1 经纬度。
            lon2, lat2: 点2 经纬度。

        Returns:
            平面距离 (米)。
        """
        x1, y1 = self.to_xy(lon1, lat1)
        x2, y2 = self.to_xy(lon2, lat2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def distance_3d(
        self, lon1: float, lat1: float, z1: float,
        lon2: float, lat2: float, z2: float,
    ) -> float:
        """两点间三维距离 (米)。

        Args:
            lon1, lat1, z1: 点1 经纬度 + 高程。
            lon2, lat2, z2: 点2 经纬度 + 高程。

        Returns:
            三维距离 (米)。
        """
        x1, y1 = self.to_xy(lon1, lat1)
        x2, y2 = self.to_xy(lon2, lat2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTM 投影手动实现（pyproj 不可用时的备选）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _lon_to_utm_zone(lon: float) -> int:
    """经度 → UTM 带号 (1-60)。"""
    return int((lon + 180) / 6) + 1


def _lat_to_utm_letter(lat: float) -> str:
    """纬度 → UTM 纬度带字母 (C-X)。"""
    letters = 'CDEFGHJKLMNPQRSTUVWX'
    if lat < -80 or lat > 84:
        return 'Z'
    idx = int((lat + 80) / 8)
    return letters[min(idx, len(letters) - 1)]


def _wgs84_to_utm_raw(
    lon: float, lat: float, zone: int, is_north: bool
) -> tuple[float, float]:
    """WGS84 → UTM 原始坐标（单点）。"""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    cm = math.radians((zone - 1) * 6 - 180 + 3)  # 中央经线

    e2 = WGS84_E2
    ep2 = e2 / (1 - e2)
    N = WGS84_A / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = ep2 * math.cos(lat_rad) ** 2
    A = math.cos(lat_rad) * (lon_rad - cm)

    # 子午线弧长
    M = WGS84_A * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * lat_rad)
    )

    easting = UTM_K0 * N * (
        A + (1 - T + C) * A ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120
    ) + UTM_E0

    northing = UTM_K0 * (
        M + N * math.tan(lat_rad) * (
            A ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720
        )
    )
    if not is_north:
        northing += UTM_N0_SOUTH

    return easting, northing


def _utm_to_wgs84_raw(
    easting: float, northing: float, zone: int, is_north: bool
) -> tuple[float, float]:
    """UTM → WGS84 经纬度（单点）。"""
    if not is_north:
        northing -= UTM_N0_SOUTH

    cm = math.radians((zone - 1) * 6 - 180 + 3)
    e2 = WGS84_E2
    ep2 = e2 / (1 - e2)
    M0 = 0.0  # 赤道子午线弧长 = 0
    M = M0 + northing / UTM_K0

    mu = M / (WGS84_A * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    J1 = (3 * e1 / 2 - 27 * e1 ** 3 / 32)
    J2 = (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32)
    J3 = (151 * e1 ** 3 / 96)
    J4 = (1097 * e1 ** 4 / 512)

    lat_rad = mu + J1 * math.sin(2 * mu) + J2 * math.sin(4 * mu) + J3 * math.sin(6 * mu) + J4 * math.sin(8 * mu)

    N = WGS84_A / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = ep2 * math.cos(lat_rad) ** 2
    R = WGS84_A * (1 - e2) / (1 - e2 * math.sin(lat_rad) ** 2) ** 1.5
    D = (easting - UTM_E0) / (N * UTM_K0)

    lat_rad2 = lat_rad - (N * math.tan(lat_rad) / R) * (
        D ** 2 / 2
        - (5 + 3 * T + 10 * C - 4 * C ** 2 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * T + 298 * C + 45 * T ** 2 - 252 * ep2 - 3 * C ** 2) * D ** 6 / 720
    )

    lon_rad = cm + (
        D - (1 + 2 * T + C) * D ** 3 / 6
        + (5 - 2 * C + 28 * T - 3 * C ** 2 + 8 * ep2 + 24 * T ** 2) * D ** 5 / 120
    ) / math.cos(lat_rad)

    return math.degrees(lon_rad), math.degrees(lat_rad2)


# ── 批量版本 ──────────────────────────────────────────────

def _wgs84_to_utm_raw_batch(
    lons: np.ndarray, lats: np.ndarray, zone: int, is_north: bool
) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 → UTM 批量转换。"""
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    shape = lons.shape
    lons_flat = lons.ravel()
    lats_flat = lats.ravel()

    e_result = np.empty_like(lons_flat)
    n_result = np.empty_like(lats_flat)

    for i in range(len(lons_flat)):
        e_result[i], n_result[i] = _wgs84_to_utm_raw(
            float(lons_flat[i]), float(lats_flat[i]), zone, is_north
        )

    return e_result.reshape(shape), n_result.reshape(shape)


def _utm_to_wgs84_raw_batch(
    eastings: np.ndarray, northings: np.ndarray, zone: int, is_north: bool
) -> tuple[np.ndarray, np.ndarray]:
    """UTM → WGS84 批量转换。"""
    eastings = np.asarray(eastings, dtype=np.float64)
    northings = np.asarray(northings, dtype=np.float64)
    shape = eastings.shape
    e_flat = eastings.ravel()
    n_flat = northings.ravel()

    lon_result = np.empty_like(e_flat)
    lat_result = np.empty_like(n_flat)

    for i in range(len(e_flat)):
        lon_result[i], lat_result[i] = _utm_to_wgs84_raw(
            float(e_flat[i]), float(n_flat[i]), zone, is_north
        )

    return lon_result.reshape(shape), lat_result.reshape(shape)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 便捷函数（无需创建对象）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def haversine_distance(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Haversine 大圆距离（米）。

    不依赖投影，直接用球面公式计算两点间最短距离。
    精度：~0.5%（对 <1000km 的距离）。

    Args:
        lon1, lat1: 点1 经纬度 (度)。
        lon2, lat2: 点2 经纬度 (度)。

    Returns:
        距离 (米)。
    """
    R = 6371000.0  # 地球平均半径 (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def degree_to_meters_at_lat(lat: float) -> tuple[float, float]:
    """给定纬度处，1°经度和1°纬度分别对应的米数。

    Args:
        lat: 纬度 (度)。

    Returns:
        (m_per_deg_lon, m_per_deg_lat) 每度对应的米数。
    """
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)
    return m_per_deg_lon, m_per_deg_lat
