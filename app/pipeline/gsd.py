"""
Ground sampling distance (GSD): how many centimetres of ground one pixel covers.

The models were trained on 1024 px tiles at a fixed GSD (UAV photos from 80–100 m).
Images shot from higher up, or orthomosaics exported at a coarser resolution, show
penguins smaller in pixels, and the models miss them. Knowing the GSD of an input
image lets the tiler rescale each tile so penguins appear at the training size.
"""
from __future__ import annotations

import math
from typing import Optional

# Metres per degree of latitude/longitude at the equator (WGS84, good to <1%).
_M_PER_DEG_LAT = 110_574.0
_M_PER_DEG_LON = 111_320.0


def gsd_from_metadata(path: str) -> Optional[float]:
    """
    GSD in cm/px read from the georeferencing of a GeoTIFF (e.g. a Metashape
    orthomosaic). Returns None for images without usable georeferencing
    (plain JPEG/PNG photos).
    """
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        # Plain JPEG/PNG: expected, we just return None below
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        src = rasterio.open(path)
    with src:
        crs = src.crs
        if crs is None or src.transform.is_identity:
            return None
        res_x, res_y = abs(src.res[0]), abs(src.res[1])

        if crs.is_projected:
            # linear_units_factor -> (unit name, metres per unit)
            try:
                _, m_per_unit = crs.linear_units_factor
            except Exception:
                m_per_unit = 1.0
            return (res_x + res_y) / 2 * m_per_unit * 100

        if crs.is_geographic:
            # Pixel size is in degrees: convert at the image's centre latitude.
            lat = (src.bounds.top + src.bounds.bottom) / 2
            mx = res_x * _M_PER_DEG_LON * math.cos(math.radians(lat))
            my = res_y * _M_PER_DEG_LAT
            return (mx + my) / 2 * 100

    return None


def gsd_from_altitude(
    altitude_m: float,
    sensor_width_mm: float,
    focal_length_mm: float,
    image_width_px: int,
) -> float:
    """GSD in cm/px of a nadir photo taken from `altitude_m` with the given camera."""
    return altitude_m * sensor_width_mm / (focal_length_mm * image_width_px) * 100
