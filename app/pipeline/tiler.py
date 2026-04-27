from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Generator

import numpy as np


@dataclass
class Tile:
    image: np.ndarray
    x: int
    y: int


def _prepare_source(path: str) -> tuple[str, bool]:
    """
    For JPEG/PNG inputs, stream-convert to a temporary tiled GeoTIFF.
    Reads the source format in GDAL blocks (never decodes the full image at once).
    Returns (tiff_path, needs_cleanup).
    TIFF inputs are passed through unchanged.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        return path, False

    import rasterio

    tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    tmp.close()

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        profile.update(
            driver='GTiff',
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512,
            interleave='band',
            photometric='rgb',
        )
        with rasterio.open(tmp.name, 'w', **profile) as dst:
            for _, window in src.block_windows(1):
                dst.write(src.read(window=window), window=window)

    return tmp.name, True


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        max_val = np.iinfo(arr.dtype).max
    else:
        max_val = float(arr.max()) or 1.0
    return (arr.astype(np.float32) / max_val * 255).clip(0, 255).astype(np.uint8)


def image_size(path: str) -> tuple[int, int]:
    """Return (height, width) by reading only file metadata."""
    import rasterio
    with rasterio.open(path) as src:
        return src.height, src.width


def count_tiles(path: str, tile_size: int, overlap: float) -> int:
    """Count total tiles without touching pixel data."""
    import rasterio
    step = max(1, int(tile_size * (1 - overlap)))
    with rasterio.open(path) as src:
        h, w = src.height, src.width

    def _n(total: int) -> int:
        n, pos = 0, 0
        while True:
            n += 1
            if min(pos + tile_size, total) >= total:
                break
            pos += step
        return n

    return _n(h) * _n(w)


def tile_generator(
    path: str,
    tile_size: int,
    overlap: float,
) -> Generator[Tile, None, None]:
    """
    Yield tiles by reading windows directly from disk.
    For JPEG/PNG: stream-converts to a temp tiled TIFF first (no full decode).
    Peak RAM ≈ batch_size × tile_size² × 3 bytes regardless of source file size.
    """
    import rasterio
    from rasterio.windows import Window

    step = max(1, int(tile_size * (1 - overlap)))
    tiff_path, is_temp = _prepare_source(path)

    try:
        with rasterio.open(tiff_path) as src:
            h, w = src.height, src.width
            n_bands = src.count

            y = 0
            while True:
                y2 = min(y + tile_size, h)
                x = 0
                while True:
                    x2 = min(x + tile_size, w)
                    data = src.read(window=Window(x, y, x2 - x, y2 - y))  # (bands, H, W)

                    if n_bands >= 3:
                        rgb = np.moveaxis(data[:3], 0, -1)
                    else:
                        rgb = np.stack([data[0]] * 3, axis=-1)

                    yield Tile(image=np.ascontiguousarray(_to_uint8(rgb)), x=x, y=y)

                    if x2 >= w:
                        break
                    x += step
                if y2 >= h:
                    break
                y += step
    finally:
        if is_temp:
            os.unlink(tiff_path)
