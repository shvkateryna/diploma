from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Generator

import numpy as np


@dataclass
class Tile:
    image: np.ndarray  # pixels fed to the model (resampled by `scale`)
    x: int             # top-left corner in source-image pixels
    y: int
    src_w: int         # window size in source-image pixels
    src_h: int
    scale: float = 1.0  # model-tile pixels per source pixel


def window_size(tile_size: int, scale: float) -> int:
    """
    Side of the source window (in source pixels) that becomes a `tile_size` tile.
    scale > 1 upsamples (coarse imagery, e.g. 120 m), scale < 1 downsamples.
    """
    return max(32, round(tile_size / scale))


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
        )
        # RGB photometric is only valid for 3+ band images (fails on greyscale)
        if src.count >= 3:
            profile['photometric'] = 'rgb'
        else:
            profile.pop('photometric', None)
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


def _positions(total: int, size: int, step: int) -> list[int]:
    """
    Tile start offsets along one axis. The last tile is shifted back to end exactly
    at the image edge instead of being cut short, so every tile is a full
    `size` square (RF-DETR stretches non-square tiles to a square, which distorts
    penguins in narrow edge strips). The shifted tile overlaps its neighbour by at
    least the normal overlap, and duplicates are removed by NMS as everywhere else.
    """
    if total <= size:
        return [0]
    pos = list(range(0, total - size, step))
    pos.append(total - size)
    return pos


def count_tiles(path: str, tile_size: int, overlap: float, scale: float = 1.0) -> int:
    """Count total tiles without touching pixel data."""
    import rasterio
    tile_size = window_size(tile_size, scale)
    step = max(1, int(tile_size * (1 - overlap)))
    with rasterio.open(path) as src:
        h, w = src.height, src.width
    return len(_positions(h, tile_size, step)) * len(_positions(w, tile_size, step))


def tile_generator(
    path: str,
    tile_size: int,
    overlap: float,
    scale: float = 1.0,
) -> Generator[Tile, None, None]:
    """
    Yield tiles by reading windows directly from disk.
    For JPEG/PNG: stream-converts to a temp tiled TIFF first (no full decode).
    Peak RAM ≈ batch_size × tile_size² × 3 bytes regardless of source file size.

    With scale != 1 each source window of `window_size(tile_size, scale)` pixels is
    resampled to `tile_size`, so objects appear at the size the model was trained on.
    All tiles are full squares (see `_positions`); only an image smaller than one
    window yields a smaller tile.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    resample = scale != 1.0
    tile_size = window_size(tile_size, scale)
    step = max(1, int(tile_size * (1 - overlap)))
    tiff_path, is_temp = _prepare_source(path)

    try:
        with rasterio.open(tiff_path) as src:
            h, w = src.height, src.width
            n_bands = src.count

            xs = _positions(w, tile_size, step)
            for y in _positions(h, tile_size, step):
                y2 = min(y + tile_size, h)
                for x in xs:
                    x2 = min(x + tile_size, w)
                    win_w, win_h = x2 - x, y2 - y
                    bands = list(range(1, min(3, n_bands) + 1))  # skip alpha
                    if resample:
                        out_shape = (
                            len(bands),
                            max(1, round(win_h * scale)),
                            max(1, round(win_w * scale)),
                        )
                        data = src.read(
                            bands,
                            window=Window(x, y, win_w, win_h),
                            out_shape=out_shape,
                            resampling=Resampling.bilinear,
                        )
                    else:
                        data = src.read(bands, window=Window(x, y, win_w, win_h))

                    if len(bands) >= 3:
                        rgb = np.moveaxis(data[:3], 0, -1)
                    else:
                        rgb = np.stack([data[0]] * 3, axis=-1)

                    yield Tile(
                        image=np.ascontiguousarray(_to_uint8(rgb)),
                        x=x, y=y, src_w=win_w, src_h=win_h, scale=scale,
                    )
    finally:
        if is_temp:
            os.unlink(tiff_path)
