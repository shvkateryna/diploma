from __future__ import annotations

from typing import Callable, Optional

from pipeline.tiler import Tile, count_tiles, image_size, tile_generator

# Detections within this many pixels of an inner tile edge are treated as cut off.
_EDGE_MARGIN = 2


def _touches_inner_edge(box: tuple, tile: Tile, img_w: int, img_h: int) -> bool:
    """
    True if the box touches a tile edge that is not also an image edge.
    Such boxes are usually partial penguins cut by the tile border; since tiles
    overlap by more than a penguin's size, the neighbouring tile sees the whole
    bird, so dropping the partial box avoids counting the same penguin twice.
    """
    x1, y1, x2, y2 = box  # in model-tile pixels
    th, tw = tile.image.shape[:2]
    return (
        (x1 <= _EDGE_MARGIN and tile.x > 0)
        or (y1 <= _EDGE_MARGIN and tile.y > 0)
        or (x2 >= tw - _EDGE_MARGIN and tile.x + tile.src_w < img_w)
        or (y2 >= th - _EDGE_MARGIN and tile.y + tile.src_h < img_h)
    )


def run(
    image_path: str,
    model,
    conf: float,
    tile_size: int,
    overlap: float,
    on_tile: Optional[Callable[[int, int], None]] = None,
    batch_size: int = 4,
    scale: float = 1.0,
) -> tuple[list, list]:
    """
    Detect objects tile by tile. `scale` resamples the image (model-tile pixels per
    source pixel) so objects match the training size; boxes are returned in
    source-image pixels.
    """
    total = count_tiles(image_path, tile_size, overlap, scale)
    img_h, img_w = image_size(image_path)
    boxes: list[tuple] = []
    scores: list[float] = []
    done = 0
    batch: list[Tile] = []

    def _flush() -> None:
        nonlocal done
        if not batch:
            return
        results = model.predict_batch([t.image for t in batch], conf)
        for tile, (t_boxes, t_scores) in zip(batch, results):
            for box, score in zip(t_boxes, t_scores):
                if _touches_inner_edge(box, tile, img_w, img_h):
                    continue
                x1, y1, x2, y2 = (v / tile.scale for v in box)
                boxes.append((
                    round(x1) + tile.x, round(y1) + tile.y,
                    round(x2) + tile.x, round(y2) + tile.y,
                ))
                scores.append(score)
        done += len(batch)
        if on_tile:
            on_tile(done, total)
        batch.clear()

    for tile in tile_generator(image_path, tile_size, overlap, scale):
        batch.append(tile)
        if len(batch) >= batch_size:
            _flush()
    _flush()

    return boxes, scores
