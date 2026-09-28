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
    x1, y1, x2, y2 = box
    th, tw = tile.image.shape[:2]
    return (
        (x1 <= _EDGE_MARGIN and tile.x > 0)
        or (y1 <= _EDGE_MARGIN and tile.y > 0)
        or (x2 >= tw - _EDGE_MARGIN and tile.x + tw < img_w)
        or (y2 >= th - _EDGE_MARGIN and tile.y + th < img_h)
    )


def run(
    image_path: str,
    model,
    conf: float,
    tile_size: int,
    overlap: float,
    on_tile: Optional[Callable[[int, int], None]] = None,
    batch_size: int = 4,
) -> tuple[list, list]:
    total = count_tiles(image_path, tile_size, overlap)
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
                x1, y1, x2, y2 = box
                boxes.append((x1 + tile.x, y1 + tile.y, x2 + tile.x, y2 + tile.y))
                scores.append(score)
        done += len(batch)
        if on_tile:
            on_tile(done, total)
        batch.clear()

    for tile in tile_generator(image_path, tile_size, overlap):
        batch.append(tile)
        if len(batch) >= batch_size:
            _flush()
    _flush()

    return boxes, scores
