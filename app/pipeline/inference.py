from __future__ import annotations

from typing import Callable, Optional

from pipeline.tiler import Tile, count_tiles, tile_generator


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
            for (x1, y1, x2, y2), score in zip(t_boxes, t_scores):
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
