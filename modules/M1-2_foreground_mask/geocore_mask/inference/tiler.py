from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    size: int


def _axis_positions(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < 0.8:
        raise ValueError("overlap must be in [0, 0.8)")
    step = max(1, int(tile_size * (1 - overlap)))
    xs = _axis_positions(width, tile_size, step)
    ys = _axis_positions(height, tile_size, step)
    return [Tile(x=x, y=y, size=tile_size) for y in ys for x in xs]
