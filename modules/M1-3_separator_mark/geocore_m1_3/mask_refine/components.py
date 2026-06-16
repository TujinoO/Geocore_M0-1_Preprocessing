from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Run:
    y: int
    x0: int
    x1: int
    label: int


@dataclass
class Component:
    component_id: int
    area: int
    bbox: list[int]
    centroid: list[float]
    touches_border: bool
    runs: list[tuple[int, int, int]] = field(default_factory=list)

    def to_dict(self, include_runs: bool = False) -> dict:
        data = {
            "component_id": self.component_id,
            "area": self.area,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "touches_border": self.touches_border,
        }
        if include_runs:
            data["runs"] = self.runs
        return data


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def make(self) -> int:
        label = len(self.parent)
        self.parent.append(label)
        return label

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    if not row.any():
        return []
    padded = np.concatenate(([False], row, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def connected_components(mask: np.ndarray, connectivity: int = 8) -> list[Component]:
    if mask.ndim != 2:
        raise ValueError("connected_components expects a 2D mask.")
    binary = mask.astype(bool)
    height, width = binary.shape
    uf = UnionFind()
    all_runs: list[Run] = []
    prev_runs: list[Run] = []
    margin = 1 if connectivity == 8 else 0

    for y in range(height):
        current_runs: list[Run] = []
        prev_index = 0
        for x0, x1 in _row_runs(binary[y]):
            label = uf.make()
            while prev_index < len(prev_runs) and prev_runs[prev_index].x1 < x0 - margin:
                prev_index += 1
            scan_index = prev_index
            while scan_index < len(prev_runs) and prev_runs[scan_index].x0 <= x1 + margin:
                prev = prev_runs[scan_index]
                if prev.x1 >= x0 - margin and prev.x0 <= x1 + margin:
                    uf.union(label, prev.label)
                scan_index += 1
            run = Run(y=y, x0=x0, x1=x1, label=label)
            current_runs.append(run)
            all_runs.append(run)
        prev_runs = current_runs

    aggregates: dict[int, dict] = {}
    for run in all_runs:
        root = uf.find(run.label)
        length = run.x1 - run.x0
        x_sum = (run.x0 + run.x1 - 1) * length / 2.0
        y_sum = run.y * length
        if root not in aggregates:
            aggregates[root] = {
                "area": 0,
                "x_sum": 0.0,
                "y_sum": 0.0,
                "x0": run.x0,
                "x1": run.x1,
                "y0": run.y,
                "y1": run.y + 1,
                "runs": [],
            }
        agg = aggregates[root]
        agg["area"] += length
        agg["x_sum"] += x_sum
        agg["y_sum"] += y_sum
        agg["x0"] = min(agg["x0"], run.x0)
        agg["x1"] = max(agg["x1"], run.x1)
        agg["y0"] = min(agg["y0"], run.y)
        agg["y1"] = max(agg["y1"], run.y + 1)
        agg["runs"].append((run.y, run.x0, run.x1))

    components: list[Component] = []
    for new_id, (_, agg) in enumerate(sorted(aggregates.items(), key=lambda item: item[1]["area"], reverse=True), start=1):
        area = int(agg["area"])
        bbox = [int(agg["x0"]), int(agg["y0"]), int(agg["x1"]), int(agg["y1"])]
        centroid = [agg["x_sum"] / area, agg["y_sum"] / area]
        touches_border = bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= width or bbox[3] >= height
        components.append(
            Component(
                component_id=new_id,
                area=area,
                bbox=bbox,
                centroid=[float(centroid[0]), float(centroid[1])],
                touches_border=touches_border,
                runs=agg["runs"],
            )
        )
    return components


def components_to_mask(components: list[Component], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for component in components:
        for y, x0, x1 in component.runs:
            mask[y, x0:x1] = True
    return mask
