from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from geocore_m1_3.utils.geometry import interval_overlap


@dataclass
class DepthAnchor:
    depth_m: float
    strip_y: float


@dataclass
class MissingInterval:
    depth_start_m: float
    depth_end_m: float
    reason: str = "unknown"
    source: str = "manual"

    def to_dict(self) -> dict:
        return {
            "depth_start_m": self.depth_start_m,
            "depth_end_m": self.depth_end_m,
            "reason": self.reason,
            "source": self.source,
        }


class DepthMapper:
    def __init__(
        self,
        depth_start_m: float,
        depth_end_m: float,
        strip_height_px: int,
        anchors: Iterable[DepthAnchor] | None = None,
        missing_intervals: Iterable[MissingInterval] | None = None,
    ) -> None:
        if depth_end_m <= depth_start_m:
            raise ValueError("depth_end_m must be greater than depth_start_m.")
        if strip_height_px <= 0:
            raise ValueError("strip_height_px must be positive.")
        self.depth_start_m = float(depth_start_m)
        self.depth_end_m = float(depth_end_m)
        self.strip_height_px = int(strip_height_px)
        self.missing_intervals = list(missing_intervals or [])

        anchor_list = list(anchors or [])
        if anchor_list:
            anchor_list = sorted(anchor_list, key=lambda anchor: anchor.strip_y)
            if anchor_list[0].strip_y > 0:
                anchor_list.insert(0, DepthAnchor(self.depth_start_m, 0.0))
            if anchor_list[-1].strip_y < self.strip_height_px:
                anchor_list.append(DepthAnchor(self.depth_end_m, float(self.strip_height_px)))
        else:
            anchor_list = [
                DepthAnchor(self.depth_start_m, 0.0),
                DepthAnchor(self.depth_end_m, float(self.strip_height_px)),
            ]
        self.anchors = anchor_list

    @property
    def meters_per_pixel(self) -> float:
        return (self.depth_end_m - self.depth_start_m) / self.strip_height_px

    def y_to_depth(self, y: float) -> float:
        y = float(np.clip(y, 0, self.strip_height_px))
        for left, right in zip(self.anchors, self.anchors[1:]):
            if left.strip_y <= y <= right.strip_y:
                denom = max(1e-9, right.strip_y - left.strip_y)
                t = (y - left.strip_y) / denom
                return float(left.depth_m + t * (right.depth_m - left.depth_m))
        return self.depth_end_m

    def depth_to_y(self, depth_m: float) -> float:
        depth_m = float(np.clip(depth_m, self.depth_start_m, self.depth_end_m))
        anchors_by_depth = sorted(self.anchors, key=lambda anchor: anchor.depth_m)
        for left, right in zip(anchors_by_depth, anchors_by_depth[1:]):
            if left.depth_m <= depth_m <= right.depth_m:
                denom = max(1e-9, right.depth_m - left.depth_m)
                t = (depth_m - left.depth_m) / denom
                return float(left.strip_y + t * (right.strip_y - left.strip_y))
        return float(self.strip_height_px)

    def missing_ratio(self, depth_start_m: float, depth_end_m: float) -> float:
        length = max(1e-9, depth_end_m - depth_start_m)
        missing = 0.0
        for interval in self.missing_intervals:
            missing += interval_overlap(depth_start_m, depth_end_m, interval.depth_start_m, interval.depth_end_m)
        return float(min(1.0, missing / length))

    def to_dict(self) -> dict:
        return {
            "mode": "piecewise_linear" if len(self.anchors) > 2 else "linear",
            "depth_start_m": self.depth_start_m,
            "depth_end_m": self.depth_end_m,
            "strip_height_px": self.strip_height_px,
            "meters_per_pixel": self.meters_per_pixel,
            "anchors": [{"depth_m": anchor.depth_m, "strip_y": anchor.strip_y} for anchor in self.anchors],
            "missing_intervals": [interval.to_dict() for interval in self.missing_intervals],
        }


def parse_depth_anchors(items: list[dict] | None) -> list[DepthAnchor]:
    anchors: list[DepthAnchor] = []
    for item in items or []:
        if "strip_y" in item and "depth_m" in item:
            anchors.append(DepthAnchor(depth_m=float(item["depth_m"]), strip_y=float(item["strip_y"])))
    return anchors


def parse_missing_intervals(items: list[dict] | None) -> list[MissingInterval]:
    intervals: list[MissingInterval] = []
    for item in items or []:
        intervals.append(
            MissingInterval(
                depth_start_m=float(item["depth_start_m"]),
                depth_end_m=float(item["depth_end_m"]),
                reason=str(item.get("reason", "unknown")),
                source=str(item.get("source", "manual")),
            )
        )
    return intervals
