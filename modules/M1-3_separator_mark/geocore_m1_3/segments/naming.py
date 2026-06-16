from __future__ import annotations

import re


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def format_depth(depth_m: float) -> str:
    return f"{depth_m:07.2f}"


def segment_id(hole_id: str, depth_start_m: float, depth_end_m: float) -> str:
    return safe_id(f"{hole_id}_{format_depth(depth_start_m)}_{format_depth(depth_end_m)}")
