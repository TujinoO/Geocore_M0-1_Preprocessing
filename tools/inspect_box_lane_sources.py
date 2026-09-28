"""Bounded-memory ENVI RGB inventory and visual sampling for M1-1/M1-3.

The output is diagnostic evidence, not ground-truth annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "M1-1_rotation_correction"))
from geocore_m1_1.config import M11Config  # noqa: E402
from geocore_m1_1.io_envi import parse_envi_header, read_envi_image  # noqa: E402
from geocore_m1_1.image_ops import make_thumbnail  # noqa: E402
from geocore_m1_1.segment_boxes import detect_core_boxes  # noqa: E402


def inspect_one(dat: Path, output: Path) -> dict:
    hdr = dat.with_suffix(".hdr")
    meta = parse_envi_header(hdr)
    h, w, bands = (int(meta[key]) for key in ("lines", "samples", "bands"))
    dtype_bytes = {1: 1, 2: 2, 12: 2, 4: 4}.get(int(meta["data type"]))
    expected_bytes = None if dtype_bytes is None else int(meta.get("header offset", 0)) + h * w * bands * dtype_bytes
    record = {
        "source": str(dat), "hdr": str(hdr), "shape_hwc": [h, w, bands],
        "interleave": meta.get("interleave"), "data_type": meta.get("data type"),
        "default_bands": meta.get("default bands"), "bytes": dat.stat().st_size,
        "expected_bytes": expected_bytes, "size_matches_header": dat.stat().st_size == expected_bytes,
    }
    if bands != 3 or not record["size_matches_header"]:
        record["error"] = "not a valid three-band RGB source"
        return record
    image, _ = read_envi_image(dat, hdr)
    thumb, scale_x, scale_y = make_thumbnail(image, 384)
    record["thumbnail_shape"] = list(thumb.shape)
    record["thumbnail_scale_xy"] = [scale_x, scale_y]
    # Each panel is a local vertical view; never squeeze the whole scan into one image.
    panel_height = min(1100, len(thumb))
    count = min(8, max(1, len(thumb) // panel_height))
    starts = np.linspace(0, max(0, len(thumb) - panel_height), count).round().astype(int)
    canvas = Image.new("RGB", (4 * 384, 2 * (panel_height + 34)), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, start in enumerate(starts):
        panel = Image.fromarray(np.ascontiguousarray(thumb[start:start + panel_height, :, :3]), "RGB")
        x = (index % 4) * 384
        y = (index // 4) * (panel_height + 34) + 34
        canvas.paste(panel, (x, y))
        draw.text((x + 5, y - 28), f"{index + 1}: raw y {int(start * scale_y):,}", fill="white")
    overview = output / "overviews" / f"{dat.stem}.jpg"
    overview.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(overview, quality=88)
    record["overview"] = str(overview)
    try:
        detection = detect_core_boxes(image, M11Config(thumbnail_width=512))
        record["detection_method"] = detection.method
        record["candidate_count"] = len(detection.candidates)
        record["candidates"] = [
            {"box_id": c.box_id, "bbox_raw": list(c.bbox_xyxy_raw), "bbox_thumb": list(c.bbox_xyxy_thumb),
             "boundary_score": c.boundary_score} for c in detection.candidates
        ]
    except Exception as exc:
        record["detection_error"] = f"{type(exc).__name__}: {exc}"
        record["candidate_count"] = 0
        record["candidates"] = []
    del image
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-stem", action="append", default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    sources = sorted(args.source_dir.glob("RGB*.dat"))
    if args.include_stem:
        sources = [dat for dat in sources if any(part in dat.stem for part in args.include_stem)]
    for dat in sources:
        print(f"Inspecting {dat.name}", flush=True)
        try:
            record = inspect_one(dat, args.output_dir)
        except Exception as exc:
            record = {"source": str(dat), "error": f"{type(exc).__name__}: {exc}"}
        records.append(record)
        (args.output_dir / "inventory.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  candidates={record.get('candidate_count')} error={record.get('error') or record.get('detection_error')}", flush=True)


if __name__ == "__main__":
    main()
