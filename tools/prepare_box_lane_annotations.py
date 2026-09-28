"""Prepare review-only RGB crops and empty ArcGIS polygon shapefiles.

Run with geo_env2 Python (GDAL/OGR). Nothing is labelled automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from datetime import date
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "M1-1_rotation_correction"))
from geocore_m1_1.io_envi import read_envi_image  # noqa: E402


def _choose(records: list[dict]) -> list[dict]:
    selected = []
    for record in records:
        name = Path(record["source"]).stem
        h = record["shape_hwc"][0]
        candidates = record.get("candidates", [])
        if "sbm" in name.lower():
            for fraction in (0.30, 0.68):
                span = min(h // 9, 10000)
                center = int(h * fraction)
                selected.append({"source": record, "bbox": [0, max(0, center - span // 2),
                                                       record["shape_hwc"][1] - 1, min(h - 1, center + span // 2)],
                                 "kind": "no_box_negative", "candidate_id": "none"})
            continue
        if not candidates:
            continue
        fractions = (0.15, 0.50, 0.80) if any(key in name for key in ("ZK", "20260702")) else (0.25, 0.72)
        indices = sorted({min(len(candidates) - 1, max(0, int(round((len(candidates) - 1) * f)))) for f in fractions})
        for index in indices:
            candidate = candidates[index]
            selected.append({"source": record, "bbox": candidate["bbox_raw"],
                             "kind": "box_candidate_review", "candidate_id": candidate["box_id"]})
    return selected


def _empty_shp(path: Path, layer_name: str) -> None:
    """Write an empty ESRI Polygon shapefile without optional GIS packages."""

    header = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, 50)
    header += struct.pack("<2i8d", 1000, 5, *([0.0] * 8))
    path.write_bytes(header)
    path.with_suffix(".shx").write_bytes(header)
    fields = [("BOX_ID", "C", 24), ("SLOT_IDX", "N", 5), ("STATUS", "C", 20)]
    today = date.today()
    dbf = struct.pack("<BBBBIHH20s", 3, today.year - 1900, today.month, today.day,
                      0, 32 + 32 * len(fields) + 1, 1 + sum(item[2] for item in fields), b"")
    for name, field_type, width in fields:
        dbf += struct.pack("<11sc4sBB14s", name.encode("ascii")[:10].ljust(11, b"\x00"),
                           field_type.encode("ascii"), b"", width, 0, b"")
    path.with_suffix(".dbf").write_bytes(dbf + b"\x0d")
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def _write_pgw(path: Path, width: int, height: int) -> None:
    # Local image space: x is pixel column, y = height - pixel row.
    path.with_suffix(".pgw").write_text(
        f"1\n0\n0\n-1\n0.5\n{height - 0.5}\n", encoding="ascii"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-stem", action="append", default=[],
                        help="Only include source names containing this text; repeatable")
    args = parser.parse_args()
    records = json.loads(args.inventory.read_text(encoding="utf-8"))
    if args.include_stem:
        records = [record for record in records if any(part in Path(record["source"]).stem
                                                     for part in args.include_stem)]
    selected = _choose(records)
    image_dir = args.output_dir / "images"
    shp_dir = args.output_dir / "shp"
    image_dir.mkdir(parents=True, exist_ok=True)
    shp_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, item in enumerate(selected, start=1):
        source = Path(item["source"]["source"])
        image, _ = read_envi_image(source, source.with_suffix(".hdr"))
        src_h, src_w = image.shape[:2]
        x0, y0, x1, y1 = item["bbox"]
        span = y1 - y0 + 1
        margin_y = max(300, int(span * 0.09)) if item["kind"] != "no_box_negative" else 0
        raw_x0, raw_x1 = 0, src_w
        raw_y0, raw_y1 = max(0, y0 - margin_y), min(src_h, y1 + margin_y + 1)
        step = max(1, math.ceil(src_w / 2048))
        crop = np.ascontiguousarray(image[raw_y0:raw_y1:step, raw_x0:raw_x1:step, :3])
        label = f"sample_{index:03d}"
        image_path = image_dir / f"{label}.png"
        box_shp = shp_dir / f"{label}_box.shp"
        slots_shp = shp_dir / f"{label}_slots.shp"
        for target in (image_path, box_shp, slots_shp):
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}")
        Image.fromarray(crop, "RGB").save(image_path, compress_level=3)
        _write_pgw(image_path, crop.shape[1], crop.shape[0])
        _empty_shp(box_shp, "box")
        _empty_shp(slots_shp, "slots")
        row = {
            "sample_id": label, "source": str(source), "source_bytes": source.stat().st_size,
            "raw_window_xyxy_exclusive": [raw_x0, raw_y0, raw_x1, raw_y1],
            "raw_stride_xy": [step, step], "image_size_wh": [crop.shape[1], crop.shape[0]],
            "image": str(image_path), "image_sha256": _sha256(image_path),
            "box_shp": str(box_shp), "slots_shp": str(slots_shp),
            "selection_kind": item["kind"], "candidate_id": item["candidate_id"],
            "annotation_status": "pending", "physical_slot_count": None,
            "coordinate_rule": "raw_x=window_x0+png_x*stride; raw_y=window_y0+png_y*stride; shapefile_y=png_height-png_y",
        }
        manifest.append(row)
        (args.output_dir / "annotation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{label}: {source.name} / {item['kind']} / {crop.shape[1]}x{crop.shape[0]}", flush=True)
    with (args.output_dir / "annotation_index.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "selection_kind", "candidate_id", "source",
                                                  "image", "box_shp", "slots_shp", "annotation_status", "physical_slot_count"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in manifest)


if __name__ == "__main__":
    main()
