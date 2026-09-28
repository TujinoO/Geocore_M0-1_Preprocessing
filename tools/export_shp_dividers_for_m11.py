"""Map reviewed physical-slot SHPs into M1-1 corrected-box pixel coordinates.

Only boxes matching an annotated original ENVI source are exported. The
rotation and crop transform are taken from M1-1 metadata, not approximated by
copying PNG/SHP x coordinates. Requires GDAL/OGR and Pillow. Read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import struct

from osgeo import ogr

ogr.UseExceptions()


def _bbox_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _png_width(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"M1-1 corrected box is not a PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if not 0 < width <= 100_000 or not 0 < height <= 10_000_000:
        raise ValueError(f"Invalid corrected-box dimensions: {path}")
    return width


def _read_features(path: Path) -> list[tuple[int | None, ogr.Geometry]]:
    dataset = ogr.Open(str(path))
    if dataset is None:
        raise ValueError(f"Cannot open shapefile: {path}")
    result = []
    for feature in dataset.GetLayer():
        geometry = feature.GetGeometryRef()
        if geometry is None or not geometry.IsValid() or geometry.GetArea() <= 0:
            raise ValueError(f"Invalid or empty geometry in {path}")
        result.append((feature.GetField("SLOT_IDX"), geometry.Clone()))
    dataset = None
    return result


def _midline_edges(geometry: ogr.Geometry, y_gis: float, width: int) -> tuple[float, float]:
    scan = ogr.Geometry(ogr.wkbLineString)
    scan.AddPoint(-width, y_gis)
    scan.AddPoint(width * 2, y_gis)
    section = geometry.Intersection(scan)
    if section is None or section.IsEmpty():
        raise ValueError("Slot polygon does not cross the box midline")
    x0, x1, _, _ = section.GetEnvelope()
    return float(x0), float(x1)


def _raw_to_corrected_x(raw_x: float, raw_y: float, box: dict) -> float:
    # M1-1 rotates the padded raw crop, then crops the rotated result.
    px0, py0 = box["source_crop_bbox_raw"][:2]
    matrix = box["rotation_matrix_2x3"]
    rotated_x = matrix[0][0] * (raw_x - px0) + matrix[0][1] * (raw_y - py0) + matrix[0][2]
    return float(rotated_x - box["bbox_xyxy_corrected"][0])


def export_one(item: dict, box: dict) -> list[int]:
    image_width, image_height = item["image_size_wh"]
    shapes = _read_features(Path(item["slots_shp"]))
    ordered = sorted(shapes, key=lambda pair: pair[0] if pair[0] is not None else -1)
    if [index for index, _ in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError(f"{item['sample_id']}: SLOT_IDX is not 1..N")
    if len(ordered) < 2:
        raise ValueError(f"{item['sample_id']}: fewer than two physical slots")
    box_shapes = _read_features(Path(item["box_shp"]))
    if len(box_shapes) != 1:
        raise ValueError(f"{item['sample_id']}: expected exactly one box polygon")
    _, _, y0_gis, y1_gis = box_shapes[0][1].GetEnvelope()
    window_x0, window_y0 = item["raw_window_xyxy_exclusive"][:2]
    stride_x, stride_y = item["raw_stride_xy"]
    dividers_by_height = []
    for fraction in (0.25, 0.50, 0.75):
        y_gis = y0_gis + fraction * (y1_gis - y0_gis)
        local_y = image_height - y_gis
        edges = [_midline_edges(geometry, y_gis, image_width) for _, geometry in ordered]
        raw_y = window_y0 + local_y * stride_y
        row = []
        for left, right in zip(edges, edges[1:]):
            local_x = (left[1] + right[0]) / 2
            raw_x = window_x0 + local_x * stride_x
            row.append(_raw_to_corrected_x(raw_x, raw_y, box))
        dividers_by_height.append(row)
    output_width = _png_width(Path(box["output_image"]))
    maximum_drift = max(max(values) - min(values) for values in zip(*dividers_by_height))
    if maximum_drift > max(12.0, 0.04 * output_width):
        raise ValueError(f"{item['sample_id']}: slot divider drift after correction is {maximum_drift:.1f}px")
    dividers = [round(median(values)) for values in zip(*dividers_by_height)]
    if any(not 0 < x < output_width for x in dividers) or any(
        b <= a for a, b in zip(dividers, dividers[1:])
    ):
        raise ValueError(f"{item['sample_id']}: transformed dividers are outside or unordered")
    return dividers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--m11-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-box-iou", type=float, default=0.75)
    parser.add_argument("--allow-identical-source-copies", action="store_true",
                        help="Permit different .dat paths only after full SHA-256 equality is verified")
    args = parser.parse_args()
    qa_path = args.output.with_name(args.output.stem + "_qa.json")
    if args.output.exists() or qa_path.exists():
        raise FileExistsError("Refusing to overwrite divider output or QA report")
    items = json.loads((args.queue / "annotation_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads(args.m11_metadata.read_text(encoding="utf-8"))
    metadata_source = Path(metadata["input_path"]).resolve()
    source = str(metadata_source).casefold()
    alias_hashes: dict[str, str] = {}
    boxes = metadata["boxes"]
    mapping, qa = {}, []
    for item in items:
        if item["selection_kind"] == "no_box_negative":
            continue
        item_source = Path(item["source"]).resolve()
        if str(item_source).casefold() != source:
            if not args.allow_identical_source_copies or item_source.name.casefold() != metadata_source.name.casefold():
                continue
            if not metadata_source.is_file() or not item_source.is_file() or metadata_source.stat().st_size != item_source.stat().st_size:
                raise ValueError("Source-copy size differs or one source is missing; cannot remap SHP")
            original_hdr = metadata_source.with_suffix(".hdr")
            labelled_hdr = item_source.with_suffix(".hdr")
            if (not original_hdr.is_file() or not labelled_hdr.is_file()
                    or original_hdr.read_bytes() != labelled_hdr.read_bytes()):
                raise ValueError("Source-copy ENVI headers differ or are missing; cannot remap SHP")
            for path in (metadata_source, item_source):
                if str(path) not in alias_hashes:
                    alias_hashes[str(path)] = _sha256(path)
            if alias_hashes[str(metadata_source)] != alias_hashes[str(item_source)]:
                raise ValueError("Source-copy SHA-256 differs; cannot remap SHP")
        box_shapes = _read_features(Path(item["box_shp"]))
        if len(box_shapes) != 1:
            qa.append({"sample_id": item["sample_id"], "status": "skipped_box_label_count"})
            continue
        x0, x1, y0, y1 = box_shapes[0][1].GetEnvelope()
        local_bbox = [x0, item["image_size_wh"][1] - y1, x1, item["image_size_wh"][1] - y0]
        wx, wy = item["raw_window_xyxy_exclusive"][:2]
        sx, sy = item["raw_stride_xy"]
        raw_bbox = [wx + local_bbox[0] * sx, wy + local_bbox[1] * sy,
                    wx + local_bbox[2] * sx, wy + local_bbox[3] * sy]
        scored = sorted(((_bbox_iou(raw_bbox, box["bbox_xyxy_raw"]), box) for box in boxes),
                        key=lambda pair: pair[0], reverse=True)
        if not scored or scored[0][0] < args.minimum_box_iou:
            qa.append({"sample_id": item["sample_id"], "status": "skipped_low_box_iou",
                       "best_iou": scored[0][0] if scored else 0.0})
            continue
        iou, box = scored[0]
        try:
            dividers = export_one(item, box)
        except ValueError as exc:
            qa.append({"sample_id": item["sample_id"], "status": "skipped_invalid_slot_label",
                       "box_id": box["box_id"], "reason": str(exc)})
            continue
        if box["box_id"] in mapping:
            raise ValueError(f"Multiple annotations matched {box['box_id']}; review duplicates")
        mapping[box["box_id"]] = dividers
        qa.append({"sample_id": item["sample_id"], "status": "exported",
                   "box_id": box["box_id"], "box_iou": iou, "corrected_dividers_x": dividers})
        print(item["sample_id"], box["box_id"], dividers, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    qa_path.write_text(json.dumps({"source": metadata["input_path"], "minimum_box_iou": args.minimum_box_iou,
                                   "identical_source_copies_sha256": alias_hashes,
                                   "records": qa}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
