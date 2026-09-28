"""Read-only QA and baseline evaluation for the M1-1/M1-3 ArcGIS labels.

Use a Python environment with GDAL/OGR, NumPy and Pillow. The source SHP,
PNG, CSV and manifest are never changed. Results are written to a new path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from osgeo import ogr
from PIL import Image

ogr.UseExceptions()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "M1-1_rotation_correction"))
sys.path.insert(0, str(ROOT / "modules" / "M1-3_separator_mark"))
from geocore_m1_1.config import M11Config  # noqa: E402
from geocore_m1_1.segment_boxes import detect_core_boxes  # noqa: E402
from geocore_m1_3.layout.rgb_slot_geometry import estimate_rgb_slot_geometry  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(path: Path, image_height: int) -> list[dict]:
    dataset = ogr.Open(str(path))
    if dataset is None:
        raise ValueError(f"Cannot open shapefile: {path}")
    layer = dataset.GetLayer()
    result = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            raise ValueError(f"Missing geometry: {path} FID={feature.GetFID()}")
        geometry = geometry.Clone()
        x0, x1, y0, y1 = geometry.GetEnvelope()
        result.append({
            "fid": feature.GetFID(),
            "slot_idx": feature.GetField("SLOT_IDX"),
            "bbox_xyxy": [float(x0), float(image_height - y1), float(x1), float(image_height - y0)],
            "area": float(geometry.GetArea()),
            "valid": bool(geometry.IsValid()),
            "geometry": geometry,
        })
    dataset = None
    return result


def _iou(left: list[float], right: list[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, area_left + area_right - intersection)


def audit_one(item: dict, baseline: bool) -> dict:
    image_path = Path(item["image"])
    with Image.open(image_path) as image:
        width, height = image.size
        image_rgb = np.asarray(image.convert("RGB")) if baseline else None
    boxes = _features(Path(item["box_shp"]), height)
    slots = _features(Path(item["slots_shp"]), height)
    findings = []
    positive = item["selection_kind"] != "no_box_negative"
    if len(boxes) != int(positive):
        findings.append("expected_one_box" if positive else "negative_has_box")
    if positive and not 2 <= len(slots) <= 8:
        findings.append("slot_count_outside_2_to_8")
    if not positive and slots:
        findings.append("negative_has_slots")
    ordered = sorted(slots, key=lambda feature: (feature["bbox_xyxy"][0] + feature["bbox_xyxy"][2]) / 2)
    indexes = [feature["slot_idx"] for feature in ordered]
    if slots and indexes != list(range(1, len(slots) + 1)):
        findings.append("slot_idx_not_left_to_right_contiguous")
    for feature in boxes + slots:
        x0, y0, x1, y1 = feature["bbox_xyxy"]
        if not feature["valid"] or feature["area"] <= 0:
            findings.append("invalid_or_empty_geometry")
        if x0 < -1 or y0 < -1 or x1 > width + 1 or y1 > height + 1:
            findings.append("geometry_outside_image")
    if boxes:
        box = boxes[0]
        for slot in slots:
            outside = slot["geometry"].Difference(box["geometry"]).GetArea()
            if outside > 0.02 * max(slot["area"], 1):
                findings.append("slot_outside_box")
                break
        for i, first in enumerate(slots):
            for second in slots[i + 1:]:
                if first["geometry"].Intersection(second["geometry"]).GetArea() > 0.01 * min(first["area"], second["area"]):
                    findings.append("overlapping_slots")
                    break
    report = {
        "sample_id": item["sample_id"], "source": item["source"],
        "selection_kind": item["selection_kind"], "image_size_wh": [width, height],
        "image_sha256_matches_manifest": _sha256(image_path) == item["image_sha256"],
        "label_file_sha256": {
            str(path): _sha256(path)
            for stem in (Path(item["box_shp"]), Path(item["slots_shp"]))
            for path in (stem, stem.with_suffix(".shx"), stem.with_suffix(".dbf"), stem.with_suffix(".cpg"))
            if path.exists()
        },
        "box_count": len(boxes), "slot_count": len(slots), "slot_indices_left_to_right": indexes,
        "box_bbox_xyxy": boxes[0]["bbox_xyxy"] if len(boxes) == 1 else None,
        "slot_bboxes_xyxy": [feature["bbox_xyxy"] for feature in ordered],
        "findings": sorted(set(findings)),
    }
    if not report["image_sha256_matches_manifest"]:
        report["findings"].append("image_hash_changed")
    if baseline and image_rgb is not None:
        detection = detect_core_boxes(image_rgb, M11Config())
        predicted = [list(candidate.bbox_xyxy_raw) for candidate in detection.candidates]
        truth = report["box_bbox_xyxy"]
        report["m11_baseline"] = {
            "method": detection.method,
            "candidate_count": len(predicted),
            "best_bbox_iou": max((_iou(truth, candidate) for candidate in predicted), default=0.0) if truth else None,
            "candidate_bboxes_xyxy": predicted,
        }
        if truth and len(slots) >= 2:
            x0, y0, x1, y1 = [int(round(value)) for value in truth]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            candidate = estimate_rgb_slot_geometry(image_rgb[y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else None
            truth_dividers = [(ordered[i]["bbox_xyxy"][2] + ordered[i + 1]["bbox_xyxy"][0]) / 2 - x0
                              for i in range(len(ordered) - 1)]
            internal = candidate.separator_x[1:-1] if candidate else []
            report["m13_baseline"] = {
                "truth_count": len(slots),
                "candidate_count": candidate.count if candidate else None,
                "candidate_review_required": candidate.review_required if candidate else True,
                "candidate_score": candidate.score if candidate else None,
                "score_margin": candidate.score_margin if candidate else None,
                "candidate_dividers_x_in_box": internal,
                "truth_dividers_x_in_box": truth_dividers,
                "mean_divider_error_px": float(np.mean(np.abs(np.asarray(internal) - truth_dividers)))
                if len(internal) == len(truth_dividers) else None,
                "candidate_rankings": candidate.candidates if candidate else [],
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
    items = json.loads((args.queue / "annotation_manifest.json").read_text(encoding="utf-8"))
    reports = []
    for item in items:
        report = audit_one(item, args.baseline)
        reports.append(report)
        print(f"{item['sample_id']}: boxes={report['box_count']} slots={report['slot_count']} "
              f"issues={','.join(report['findings']) or 'none'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
