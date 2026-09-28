"""Evaluate M1-1 detections on original ENVI scans against annotated crops.

Only labelled boxes/windows are scored. Unlabelled parts of a long scan are
not treated as negatives. Source images and SHPs are read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "M1-1_rotation_correction"))
from geocore_m1_1.config import M11Config  # noqa: E402
from geocore_m1_1.io_envi import read_envi_image  # noqa: E402
from geocore_m1_1.segment_boxes import detect_core_boxes  # noqa: E402


def _iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    manifest = json.loads((args.queue / "annotation_manifest.json").read_text(encoding="utf-8"))
    audit = {row["sample_id"]: row for row in json.loads(args.audit.read_text(encoding="utf-8"))}
    grouped = {}
    for row in manifest:
        grouped.setdefault(row["source"], []).append(row)
    result = []
    for source, rows in grouped.items():
        image, metadata = read_envi_image(source, Path(source).with_suffix(".hdr"))
        detection = detect_core_boxes(image, M11Config())
        candidates = [list(candidate.bbox_xyxy_raw) for candidate in detection.candidates]
        source_result = {"source": source, "source_size_wh": [image.shape[1], image.shape[0]],
                         "method": detection.method, "candidate_count_full_source": len(candidates),
                         "samples": []}
        for row in rows:
            sample = audit[row["sample_id"]]
            x0, y0, _, _ = row["raw_window_xyxy_exclusive"]
            sx, sy = row["raw_stride_xy"]
            truth = sample["box_bbox_xyxy"]
            if truth is not None:
                truth_raw = [x0 + truth[0] * sx, y0 + truth[1] * sy,
                             x0 + truth[2] * sx, y0 + truth[3] * sy]
                best = max(((_iou(truth_raw, bbox), bbox) for bbox in candidates), default=(0.0, None))
                evaluation = {"sample_id": row["sample_id"], "truth_bbox_raw": truth_raw,
                              "best_bbox_iou": best[0], "best_candidate_raw": best[1]}
            else:
                window = row["raw_window_xyxy_exclusive"]
                intersecting = [bbox for bbox in candidates if _iou(window, bbox) > 0.1]
                evaluation = {"sample_id": row["sample_id"], "negative_window_raw": window,
                              "candidate_overlap_gt_0_1": intersecting}
            source_result["samples"].append(evaluation)
            print(row["sample_id"],
                  evaluation["best_bbox_iou"] if truth is not None else f"false_candidates={len(intersecting)}",
                  flush=True)
        result.append(source_result)
        del image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
