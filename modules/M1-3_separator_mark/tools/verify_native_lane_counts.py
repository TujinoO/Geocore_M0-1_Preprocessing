"""Run V4 and native-resolution lane-count QA for every M1-1 corrected box."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from geocore_mask.inference.predictor import CoreMaskPredictor
from geocore_mask.utils.model_package import load_model_package
from geocore_m1_3.config import LayoutConfig
from geocore_m1_3.layout.frame_bounds import trim_orange_frame_rows
from geocore_m1_3.layout.lane_detector import detect_lanes, estimate_lane_count


def _same_file_content(a: Path, b: Path) -> bool:
    if not a.is_file() or not b.is_file() or a.stat().st_size != b.stat().st_size:
        return False
    def digest(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()
    return digest(a) == digest(b)


def verify_batch(m11_dir: Path, output_dir: Path, expected_count: int | None = None) -> dict:
    boxes = sorted(json.loads((m11_dir / "metadata.json").read_text(encoding="utf-8"))["boxes"],
                   key=lambda item: int(item["order_index"]))
    if expected_count is not None and len(boxes) != expected_count:
        raise ValueError(f"Expected {expected_count} boxes; M1-1 metadata has {len(boxes)}")
    package = load_model_package()
    predictor = CoreMaskPredictor(package, device="auto")
    Image.MAX_IMAGE_PIXELS = 300_000_000
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "native_lane_count_qa.json"
    rows: list[dict] = []
    for box in boxes:
        box_id = box["box_id"]
        source = Path(box["output_image"])
        m12_dir = output_dir / f"{box_id}_v4"
        metadata_path = m12_dir / "metadata.json"
        try:
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                existing_source = Path(metadata.get("input_path", ""))
                if metadata.get("model_version") != package.model_version or (existing_source != source and not _same_file_content(existing_source, source)):
                    raise ValueError("Existing V4 output does not match model version or input image; do not overwrite")
                if not (m12_dir / "mask.png").is_file():
                    raise FileNotFoundError("Existing V4 metadata has no mask.png")
            else:
                predictor.predict(source, m12_dir, output_preview=False)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(m12_dir / "mask.png") as image:
                mask = np.asarray(image.convert("L")) > 127
            if rgb.shape[:2] != mask.shape:
                raise ValueError(f"Mask and RGB dimensions differ: {mask.shape} vs {rgb.shape[:2]}")
            trim = trim_orange_frame_rows(mask, rgb)
            estimate = estimate_lane_count(mask)
            row = {
                "box_id": box_id,
                "native_size_px": [int(mask.shape[1]), int(mask.shape[0])],
                "model_version": metadata["model_version"],
                "foreground_ratio": round(float(mask.mean()), 4),
                "orange_frame_trim": trim,
                "lane_count": estimate.count,
                "lane_confidence": round(float(estimate.confidence), 4),
                "lane_reason": estimate.reason,
                "lane_runs_x": estimate.runs,
                "m1_2_output_dir": str(m12_dir),
            }
            try:
                lanes = detect_lanes(mask, LayoutConfig(), core_box_id=box_id)
                row["lane_bboxes"] = [lane.bbox for lane in lanes]
                row["passed_auto_lane_count"] = True
            except ValueError as exc:
                row["passed_auto_lane_count"] = False
                row["lane_error"] = str(exc)
            del rgb, mask
        except Exception as exc:
            row = {"box_id": box_id, "passed_auto_lane_count": False, "error": str(exc)}
        rows.append(row)
        report = {
            "scope": "Native-resolution V4 mask and M1-3 automatic lane count; full reconstruction and true depth order are separate checks",
            "model_version": package.model_version,
            "box_count_expected": len(boxes),
            "box_count_processed": len(rows),
            "passed_auto_lane_count": sum(bool(item["passed_auto_lane_count"]) for item in rows),
            "boxes": rows,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{box_id}: lanes={row.get('lane_count', 'error')} reason={row.get('lane_reason', row.get('error'))}", flush=True)
        gc.collect()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m11-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    report = verify_batch(args.m11_dir, args.output_dir, args.expected_count)
    print(f"Native auto lane count passed {report['passed_auto_lane_count']}/{report['box_count_processed']}")
    return 0 if report["passed_auto_lane_count"] == report["box_count_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
