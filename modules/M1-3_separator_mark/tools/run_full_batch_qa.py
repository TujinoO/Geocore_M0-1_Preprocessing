"""Reconstruct corrected boxes for geometry QA with explicitly fictitious 0-1 m depths."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path


def _run_one(job: dict) -> dict:
    from geocore_m1_3.pipeline import run_segment_depth

    box_id = job["box_id"]
    output = Path(job["m13_root"]) / f"{box_id}_full_qa"
    if output.exists():
        required = [output / "quality_report.json", output / "reconstructed_strip.png", output / "lane_detection.json"]
        if not all(path.is_file() for path in required):
            return {"box_id": box_id, "status": "incomplete_existing_output_not_overwritten", "output_dir": str(output)}
        quality = json.loads(required[0].read_text(encoding="utf-8"))
        return {
            "box_id": box_id, "status": "existing_verified", "output_dir": str(output),
            "lane_count": quality.get("lane_count_detected"), "segment_count": quality.get("segment_count"),
            "warnings": quality.get("warnings", []),
        }
    payload = {
        "image_path": job["image_path"],
        "m1_2_output_dir": job["m12_dir"],
        "output_dir": job["m13_root"],
        "hole_id": "ZKZ4-5_QA_ONLY",
        "core_box_id": box_id,
        "task_id": f"{box_id}_full_qa",
        "depth_start_m": 0,
        "depth_end_m": 1,
        "qa_assumed_depth_not_field_truth": True,
        "segmentation": {"segment_length_cm": 25.0},
        "depth_order_confirmed": False,
    }
    result = run_segment_depth(payload)
    quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    return {
        "box_id": box_id, "status": result["status"], "output_dir": str(output),
        "lane_count": quality.get("lane_count_detected"), "segment_count": quality.get("segment_count"),
        "warnings": quality.get("warnings", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m11-dir", type=Path, required=True)
    parser.add_argument("--m12-root", type=Path, required=True)
    parser.add_argument("--m13-root", type=Path, required=True)
    parser.add_argument("--native-lane-report", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 4:
        parser.error("--max-workers must be between 1 and 4")
    boxes = sorted(json.loads((args.m11_dir / "metadata.json").read_text(encoding="utf-8"))["boxes"],
                   key=lambda item: int(item["order_index"]))
    if not boxes:
        raise ValueError("M1-1 metadata has no boxes")
    lane_report = json.loads(args.native_lane_report.read_text(encoding="utf-8"))
    passed_ids = {item["box_id"] for item in lane_report["boxes"] if item["passed_auto_lane_count"]}
    expected_lane_counts = {item["box_id"]: int(item["lane_count"]) for item in lane_report["boxes"]}
    if len(passed_ids) != len(boxes) or {box["box_id"] for box in boxes} != passed_ids:
        raise ValueError("Native-resolution automatic lane count must pass every box before full-batch QA")
    jobs = [{
        "box_id": box["box_id"], "image_path": box["output_image"],
        "m12_dir": str(args.m12_root / f"{box['box_id']}_v4"), "m13_root": str(args.m13_root),
    } for box in boxes]
    args.m13_root.mkdir(parents=True, exist_ok=True)
    report_path = args.m13_root / "full_batch_qa.json"
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_box = {pool.submit(_run_one, job): job["box_id"] for job in jobs}
        for future in as_completed(future_to_box):
            box_id = future_to_box[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"box_id": box_id, "status": "failed", "error": str(exc)}
            row["expected_lane_count"] = expected_lane_counts[box_id]
            rows.append(row)
            rows.sort(key=lambda item: item["box_id"])
            report = {
                "scope": "Full-resolution M1-3 reconstruction QA; fictitious 0-1 m per box, depth order not confirmed",
                "box_count_expected": len(boxes), "box_count_processed": len(rows),
                "succeeded": sum(item.get("status") in {"succeeded", "existing_verified"}
                                 and item.get("lane_count") == item["expected_lane_count"] for item in rows),
                "boxes": rows,
            }
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{box_id}: {row['status']} lanes={row.get('lane_count', 'n/a')} warnings={len(row.get('warnings', []))}", flush=True)
    return 0 if report["succeeded"] == len(boxes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
