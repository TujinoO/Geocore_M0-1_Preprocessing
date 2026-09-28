"""Compare labelled M1-1 box candidates with no-box false candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "M1-1_rotation_correction"))
sys.path.insert(0, str(ROOT / "modules" / "M1-3_separator_mark"))
from geocore_m1_1.io_envi import read_envi_image  # noqa: E402
from geocore_m1_3.layout.rgb_slot_geometry import estimate_rgb_slot_geometry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    result = []
    for source in json.loads(args.full_source.read_text(encoding="utf-8")):
        path = Path(source["source"])
        image, _ = read_envi_image(path, path.with_suffix(".hdr"))
        for sample in source["samples"]:
            boxes = [sample["best_candidate_raw"]] if sample.get("best_candidate_raw") else sample.get("candidate_overlap_gt_0_1", [])
            for bbox in boxes:
                x0, y0, x1, y1 = [int(value) for value in bbox]
                step = max(1, (x1 - x0 + 1) // 1024, (y1 - y0 + 1) // 2048)
                crop = np.ascontiguousarray(image[y0:y1 + 1:step, x0:x1 + 1:step, :3])
                candidate = estimate_rgb_slot_geometry(crop)
                gray = (crop[:, :, 0].astype(float) * 0.299 + crop[:, :, 1] * 0.587 + crop[:, :, 2] * 0.114)
                dx = np.abs(gray[:, 2:] - gray[:, :-2])
                record = {"sample_id": sample["sample_id"], "positive": bool(sample.get("best_candidate_raw")),
                          "candidate_bbox_raw": bbox, "slot_candidate_count": candidate.count if candidate else None,
                          "slot_candidate_score": candidate.score if candidate else None,
                          "slot_weakest_separator": candidate.candidates[0]["weakest_separator"] if candidate else None,
                          "slot_score_margin": candidate.score_margin if candidate else None,
                          "gray_mean": float(gray.mean()), "gray_std": float(gray.std()),
                          "column_edge_75_mean": float(np.percentile(dx, 75, axis=0).mean())}
                result.append(record)
                print(record["sample_id"], record["positive"], record["slot_candidate_score"], flush=True)
        del image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
