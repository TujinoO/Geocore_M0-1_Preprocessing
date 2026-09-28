"""Reproducible development-set QA of RGB dark-rail slot proposals.

The labels here were already inspected while developing the heuristic. This
report is not an independent-test or deployment-authorization result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules" / "M1-3_separator_mark"))
from geocore_m1_3.layout.rgb_slot_geometry import estimate_dark_rail_geometry  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows = []
    for sample in audit:
        sample_id = sample["sample_id"]
        image_path = (args.crops / f"{sample_id}.png" if sample["box_count"] == 1
                      else args.queue / "images" / f"{sample_id}.png")
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        candidate = estimate_dark_rail_geometry(rgb)
        truth_count = int(sample["slot_count"])
        internal = candidate.separator_x[1:-1] if candidate else []
        if truth_count >= 2:
            box = sample["box_bbox_xyxy"]
            crop_x0 = max(0, int(round(box[0])))
            slots = sample["slot_bboxes_xyxy"]
            truth_dividers = [
                (slots[i][2] + slots[i + 1][0]) / 2 - crop_x0
                for i in range(len(slots) - 1)
            ]
            errors = [abs(left - right) for left, right in zip(internal, truth_dividers)] \
                if len(internal) == len(truth_dividers) else []
        else:
            truth_dividers, errors = [], []
        rows.append({
            "sample_id": sample_id,
            "source": sample["source"],
            "image": str(image_path),
            "image_sha256": _sha256(image_path),
            "truth_physical_slot_count": truth_count,
            "truth_internal_dividers_x": truth_dividers,
            "candidate": candidate.to_dict() if candidate else None,
            "candidate_count_correct": candidate.count == truth_count if candidate and truth_count >= 2 else None,
            "internal_divider_mean_abs_error_px": float(np.mean(errors)) if errors else None,
            "internal_divider_max_abs_error_px": float(np.max(errors)) if errors else None,
        })
        print(sample_id, truth_count, candidate.count if candidate else None,
              candidate.review_required if candidate else True, flush=True)
    positives = [row for row in rows if row["truth_physical_slot_count"] >= 2]
    accepted = [row for row in positives if row["candidate"] and not row["candidate"]["review_required"]]
    negatives = [row for row in rows if row["truth_physical_slot_count"] == 0]
    report = {
        "evaluation_status": "development_reuse_not_independent_validation",
        "annotation_audit_sha256": _sha256(args.audit),
        "positive_count": len(positives),
        "candidate_count_correct": sum(row["candidate_count_correct"] is True for row in positives),
        "unflagged_positive_count": len(accepted),
        "unflagged_positive_count_correct": sum(row["candidate_count_correct"] is True for row in accepted),
        "negative_count": len(negatives),
        "unflagged_negative_proposal_count": sum(
            bool(row["candidate"] and not row["candidate"]["review_required"]) for row in negatives
        ),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
