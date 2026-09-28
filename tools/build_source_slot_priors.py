"""Build conservative, source-specific slot-count priors from reviewed SHPs.

This never edits the SHPs. A prior is advisory: it cannot manufacture a slot
polygon or prove that every box in the original acquisition has that layout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median


def _divider_fractions(sample: dict) -> list[float]:
    box = sample["box_bbox_xyxy"]
    slots = sample["slot_bboxes_xyxy"]
    width = max(1.0, box[2] - box[0])
    return [float(((slots[i][2] + slots[i + 1][0]) / 2 - box[0]) / width)
            for i in range(len(slots) - 1)]


def build_priors(audit: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for sample in audit:
        if sample["selection_kind"] == "no_box_negative" or sample["box_count"] != 1:
            continue
        grouped.setdefault(sample["source"], []).append(sample)
    sources = {}
    for path, samples in grouped.items():
        counts = [sample["slot_count"] for sample in samples]
        unanimous = len(set(counts)) == 1
        complete_ids = all("slot_idx_not_left_to_right_contiguous" not in sample["findings"] for sample in samples)
        fractions = [_divider_fractions(sample) for sample in samples]
        median_fractions = [median(items) for items in zip(*fractions)] if unanimous else []
        maximum_fraction_deviation = (max(abs(values[i] - median_fractions[i])
                                          for values in fractions for i in range(len(median_fractions)))
                                      if median_fractions else None)
        count_eligible = len(samples) >= 2 and unanimous and all(2 <= count <= 8 for count in counts)
        geometry_eligible = (count_eligible and complete_ids and maximum_fraction_deviation is not None
                             and maximum_fraction_deviation <= 0.035)
        sources[path] = {
            "physical_slot_count": counts[0] if unanimous else None,
            "support_sample_ids": [sample["sample_id"] for sample in samples],
            "support_sample_count": len(samples),
            "count_prior_eligible": count_eligible,
            "geometry_prior_eligible": geometry_eligible,
            "median_internal_divider_fraction": median_fractions if geometry_eligible else None,
            "max_internal_divider_fraction_deviation": maximum_fraction_deviation,
            "annotation_issues": {sample["sample_id"]: sample["findings"] for sample in samples if sample["findings"]},
            "status": "advisory_source_prior_not_per_box_truth",
        }
    return {"schema": "geocore_source_slot_priors.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_count": len(sources), "sources": sources,
            "meaning": "Manual slot count observed in labelled boxes from the same original RGB scan; not an automatic per-box label."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    priors = build_priors(audit)
    priors["annotation_audit_sha256"] = hashlib.sha256(args.audit.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(priors, ensure_ascii=False, indent=2), encoding="utf-8")
    for path, prior in priors["sources"].items():
        print(Path(path).name, prior["physical_slot_count"], prior["support_sample_count"],
              prior["count_prior_eligible"], prior["geometry_prior_eligible"])


if __name__ == "__main__":
    main()
