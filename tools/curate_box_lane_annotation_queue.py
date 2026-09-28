"""Copy visually screened crops into one unambiguous annotation queue.

Source draft queues remain intact for audit. No source or output is deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil


# Screened on contact sheets and representative full-size crops, 2026-09-28.
# These are candidate crops for manual annotation, not approved labels.
SELECTION = {
    "01_annotation_queue": (1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25),
    "02_revised_large_box_queue": (1, 2, 3, 5, 7),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image_dir, shp_dir = args.output / "images", args.output / "shp"
    image_dir.mkdir(parents=True, exist_ok=True)
    shp_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for queue, indices in SELECTION.items():
        parent = args.root / queue
        manifest = {item["sample_id"]: item for item in json.loads(
            (parent / "annotation_manifest.json").read_text(encoding="utf-8")
        )}
        for source_index in indices:
            old_id = f"sample_{source_index:03d}"
            source_record = manifest[old_id]
            new_id = f"sample_{len(records) + 1:03d}"
            targets = []
            for src, dst in (
                (parent / "images" / f"{old_id}.png", image_dir / f"{new_id}.png"),
                (parent / "images" / f"{old_id}.pgw", image_dir / f"{new_id}.pgw"),
            ):
                targets.append((src, dst))
            for kind in ("box", "slots"):
                for suffix in (".shp", ".shx", ".dbf", ".cpg"):
                    targets.append((parent / "shp" / f"{old_id}_{kind}{suffix}",
                                    shp_dir / f"{new_id}_{kind}{suffix}"))
            for src, dst in targets:
                if dst.exists():
                    raise FileExistsError(f"Refusing to overwrite {dst}")
                shutil.copy2(src, dst)
            record = dict(source_record)
            record.update({
                "sample_id": new_id, "draft_queue": queue, "draft_sample_id": old_id,
                "image": str(image_dir / f"{new_id}.png"),
                "box_shp": str(shp_dir / f"{new_id}_box.shp"),
                "slots_shp": str(shp_dir / f"{new_id}_slots.shp"),
                "visual_qa": ("no_box_negative_candidate" if record["selection_kind"] == "no_box_negative"
                              else "complete_box_visible_for_manual_tracing"),
            })
            records.append(record)
            print(f"{new_id} <- {queue}/{old_id}", flush=True)
    (args.output / "annotation_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "annotation_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("sample_id", "selection_kind", "visual_qa", "draft_queue", "draft_sample_id",
                  "source", "image", "box_shp", "slots_shp", "annotation_status", "physical_slot_count")
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({key: item.get(key) for key in fields} for item in records)


if __name__ == "__main__":
    main()
