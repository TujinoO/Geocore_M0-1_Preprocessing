"""Create immutable evaluation crops from reviewed M1-1 box polygons.

The source PNG/SHP files are never modified. Each crop is a derived input for
testing the fixed M1-2 model and M1-3 lane-count code on the same boxes that
have physical-slot labels. Refuses to overwrite any existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation crops: {args.output_dir}")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    manifest = {
        item["sample_id"]: item
        for item in json.loads((args.queue / "annotation_manifest.json").read_text(encoding="utf-8"))
    }
    prepared = []
    args.output_dir.mkdir(parents=True)
    for row in audit:
        if row["box_count"] != 1 or row["slot_count"] < 2:
            continue
        item = manifest[row["sample_id"]]
        image_path = Path(item["image"])
        bbox = row["box_bbox_xyxy"]
        with Image.open(image_path) as source:
            x0 = max(0, int(round(bbox[0])))
            y0 = max(0, int(round(bbox[1])))
            x1 = min(source.width, int(round(bbox[2])))
            y1 = min(source.height, int(round(bbox[3])))
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"Invalid clipped box for {row['sample_id']}: {bbox}")
            crop = source.crop((x0, y0, x1, y1)).convert("RGB")
            output = args.output_dir / f"{row['sample_id']}.png"
            crop.save(output)
        prepared.append({
            "sample_id": row["sample_id"],
            "source": row["source"],
            "source_png": str(image_path),
            "source_png_sha256": _sha256(image_path),
            "box_shp": item["box_shp"],
            "slots_shp": item["slots_shp"],
            "crop_bbox_xyxy_exclusive": [x0, y0, x1, y1],
            "physical_slot_count": row["slot_count"],
            "crop_png": str(output),
            "crop_png_sha256": _sha256(output),
        })
        print(row["sample_id"], crop.size, row["slot_count"], flush=True)
    (args.output_dir / "eval_crops_manifest.json").write_text(
        json.dumps({"annotation_audit_sha256": _sha256(args.audit), "samples": prepared},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
