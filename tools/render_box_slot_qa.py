"""Render a non-destructive visual QA overlay for one annotated crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--full-source", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    manifest = {row["sample_id"]: row for row in json.loads((args.queue / "annotation_manifest.json").read_text(encoding="utf-8"))}
    audit = {row["sample_id"]: row for row in json.loads(args.audit.read_text(encoding="utf-8"))}
    full = {entry["sample_id"]: entry for source in json.loads(args.full_source.read_text(encoding="utf-8")) for entry in source["samples"]}
    sample = manifest[args.sample]
    report = audit[args.sample]
    with Image.open(sample["image"]) as source_image:
        image = source_image.convert("RGB")
    scale = min(1.0, 768 / image.width, 2048 / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)

    def rectangle(bbox: list[float], color: str, width: int = 4) -> None:
        draw.rectangle([round(value * scale) for value in bbox], outline=color, width=width)

    if report["box_bbox_xyxy"] is not None:
        rectangle(report["box_bbox_xyxy"], "red", 5)
    for bbox in report["slot_bboxes_xyxy"]:
        rectangle(bbox, "lime", 2)
    x0, y0, _, _ = sample["raw_window_xyxy_exclusive"]
    sx, sy = sample["raw_stride_xy"]
    candidate = full[args.sample].get("best_candidate_raw")
    if candidate is not None:
        rectangle([(candidate[0] - x0) / sx, (candidate[1] - y0) / sy,
                   (candidate[2] - x0) / sx, (candidate[3] - y0) / sy], "cyan", 5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=88)


if __name__ == "__main__":
    main()
