"""Audit a completed orange-box M1-1 batch without altering its outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from geocore_m1_1.config import M11Config
from geocore_m1_1.geometry import estimate_box_angle


def _rail_rows(rgb: np.ndarray) -> tuple[bool, bool]:
    red, green, blue = [rgb[:, :, i].astype(np.int16) for i in range(3)]
    orange = (red > 125) & (red > 1.35 * green) & (green > 1.35 * blue) & (red - blue > 80)
    width = rgb.shape[1]
    coverage = orange[:, int(width * 0.10):int(width * 0.90)].mean(axis=1)
    band = max(1, int(len(coverage) * 0.10))
    return bool(np.any(coverage[:band] > 0.35)), bool(np.any(coverage[-band:] > 0.35))


def verify_batch(output_dir: Path, expected_count: int, max_residual_deg: float = 0.20) -> dict:
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    boxes = sorted(metadata["boxes"], key=lambda item: int(item["order_index"]))
    count_ok = len(boxes) == expected_count == int(metadata["box_count"])
    bbox_nonoverlap = all(int(a["bbox_xyxy_raw"][3]) < int(b["bbox_xyxy_raw"][1])
                          for a, b in zip(boxes, boxes[1:]))
    Image.MAX_IMAGE_PIXELS = 300_000_000
    items = []
    for box in boxes:
        image_path = Path(box["output_image"])
        mask_path = Path(box["output_mask"])
        if not image_path.is_file() or not mask_path.is_file():
            items.append({"box_id": box["box_id"], "passed": False, "error": "missing output"})
            continue
        with Image.open(image_path) as image:
            full_size = image.size
            image.thumbnail((1024, 4096), Image.Resampling.BILINEAR)
            rgb = np.asarray(image.convert("RGB"))
            corrected_size = image.size
        with Image.open(mask_path) as mask_image:
            mask_size = mask_image.size
        top_rail, bottom_rail = _rail_rows(rgb)
        angle = estimate_box_angle(
            rgb, (0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1), M11Config()
        )
        passed = bool(top_rail and bottom_rail and mask_size == full_size
                      and abs(angle.angle_deg) <= max_residual_deg)
        items.append({
            "box_id": box["box_id"], "passed": passed,
            "thumbnail_size_px": corrected_size, "mask_size_px": mask_size,
            "corrected_size_px": full_size,
            "top_rail_found": top_rail, "bottom_rail_found": bottom_rail,
            "residual_angle_deg": round(float(angle.angle_deg), 4),
            "source_bbox_raw": box["bbox_xyxy_raw"],
        })
    return {
        "expected_count": expected_count, "detected_count": len(boxes),
        "count_ok": count_ok, "source_boxes_nonoverlapping": bbox_nonoverlap,
        "passed": count_ok and bbox_nonoverlap and all(item["passed"] for item in items),
        "boxes": items,
    }


def make_contact_sheet(output_dir: Path, report: dict) -> Path:
    columns, cell_width, cell_height = 5, 310, 510
    count = len(report["boxes"])
    canvas = Image.new("RGB", (columns * cell_width, ((count + columns - 1) // columns) * cell_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(report["boxes"]):
        path = output_dir / "previews" / f"{item['box_id']}_before_after.jpg"
        if not path.is_file():
            continue
        with Image.open(path) as image:
            image.thumbnail((cell_width - 12, cell_height - 35), Image.Resampling.BILINEAR)
            x, y = (index % columns) * cell_width + 6, (index // columns) * cell_height + 30
            canvas.paste(image.convert("RGB"), (x, y))
        draw.text((x, y - 22), f"{item['box_id']}  residual {item.get('residual_angle_deg', 'n/a')} deg", fill=(0, 0, 0))
    path = output_dir / "qa_orange_contact_sheet.jpg"
    canvas.save(path, quality=90)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--max-residual-deg", type=float, default=0.20)
    args = parser.parse_args()
    report = verify_batch(args.output_dir, args.expected_count, args.max_residual_deg)
    output_path = args.output_dir / "qa_orange_batch.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sheet = make_contact_sheet(args.output_dir, report)
    print(f"passed={report['passed']} boxes={report['detected_count']} report={output_path} contact_sheet={sheet}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
