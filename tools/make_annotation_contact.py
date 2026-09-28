"""Create a compact QA contact sheet without altering annotation images."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.image_dir.glob("*.png"))
    if not paths:
        raise ValueError("No PNG images found")
    panel_w, panel_h = 280, 700
    cols = min(5, len(paths))
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((panel_w - 8, panel_h - 35))
        x, y = (index % cols) * panel_w, (index // cols) * panel_h
        sheet.paste(preview, (x + (panel_w - preview.width) // 2, y + 30))
        draw.text((x + 8, y + 8), path.stem, fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=88)


if __name__ == "__main__":
    main()
