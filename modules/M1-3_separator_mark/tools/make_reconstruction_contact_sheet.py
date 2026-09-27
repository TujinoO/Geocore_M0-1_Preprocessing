"""Make a compact visual index of full-resolution reconstruction previews."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def make_sheet(batch_dir: Path, output_path: Path) -> tuple[Path, int]:
    previews = sorted(batch_dir.glob("box_*_full_qa/reconstructed_strip_preview.jpg"))
    if not previews:
        raise FileNotFoundError(f"No reconstructed-strip previews under {batch_dir}")
    cell_width, cell_height = 126, 2080
    canvas = Image.new("RGB", (cell_width * len(previews), cell_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(previews):
        x = index * cell_width
        draw.text((x + 4, 5), path.parent.name.replace("_full_qa", ""), fill=(0, 0, 0))
        with Image.open(path) as preview:
            preview.thumbnail((cell_width - 8, cell_height - 28), Image.Resampling.BILINEAR)
            canvas.paste(preview.convert("RGB"), (x + (cell_width - preview.width) // 2, 25))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90)
    return output_path, len(previews)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path, count = make_sheet(args.batch_dir, args.output)
    print(f"Contact sheet: {count} boxes -> {path}")


if __name__ == "__main__":
    main()
