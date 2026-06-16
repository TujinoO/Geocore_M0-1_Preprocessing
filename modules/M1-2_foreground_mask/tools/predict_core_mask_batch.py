from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geocore_mask.inference.predictor import CoreMaskPredictor
from geocore_mask.utils.model_package import load_model_package


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M1-2 foreground mask extraction for a folder of images.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-package", default="models/core_mask_unet_v2")
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def collect_images(input_dir: Path, patterns: list[str] | None) -> list[Path]:
    if patterns:
        images: list[Path] = []
        for pattern in patterns:
            images.extend(input_dir.glob(pattern))
    else:
        images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted({p.resolve() for p in images})


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    model_package_path = resolve_path(args.model_package)

    images = collect_images(input_dir, args.pattern)
    if not images:
        raise RuntimeError(f"no images found in {input_dir}")

    package = load_model_package(model_package=str(model_package_path))
    predictor = CoreMaskPredictor(package, device=args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for image_path in images:
        sample_output = output_dir / image_path.stem
        result = predictor.predict(
            image_path,
            sample_output,
            threshold=args.threshold,
            enable_postprocess=not args.disable_postprocess,
            output_preview=True,
        )
        summary.append(
            {
                "image": str(image_path),
                "output_dir": str(sample_output),
                "metrics": result["metrics"],
                "warnings": result["warnings"],
                "output_files": result["output_files"],
            }
        )
        print(f"[ok] {image_path.name} -> {sample_output}")

    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
