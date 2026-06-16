from __future__ import annotations

import argparse
import json

from .config import M11Config
from .pipeline import run_m11_rotation_correction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M1-1 core-box rotation correction")
    parser.add_argument("--input", required=True, help="Input .dat/.tif/.png/.jpg path")
    parser.add_argument("--hdr", default=None, help="Optional ENVI .hdr path")
    parser.add_argument("--output", default="outputs/m1_1", help="Output directory")
    parser.add_argument("--thumbnail-width", type=int, default=512)
    parser.add_argument("--manual-angle-delta", type=float, default=0.0)
    parser.add_argument("--expected-box-count", type=int, default=None)
    parser.add_argument("--no-preview", action="store_true", help="Do not write QA preview images")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = M11Config(
        thumbnail_width=args.thumbnail_width,
        manual_angle_delta_deg=args.manual_angle_delta,
        expected_box_count=args.expected_box_count,
        save_previews=not args.no_preview,
    )
    result = run_m11_rotation_correction(
        input_path=args.input,
        hdr_path=args.hdr,
        output_dir=args.output,
        config=config,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

