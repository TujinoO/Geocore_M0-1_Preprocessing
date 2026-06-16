"""CLI for aligned ROI extraction."""

from __future__ import annotations

import argparse

from .roi import prepare_aligned_roi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare aligned RGB/NIR/SWIR ROI from raw ENVI triplets")
    parser.add_argument("--root", required=True, help="Directory containing RGB/NIR/SWIR .hdr/.dat files")
    parser.add_argument("--output", required=True, help="Output directory for aligned ROI files")
    parser.add_argument("--crop-height", type=int, default=768)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--preview-width", type=int, default=256)
    parser.add_argument("--max-preview-height", type=int, default=3072)
    parser.add_argument("--registration-mode", choices=["scale_only", "phase"], default="scale_only")
    parser.add_argument("--no-local-refine", action="store_true")
    parser.add_argument("--no-anchor-nir-to-swir", action="store_true")
    parser.add_argument("--no-refine-anchored-nir", action="store_true")
    parser.add_argument("--no-local-warp", action="store_true")
    parser.add_argument("--no-final-hsi-rgb-warp", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_aligned_roi(
        args.root,
        args.output,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        preview_width=args.preview_width,
        max_preview_height=args.max_preview_height,
        registration_mode=args.registration_mode,
        local_refine=not args.no_local_refine,
        anchor_nir_to_swir=not args.no_anchor_nir_to_swir,
        refine_anchored_nir=not args.no_refine_anchored_nir,
        local_warp=not args.no_local_warp,
        final_hsi_rgb_warp=not args.no_final_hsi_rgb_warp,
    )
    print(f"Aligned ROI written: {result['output_dir']}")
    print(f"ROI manifest: {result['output_dir']}\\roi_manifest.json")
    print(f"RGB crop: {result['crop_rgb_window']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
