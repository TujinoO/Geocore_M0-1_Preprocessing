from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import inspect_manifest, run_preprocessing_pipeline
from .paths import load_module_config, workspace_root


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODULE_CONFIG = WORKSPACE_ROOT / "configs" / "module_paths.json"


def _load_module_config() -> dict:
    return load_module_config()


def cmd_modules(_: argparse.Namespace) -> int:
    config = _load_module_config()
    root = workspace_root(config)
    for module_id, module in config["modules"].items():
        path = root / module["path"]
        python_path = root / module["python_path"]
        print(f"{module_id} {module['name']}")
        print(f"  path: {path}")
        print(f"  python_path: {python_path}")
        print(f"  package: {module['package']}")
    return 0


def _load_json(path: str | None):
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_inspect_manifest(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_manifest(args.manifest), ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    payload = {
        "output_dir": args.output_dir,
        "manifest_path": args.manifest,
        "m0_demo": args.m0_demo,
        "m0_mode": args.m0_mode,
        "m0_streaming": args.m0_streaming,
        "rgb_hdr": args.rgb_hdr,
        "rgb_dat": args.rgb_dat,
        "nir_hdr": args.nir_hdr,
        "nir_dat": args.nir_dat,
        "swir_hdr": args.swir_hdr,
        "swir_dat": args.swir_dat,
        "project_id": args.project_id,
        "hole_id": args.hole_id,
        "box_id": args.box_id,
        "core_box_prefix": args.core_box_prefix,
        "depth_start_m": args.depth_start_m,
        "depth_end_m": args.depth_end_m,
        "expected_box_count": args.expected_box_count,
        "limit_boxes": args.limit_boxes,
        "m1_1_input_image": args.m1_1_input_image,
        "m1_1_hdr_path": args.m1_1_hdr_path,
        "m1_2_engine": args.m1_2_engine,
        "m1_2_model_package": args.m1_2_model_package,
        "m1_2_threshold": args.m1_2_threshold,
        "segment_length_cm": args.segment_length_cm,
        "overlap_cm": args.overlap_cm,
        "lane_count": args.lane_count,
        "lane_order": args.lane_order,
        "lane_direction": args.lane_direction,
        "gap_policy": args.gap_policy,
        "missing_intervals": _load_json(args.missing_intervals_json),
        "depth_anchors": _load_json(args.depth_anchors_json),
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    result = run_preprocessing_pipeline(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geocore-preprocess",
        description="Utilities for the unified Geo-Core AI M0/M1 preprocessing workspace.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    modules = subparsers.add_parser("modules", help="Show migrated module locations.")
    modules.set_defaults(func=cmd_modules)

    inspect = subparsers.add_parser("inspect-manifest", help="Inspect an M0-1 manifest and resolved handoff paths.")
    inspect.add_argument("--manifest", required=True, help="Path to M0-1 fusion manifest.json.")
    inspect.set_defaults(func=cmd_inspect_manifest)

    run = subparsers.add_parser("run", help="Run M0-1 -> M1-1 -> M1-2 -> M1-3 preprocessing pipeline.")
    run.add_argument("--output-dir", required=True, help="Unified pipeline output directory.")
    run.add_argument("--manifest", help="Existing M0-1 manifest.json. If omitted, M0 inputs are required.")
    run.add_argument("--m0-demo", action="store_true", help="Run M0-1 on synthetic demo data before M1 processing.")
    run.add_argument("--m0-mode", default="classical", choices=["upsample_only", "fast_preview", "classical", "deep_unsupervised"])
    run.add_argument("--m0-streaming", action="store_true", help="Use M0-1 streaming ENVI fusion path.")
    run.add_argument("--rgb-hdr")
    run.add_argument("--rgb-dat")
    run.add_argument("--nir-hdr")
    run.add_argument("--nir-dat")
    run.add_argument("--swir-hdr")
    run.add_argument("--swir-dat")
    run.add_argument("--project-id")
    run.add_argument("--hole-id")
    run.add_argument("--box-id")
    run.add_argument("--core-box-prefix", help="Prefix used for M1-3 core_box_id values.")
    run.add_argument("--depth-start-m", type=float)
    run.add_argument("--depth-end-m", type=float)
    run.add_argument("--expected-box-count", type=int)
    run.add_argument("--limit-boxes", type=int, help="Process only the first N detected boxes.")
    run.add_argument("--m1-1-input-image", help="Override the image used by M1-1. Defaults to manifest preview_rgb.")
    run.add_argument("--m1-1-hdr-path", help="Optional ENVI header when --m1-1-input-image is a .dat file.")
    run.add_argument("--m1-2-engine", choices=["auto", "model", "classical"], default="auto")
    run.add_argument("--m1-2-model-package", help="Path to M1-2 model package directory containing model_manifest.json.")
    run.add_argument("--m1-2-threshold", type=float)
    run.add_argument("--segment-length-cm", type=float, default=10.0)
    run.add_argument("--overlap-cm", type=float, default=0.0)
    run.add_argument("--lane-count", type=int)
    run.add_argument("--lane-order", choices=["left_to_right", "right_to_left"])
    run.add_argument("--lane-direction", choices=["top_to_bottom", "bottom_to_top"])
    run.add_argument("--gap-policy", choices=["close_artificial_gaps", "preserve_all_gaps"])
    run.add_argument("--missing-intervals-json")
    run.add_argument("--depth-anchors-json")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
