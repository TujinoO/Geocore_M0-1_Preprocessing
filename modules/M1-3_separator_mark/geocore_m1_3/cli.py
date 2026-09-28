from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from geocore_m1_3.pipeline import run_segment_depth


def _load_json_arg(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geocore-m1-3", description="M1-3 core segmentation and depth marking tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment = subparsers.add_parser("segment-depth", help="Run the full M1-3 pipeline.")
    segment.add_argument("--image", dest="image_path", help="Corrected core box image path.")
    segment.add_argument("--m1-2-output-dir", dest="m1_2_output_dir", required=True, help="M1-2 output directory.")
    segment.add_argument("--mask-path", help="Override M1-2 mask path.")
    segment.add_argument("--output-dir", required=True, help="Output root directory.")
    segment.add_argument("--hole-id", required=True, help="Drill hole id.")
    segment.add_argument("--core-box-id", required=True, help="Core box id.")
    segment.add_argument("--depth-start-m", type=float, required=True, help="Start depth in meters.")
    segment.add_argument("--depth-end-m", type=float, required=True, help="End depth in meters.")
    segment.add_argument("--config-path", help="Optional JSON config path.")
    segment.add_argument("--task-id", help="Optional deterministic task id.")
    segment.add_argument("--lane-count", type=int, help="Expected lane count.")
    segment.add_argument("--lane-dividers-json", help="JSON list of reviewed internal divider x coordinates in the corrected image.")
    segment.add_argument("--lane-order", choices=["left_to_right", "right_to_left"], help="Lane ordering rule.")
    segment.add_argument("--lane-direction", choices=["top_to_bottom", "bottom_to_top"], help="Lane depth direction.")
    segment.add_argument("--segment-length-cm", type=float, help="Segment length in centimeters.")
    segment.add_argument("--overlap-cm", type=float, help="Segment overlap in centimeters.")
    segment.add_argument("--gap-policy", choices=["close_artificial_gaps", "preserve_all_gaps"], help="Gap handling policy.")
    segment.add_argument("--missing-intervals-json", help="JSON file with missing_intervals list.")
    segment.add_argument("--depth-anchors-json", help="JSON file with depth_anchors list.")
    segment.add_argument("--depth-order-confirmed", action="store_true", help="Assert lane order/direction was verified against field records.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "segment-depth":
        payload: dict[str, Any] = {
            "image_path": args.image_path,
            "m1_2_output_dir": args.m1_2_output_dir,
            "mask_path": args.mask_path,
            "output_dir": args.output_dir,
            "hole_id": args.hole_id,
            "core_box_id": args.core_box_id,
            "depth_start_m": args.depth_start_m,
            "depth_end_m": args.depth_end_m,
            "config_path": args.config_path,
            "task_id": args.task_id,
            "depth_order_confirmed": args.depth_order_confirmed,
        }
        layout: dict[str, Any] = {}
        if args.lane_count is not None:
            layout["lane_count"] = args.lane_count
        if args.lane_dividers_json:
            dividers = json.loads(Path(args.lane_dividers_json).read_text(encoding="utf-8"))
            if not isinstance(dividers, list):
                raise ValueError("--lane-dividers-json must contain a JSON list")
            layout["lane_dividers_x"] = dividers
        if args.lane_order:
            layout["lane_order"] = args.lane_order
        if args.lane_direction:
            layout["lane_direction"] = args.lane_direction
        if layout:
            payload["layout"] = layout
        segmentation: dict[str, Any] = {}
        if args.segment_length_cm is not None:
            segmentation["segment_length_cm"] = args.segment_length_cm
        if args.overlap_cm is not None:
            segmentation["overlap_cm"] = args.overlap_cm
        if segmentation:
            payload["segmentation"] = segmentation
        if args.gap_policy:
            payload["gap_policy"] = args.gap_policy
        if args.missing_intervals_json:
            payload["missing_intervals"] = _load_json_arg(args.missing_intervals_json)
        if args.depth_anchors_json:
            payload["depth_anchors"] = _load_json_arg(args.depth_anchors_json)
        payload = {key: value for key, value in payload.items() if value is not None}
        result = run_segment_depth(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
