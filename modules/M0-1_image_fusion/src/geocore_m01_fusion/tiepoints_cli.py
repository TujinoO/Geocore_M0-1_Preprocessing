"""CLI helpers for interactive tie-point session workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api_service import (
    apply_tie_point_session_to_envi,
    available_api_operations,
    create_tie_point_session_from_roi,
    update_tie_point_session,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Geo-Core M0-1 interactive tie-point utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create-session", help="Create editable tie-point session from ROI manifest")
    create.add_argument("--roi-dir", required=True)
    create.add_argument("--stage", required=True)
    create.add_argument("--output-dir")
    create.add_argument("--no-preview", action="store_true")

    edit = sub.add_parser("edit-session", help="Apply frontend-style operations to a tie-point session")
    edit.add_argument("--session", required=True)
    edit.add_argument("--operations-json")
    edit.add_argument("--operations-file")
    edit.add_argument("--roi-dir")
    edit.add_argument("--no-preview", action="store_true")

    apply = sub.add_parser("apply-session", help="Warp an ENVI cube using active tie points")
    apply.add_argument("--session", required=True)
    apply.add_argument("--input-hdr", required=True)
    apply.add_argument("--input-dat")
    apply.add_argument("--output-hdr", required=True)
    apply.add_argument("--output-dat")
    apply.add_argument("--min-active-points", type=int, default=3)

    sub.add_parser("api-summary", help="Print supported API operations")

    args = parser.parse_args(argv)
    if args.command == "create-session":
        result = create_tie_point_session_from_roi(
            args.roi_dir,
            stage=args.stage,
            output_dir=args.output_dir,
            render_preview=not args.no_preview,
        )
    elif args.command == "edit-session":
        operations = _load_operations(args.operations_json, args.operations_file)
        result = update_tie_point_session(
            args.session,
            operations,
            roi_dir=args.roi_dir,
            render_preview=not args.no_preview,
        )
    elif args.command == "apply-session":
        result = apply_tie_point_session_to_envi(
            args.session,
            input_hdr=args.input_hdr,
            input_dat=args.input_dat,
            output_hdr=args.output_hdr,
            output_dat=args.output_dat,
            min_active_points=args.min_active_points,
        )
    else:
        result = available_api_operations()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _load_operations(operations_json: str | None, operations_file: str | None) -> list[dict[str, object]]:
    if operations_json:
        data = json.loads(operations_json)
    elif operations_file:
        data = json.loads(Path(operations_file).read_text(encoding="utf-8"))
    else:
        raise ValueError("Either --operations-json or --operations-file is required")
    if not isinstance(data, list):
        raise ValueError("Tie-point operations must be a JSON list")
    return [dict(item) for item in data]


if __name__ == "__main__":
    raise SystemExit(main())
