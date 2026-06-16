"""Interactive tie-point editing and visualization utilities.

This module keeps the algorithm layer independent from a web framework.  A
backend API can persist the JSON session, let the frontend edit point status or
add manual points, then rebuild a LocalWarpModel from active points.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .registration import LocalWarpModel, TiePoint, warp_cube_with_model
from .resample import resize_cube


ACTIVE_STATUS = "active"
REJECTED_STATUS = "rejected"
DELETED_STATUS = "deleted"
VALID_STATUSES = {ACTIVE_STATUS, REJECTED_STATUS, DELETED_STATUS}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


@dataclass(slots=True)
class InteractiveTiePoint:
    id: str
    ref_y: float
    ref_x: float
    moving_y: float
    moving_x: float
    approx_y: float
    approx_x: float
    score: float
    status: str = ACTIVE_STATUS
    source: str = "auto"
    note: str = ""

    @property
    def delta_y(self) -> float:
        return self.moving_y - self.approx_y

    @property
    def delta_x(self) -> float:
        return self.moving_x - self.approx_x

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ref_y": float(self.ref_y),
            "ref_x": float(self.ref_x),
            "moving_y": float(self.moving_y),
            "moving_x": float(self.moving_x),
            "approx_y": float(self.approx_y),
            "approx_x": float(self.approx_x),
            "delta_y": float(self.delta_y),
            "delta_x": float(self.delta_x),
            "score": float(self.score),
            "status": self.status,
            "source": self.source,
            "note": self.note,
        }


@dataclass(slots=True)
class TiePointSession:
    session_id: str
    stage: str
    reference_name: str
    moving_name: str
    reference_shape: tuple[int, int]
    moving_shape: tuple[int, int]
    approx_offset_y: float
    approx_offset_x: float
    approx_scale_y: float
    approx_scale_x: float
    points: list[InteractiveTiePoint] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        quality = session_quality(self)
        return {
            "schema_version": "m0_tie_point_session.v1",
            "session_id": self.session_id,
            "stage": self.stage,
            "reference_name": self.reference_name,
            "moving_name": self.moving_name,
            "reference_shape": [int(x) for x in self.reference_shape],
            "moving_shape": [int(x) for x in self.moving_shape],
            "approx_offset_y": float(self.approx_offset_y),
            "approx_offset_x": float(self.approx_offset_x),
            "approx_scale_y": float(self.approx_scale_y),
            "approx_scale_x": float(self.approx_scale_x),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "quality": quality,
            "points": [point.to_dict() for point in self.points],
        }


def create_session_from_model(
    model: LocalWarpModel,
    *,
    stage: str,
    reference_name: str,
    moving_name: str,
    metadata: dict[str, Any] | None = None,
) -> TiePointSession:
    """Create an editable session from an automatic LocalWarpModel."""

    points: list[InteractiveTiePoint] = []
    for idx, point in enumerate(model.tie_points):
        points.append(_interactive_from_tie_point(point, idx=idx, status=ACTIVE_STATUS, source="auto"))
    start = len(points)
    for idx, point in enumerate(model.rejected_tie_points, start=start):
        points.append(_interactive_from_tie_point(point, idx=idx, status=REJECTED_STATUS, source="auto_rejected"))
    return TiePointSession(
        session_id=str(uuid.uuid4()),
        stage=stage,
        reference_name=reference_name,
        moving_name=moving_name,
        reference_shape=tuple(int(x) for x in model.reference_shape),
        moving_shape=tuple(int(x) for x in model.moving_shape),
        approx_offset_y=float(model.approx_offset_y),
        approx_offset_x=float(model.approx_offset_x),
        approx_scale_y=float(model.approx_scale_y),
        approx_scale_x=float(model.approx_scale_x),
        points=points,
        metadata=metadata or {},
    )


def session_from_dict(data: dict[str, Any]) -> TiePointSession:
    points = []
    for raw in data.get("points", []):
        status = str(raw.get("status", ACTIVE_STATUS))
        _validate_status(status)
        points.append(
            InteractiveTiePoint(
                id=str(raw.get("id") or uuid.uuid4()),
                ref_y=float(raw["ref_y"]),
                ref_x=float(raw["ref_x"]),
                moving_y=float(raw["moving_y"]),
                moving_x=float(raw["moving_x"]),
                approx_y=float(raw.get("approx_y", raw["moving_y"])),
                approx_x=float(raw.get("approx_x", raw["moving_x"])),
                score=float(raw.get("score", 1.0)),
                status=status,
                source=str(raw.get("source", "manual")),
                note=str(raw.get("note", "")),
            )
        )
    return TiePointSession(
        session_id=str(data.get("session_id") or uuid.uuid4()),
        stage=str(data["stage"]),
        reference_name=str(data.get("reference_name", "reference")),
        moving_name=str(data.get("moving_name", "moving")),
        reference_shape=tuple(int(x) for x in data["reference_shape"]),
        moving_shape=tuple(int(x) for x in data["moving_shape"]),
        approx_offset_y=float(data.get("approx_offset_y", 0.0)),
        approx_offset_x=float(data.get("approx_offset_x", 0.0)),
        approx_scale_y=float(data.get("approx_scale_y", 1.0)),
        approx_scale_x=float(data.get("approx_scale_x", 1.0)),
        points=points,
        created_at=str(data.get("created_at") or _now()),
        updated_at=str(data.get("updated_at") or _now()),
        metadata=dict(data.get("metadata", {})),
    )


def read_session(path: str | Path) -> TiePointSession:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return session_from_dict(data)


def write_session(session: TiePointSession, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(session.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def add_tie_point(
    session: TiePointSession,
    *,
    ref_y: float,
    ref_x: float,
    moving_y: float,
    moving_x: float,
    score: float = 1.0,
    status: str = ACTIVE_STATUS,
    source: str = "manual",
    note: str = "",
) -> InteractiveTiePoint:
    _validate_status(status)
    approx_y = ref_y * session.approx_scale_y + session.approx_offset_y
    approx_x = ref_x * session.approx_scale_x + session.approx_offset_x
    point = InteractiveTiePoint(
        id=str(uuid.uuid4()),
        ref_y=float(ref_y),
        ref_x=float(ref_x),
        moving_y=float(moving_y),
        moving_x=float(moving_x),
        approx_y=float(approx_y),
        approx_x=float(approx_x),
        score=float(score),
        status=status,
        source=source,
        note=note,
    )
    _validate_point_bounds(point, session)
    session.points.append(point)
    _touch(session)
    return point


def update_tie_point(session: TiePointSession, point_id: str, **updates: Any) -> InteractiveTiePoint:
    point = _find_point(session, point_id)
    for key in ["ref_y", "ref_x", "moving_y", "moving_x", "approx_y", "approx_x", "score"]:
        if key in updates:
            setattr(point, key, float(updates[key]))
    if "status" in updates:
        status = str(updates["status"])
        _validate_status(status)
        point.status = status
    if "source" in updates:
        point.source = str(updates["source"])
    if "note" in updates:
        point.note = str(updates["note"])
    _validate_point_bounds(point, session)
    _touch(session)
    return point


def set_tie_point_status(session: TiePointSession, point_id: str, status: str) -> InteractiveTiePoint:
    _validate_status(status)
    point = _find_point(session, point_id)
    point.status = status
    _touch(session)
    return point


def apply_tie_point_operations(session: TiePointSession, operations: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply frontend edit operations to a session and return changed point IDs."""

    changed: list[str] = []
    for op in operations:
        name = str(op.get("op", "")).lower()
        if name == "add":
            point = add_tie_point(
                session,
                ref_y=float(op["ref_y"]),
                ref_x=float(op["ref_x"]),
                moving_y=float(op["moving_y"]),
                moving_x=float(op["moving_x"]),
                score=float(op.get("score", 1.0)),
                status=str(op.get("status", ACTIVE_STATUS)),
                source=str(op.get("source", "manual")),
                note=str(op.get("note", "")),
            )
            changed.append(point.id)
        elif name == "update":
            point_id = str(op["id"])
            updates = {k: v for k, v in op.items() if k not in {"op", "id"}}
            point = update_tie_point(session, point_id, **updates)
            changed.append(point.id)
        elif name in {"reject", "delete", "activate", "restore"}:
            point_id = str(op["id"])
            status = {
                "reject": REJECTED_STATUS,
                "delete": DELETED_STATUS,
                "activate": ACTIVE_STATUS,
                "restore": ACTIVE_STATUS,
            }[name]
            point = set_tie_point_status(session, point_id, status)
            changed.append(point.id)
        else:
            raise ValueError(f"Unsupported tie-point operation: {name}")
    return {"changed_point_ids": changed, "quality": session_quality(session)}


def build_warp_model_from_session(session: TiePointSession, *, min_active_points: int = 3) -> LocalWarpModel:
    active = [point for point in session.points if point.status == ACTIVE_STATUS]
    if len(active) < min_active_points:
        raise ValueError(f"At least {min_active_points} active tie points are required for local warp")
    rejected = [point for point in session.points if point.status != ACTIVE_STATUS]
    return LocalWarpModel(
        reference_shape=session.reference_shape,
        moving_shape=session.moving_shape,
        approx_offset_y=session.approx_offset_y,
        approx_offset_x=session.approx_offset_x,
        approx_scale_y=session.approx_scale_y,
        approx_scale_x=session.approx_scale_x,
        tie_points=[_tie_point_from_interactive(point) for point in active],
        rejected_tie_points=[_tie_point_from_interactive(point) for point in rejected],
        method="interactive_tie_points_idw_warp",
    )


def warp_cube_with_session(cube: np.ndarray, session: TiePointSession, *, min_active_points: int = 3) -> np.ndarray:
    model = build_warp_model_from_session(session, min_active_points=min_active_points)
    return warp_cube_with_model(cube, model)


def session_quality(session: TiePointSession) -> dict[str, Any]:
    active = [point for point in session.points if point.status == ACTIVE_STATUS]
    rejected = [point for point in session.points if point.status == REJECTED_STATUS]
    deleted = [point for point in session.points if point.status == DELETED_STATUS]
    manual = [point for point in session.points if point.source.startswith("manual")]
    scores = np.asarray([point.score for point in active], dtype=np.float32)
    dy = np.asarray([point.delta_y for point in active], dtype=np.float32)
    dx = np.asarray([point.delta_x for point in active], dtype=np.float32)
    warnings: list[str] = []
    if len(active) < 8:
        warnings.append("active_tie_point_count_below_recommended_8")
    if scores.size and float(np.mean(scores)) < 0.08:
        warnings.append("mean_tie_point_score_low")
    return {
        "total_count": len(session.points),
        "active_count": len(active),
        "rejected_count": len(rejected),
        "deleted_count": len(deleted),
        "manual_count": len(manual),
        "mean_score": _safe_mean(scores),
        "median_delta_y": _safe_median(dy),
        "median_delta_x": _safe_median(dx),
        "delta_y_std": _safe_std(dy),
        "delta_x_std": _safe_std(dx),
        "warnings": warnings,
    }


def render_tie_point_visualization(
    reference: np.ndarray,
    moving: np.ndarray,
    session: TiePointSession,
    output_dir: str | Path,
    *,
    max_panel_size: int = 720,
    include_rejected: bool = True,
) -> dict[str, str]:
    """Render frontend-friendly tie-point preview images."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_u8 = _to_uint8_rgb(reference)
    mov_u8 = _to_uint8_rgb(moving)
    side = _render_side_by_side(ref_u8, mov_u8, session, max_panel_size, include_rejected=include_rejected)
    ref_overlay = _render_points_on_single(ref_u8, session, max_panel_size, which="reference", include_rejected=include_rejected)
    moving_overlay = _render_points_on_single(mov_u8, session, max_panel_size, which="moving", include_rejected=include_rejected)
    checker = _render_checkerboard(reference, moving, session, max_panel_size)
    paths = {
        "side_by_side": str(out / f"{session.stage}_tiepoints_side_by_side.png"),
        "reference_overlay": str(out / f"{session.stage}_reference_points.png"),
        "moving_overlay": str(out / f"{session.stage}_moving_points.png"),
        "checkerboard": str(out / f"{session.stage}_checkerboard.png"),
    }
    side.save(paths["side_by_side"])
    ref_overlay.save(paths["reference_overlay"])
    moving_overlay.save(paths["moving_overlay"])
    checker.save(paths["checkerboard"])
    return paths


def render_tie_point_context_visualization(
    reference_context: np.ndarray,
    moving_context: np.ndarray,
    session: TiePointSession,
    output_dir: str | Path,
    *,
    max_panel_size: int = 720,
    include_rejected: bool = True,
) -> dict[str, str]:
    """Render human-readable context previews for frontend inspection.

    Coordinates in the session remain in the low-resolution registration grid;
    this renderer scales them onto full-size RGB/false-color context images.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_u8 = _to_uint8_rgb(reference_context)
    mov_u8 = _to_uint8_rgb(moving_context)
    side = _render_context_side_by_side(ref_u8, mov_u8, session, max_panel_size, include_rejected=include_rejected)
    ref_overlay = _render_context_points_on_single(
        ref_u8,
        session,
        max_panel_size,
        which="reference",
        include_rejected=include_rejected,
    )
    moving_overlay = _render_context_points_on_single(
        mov_u8,
        session,
        max_panel_size,
        which="moving",
        include_rejected=include_rejected,
    )
    paths = {
        "context_side_by_side": str(out / f"{session.stage}_context_side_by_side.png"),
        "context_reference_overlay": str(out / f"{session.stage}_context_reference_points.png"),
        "context_moving_overlay": str(out / f"{session.stage}_context_moving_points.png"),
    }
    side.save(paths["context_side_by_side"])
    ref_overlay.save(paths["context_reference_overlay"])
    moving_overlay.save(paths["context_moving_overlay"])
    return paths


def _render_side_by_side(
    reference: np.ndarray,
    moving: np.ndarray,
    session: TiePointSession,
    max_panel_size: int,
    *,
    include_rejected: bool,
) -> Image.Image:
    ref_img, ref_scale = _resize_for_panel(reference, max_panel_size)
    mov_img, mov_scale = _resize_for_panel(moving, max_panel_size)
    gap = 24
    canvas = Image.new("RGB", (ref_img.width + gap + mov_img.width, max(ref_img.height, mov_img.height)), "white")
    canvas.paste(ref_img, (0, 0))
    moving_x0 = ref_img.width + gap
    canvas.paste(mov_img, (moving_x0, 0))
    draw = ImageDraw.Draw(canvas)
    for point in session.points:
        if not _should_draw(point, include_rejected):
            continue
        color = _point_color(point)
        rx = point.ref_x * ref_scale
        ry = point.ref_y * ref_scale
        mx = moving_x0 + point.moving_x * mov_scale
        my = point.moving_y * mov_scale
        _draw_cross(draw, rx, ry, color)
        _draw_cross(draw, mx, my, color)
        draw.line((rx, ry, mx, my), fill=color, width=1)
    return canvas


def _render_context_side_by_side(
    reference: np.ndarray,
    moving: np.ndarray,
    session: TiePointSession,
    max_panel_size: int,
    *,
    include_rejected: bool,
) -> Image.Image:
    ref_img, ref_display_scale = _resize_for_panel(reference, max_panel_size)
    mov_img, mov_display_scale = _resize_for_panel(moving, max_panel_size)
    ref_map_y = reference.shape[0] / float(session.reference_shape[0])
    ref_map_x = reference.shape[1] / float(session.reference_shape[1])
    mov_map_y = moving.shape[0] / float(session.moving_shape[0])
    mov_map_x = moving.shape[1] / float(session.moving_shape[1])
    gap = 24
    canvas = Image.new("RGB", (ref_img.width + gap + mov_img.width, max(ref_img.height, mov_img.height)), "white")
    canvas.paste(ref_img, (0, 0))
    moving_x0 = ref_img.width + gap
    canvas.paste(mov_img, (moving_x0, 0))
    draw = ImageDraw.Draw(canvas)
    for point in session.points:
        if not _should_draw(point, include_rejected):
            continue
        color = _point_color(point)
        rx = point.ref_x * ref_map_x * ref_display_scale
        ry = point.ref_y * ref_map_y * ref_display_scale
        mx = moving_x0 + point.moving_x * mov_map_x * mov_display_scale
        my = point.moving_y * mov_map_y * mov_display_scale
        _draw_cross(draw, rx, ry, color)
        _draw_cross(draw, mx, my, color)
        draw.line((rx, ry, mx, my), fill=color, width=1)
    return canvas


def _render_points_on_single(
    image: np.ndarray,
    session: TiePointSession,
    max_panel_size: int,
    *,
    which: str,
    include_rejected: bool,
) -> Image.Image:
    img, scale = _resize_for_panel(image, max_panel_size)
    draw = ImageDraw.Draw(img)
    for point in session.points:
        if not _should_draw(point, include_rejected):
            continue
        color = _point_color(point)
        y = point.ref_y if which == "reference" else point.moving_y
        x = point.ref_x if which == "reference" else point.moving_x
        _draw_cross(draw, x * scale, y * scale, color)
    return img


def _render_context_points_on_single(
    image: np.ndarray,
    session: TiePointSession,
    max_panel_size: int,
    *,
    which: str,
    include_rejected: bool,
) -> Image.Image:
    img, display_scale = _resize_for_panel(image, max_panel_size)
    if which == "reference":
        map_y = image.shape[0] / float(session.reference_shape[0])
        map_x = image.shape[1] / float(session.reference_shape[1])
    else:
        map_y = image.shape[0] / float(session.moving_shape[0])
        map_x = image.shape[1] / float(session.moving_shape[1])
    draw = ImageDraw.Draw(img)
    for point in session.points:
        if not _should_draw(point, include_rejected):
            continue
        color = _point_color(point)
        y = point.ref_y if which == "reference" else point.moving_y
        x = point.ref_x if which == "reference" else point.moving_x
        _draw_cross(draw, x * map_x * display_scale, y * map_y * display_scale, color)
    return img


def _render_checkerboard(reference: np.ndarray, moving: np.ndarray, session: TiePointSession, max_panel_size: int) -> Image.Image:
    ref = _normalize_gray(reference)
    mov = _normalize_gray(moving)
    shape = session.reference_shape
    if mov.shape != session.moving_shape:
        mov = resize_cube(mov, session.moving_shape, method="bilinear")
    try:
        mov = warp_cube_with_session(mov, session, min_active_points=3)
    except ValueError:
        if mov.shape != shape:
            mov = resize_cube(mov, shape, method="bilinear")
        pass
    if ref.shape != shape:
        ref = resize_cube(ref, shape, method="bilinear")
    block = max(12, min(shape) // 8)
    yy, xx = np.indices(shape)
    mask = ((yy // block + xx // block) % 2).astype(bool)
    checker = np.where(mask, ref, mov)
    rgb = np.stack([checker, checker, checker], axis=2)
    img, _ = _resize_for_panel((rgb * 255).round().astype(np.uint8), max_panel_size)
    return img


def _interactive_from_tie_point(point: TiePoint, *, idx: int, status: str, source: str) -> InteractiveTiePoint:
    return InteractiveTiePoint(
        id=f"tp_{idx:04d}",
        ref_y=float(point.ref_y),
        ref_x=float(point.ref_x),
        moving_y=float(point.moving_y),
        moving_x=float(point.moving_x),
        approx_y=float(point.approx_y),
        approx_x=float(point.approx_x),
        score=float(point.score),
        status=status,
        source=source,
    )


def _tie_point_from_interactive(point: InteractiveTiePoint) -> TiePoint:
    return TiePoint(
        ref_y=float(point.ref_y),
        ref_x=float(point.ref_x),
        moving_y=float(point.moving_y),
        moving_x=float(point.moving_x),
        approx_y=float(point.approx_y),
        approx_x=float(point.approx_x),
        score=float(point.score),
    )


def _find_point(session: TiePointSession, point_id: str) -> InteractiveTiePoint:
    for point in session.points:
        if point.id == point_id:
            return point
    raise KeyError(f"Tie point not found: {point_id}")


def _validate_status(status: str) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Unsupported tie-point status: {status}")


def _validate_point_bounds(point: InteractiveTiePoint, session: TiePointSession) -> None:
    ref_h, ref_w = session.reference_shape
    mov_h, mov_w = session.moving_shape
    if not (0 <= point.ref_y < ref_h and 0 <= point.ref_x < ref_w):
        raise ValueError("Reference coordinate is outside reference image bounds")
    if not (0 <= point.moving_y < mov_h and 0 <= point.moving_x < mov_w):
        raise ValueError("Moving coordinate is outside moving image bounds")


def _touch(session: TiePointSession) -> None:
    session.updated_at = _now()


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
        if rgb.dtype == np.uint8:
            return rgb
        gray = _normalize_gray(rgb)
        return (np.stack([gray, gray, gray], axis=2) * 255).round().astype(np.uint8)
    gray = _normalize_gray(arr)
    return (np.stack([gray, gray, gray], axis=2) * 255).round().astype(np.uint8)


def _normalize_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.nanmean(arr[:, :, : min(arr.shape[2], 3)], axis=2)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _resize_for_panel(image: np.ndarray, max_panel_size: int) -> tuple[Image.Image, float]:
    arr = np.asarray(image)
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_panel_size) / float(max(h, w)))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    img = Image.fromarray(arr.astype(np.uint8, copy=False))
    if scale != 1.0:
        img = img.resize((out_w, out_h), Image.Resampling.BILINEAR)
    return img.convert("RGB"), scale


def _should_draw(point: InteractiveTiePoint, include_rejected: bool) -> bool:
    if point.status == DELETED_STATUS:
        return False
    if point.status == REJECTED_STATUS and not include_rejected:
        return False
    return True


def _point_color(point: InteractiveTiePoint) -> tuple[int, int, int]:
    if point.status == REJECTED_STATUS:
        return (255, 128, 0)
    if point.source.startswith("manual"):
        return (255, 230, 0)
    return (0, 220, 255)


def _draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int]) -> None:
    r = 4
    draw.line((x - r, y, x + r, y), fill=color, width=2)
    draw.line((x, y - r, x, y + r), fill=color, width=2)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=1)


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _safe_median(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _safe_std(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.std(finite))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
