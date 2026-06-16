from __future__ import annotations

import fnmatch
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from skimage import draw, io

from geocore_mask.inference.predictor import CoreMaskPredictor
from geocore_mask.postprocess.contour import mask_to_components
from geocore_mask.utils.image_io import read_image, save_mask_png, save_overlay_png, save_tiff_like
from geocore_mask.utils.model_package import build_ad_hoc_manifest, load_model_package

from .schemas import ForegroundMaskRequest, JobStatus, MaskEditRequest, TaskInfo


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "api_runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"

TASKS: Dict[str, TaskInfo] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_task() -> TaskInfo:
    task_id = uuid4().hex
    now = utc_now()
    task = TaskInfo(task_id=task_id, status=JobStatus.pending, created_at=now, updated_at=now)
    TASKS[task_id] = task
    return task


def update_task(task_id: str, **changes) -> TaskInfo:
    task = TASKS[task_id].copy(update={**changes, "updated_at": utc_now()})
    TASKS[task_id] = task
    return task


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_csv_floats(value: Optional[str]) -> Optional[List[float]]:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def resolve_input_paths(input_path: str, image_pattern: str, recursive: bool) -> List[str]:
    root = Path(input_path)
    if root.is_file():
        return [str(root)]
    if not root.is_dir():
        raise FileNotFoundError(f"input_path does not exist: {input_path}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    paths = [str(path) for path in iterator if path.is_file() and fnmatch.fnmatch(path.name, image_pattern)]
    if not paths:
        raise FileNotFoundError(f"No input images matched {image_pattern} under {input_path}")
    return paths


def build_predictor(request: ForegroundMaskRequest) -> CoreMaskPredictor:
    if request.model_path:
        package = build_ad_hoc_manifest(
            model_name=request.model_name or "UNet",
            model_path=request.model_path,
            input_channels=request.band_num,
            num_classes=request.num_classes,
            mean=request.mean,
            std=request.std,
            tile_size=request.target_size,
            overlap=request.overlap_rate,
            threshold=request.threshold if request.threshold is not None else 0.5,
        )
    else:
        package = load_model_package(model_profile=request.model_profile, model_package=request.model_package)
    return CoreMaskPredictor(package)


def run_foreground_mask(task_id: str, request: ForegroundMaskRequest) -> None:
    try:
        ensure_runtime_dirs()
        output_dir = Path(request.output_dir) if request.output_dir else OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        input_paths = resolve_input_paths(request.input_path, request.image_pattern, request.recursive)
        update_task(
            task_id,
            status=JobStatus.running,
            message="loading foreground mask model",
            input_paths=input_paths,
            output_dir=str(output_dir),
        )

        predictor = build_predictor(request)
        all_outputs: Dict[str, str] = {}
        all_metrics: Dict[str, float] = {}
        all_warnings: List[str] = []
        result_index = []

        for index, input_path in enumerate(input_paths):
            update_task(task_id, message=f"running foreground mask inference ({index + 1}/{len(input_paths)})")
            per_output_dir = output_dir if len(input_paths) == 1 else output_dir / Path(input_path).stem
            result = predictor.predict(
                input_path,
                per_output_dir,
                threshold=request.threshold,
                enable_postprocess=request.enable_postprocess,
                output_preview=request.output_preview,
            )
            prefix = "" if len(input_paths) == 1 else f"{Path(input_path).stem}."
            for key, value in result["output_files"].items():
                all_outputs[f"{prefix}{key}"] = value
            all_warnings.extend(result.get("warnings", []))
            result_index.append({"input_path": input_path, **result})

            if len(input_paths) == 1:
                all_metrics = result.get("metrics", {})

        if len(input_paths) > 1:
            index_path = output_dir / "results.json"
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(result_index, f, ensure_ascii=False, indent=2)
            all_outputs["results_json"] = str(index_path)
            all_metrics = {"image_count": float(len(input_paths))}

        update_task(
            task_id,
            status=JobStatus.succeeded,
            message="foreground mask extraction completed",
            output_files=all_outputs,
            metrics=all_metrics,
            warnings=all_warnings,
        )
    except Exception as exc:
        update_task(task_id, status=JobStatus.failed, message="foreground mask extraction failed", error=str(exc))


def _read_binary_mask(path: str | Path) -> np.ndarray:
    array = io.imread(str(path))
    if array.ndim == 3:
        array = array[:, :, 0]
    return (array > 0).astype(np.uint8)


def _output_file(task: TaskInfo, key: str, result_key_prefix: str = "") -> Optional[str]:
    prefixed = f"{result_key_prefix}{key}"
    if prefixed in task.output_files:
        return task.output_files[prefixed]
    if result_key_prefix == "":
        return task.output_files.get(key)
    return None


def _scale_point(x: float, y: float, request: MaskEditRequest, width: int, height: int) -> tuple[float, float]:
    if request.display_width and request.display_height:
        x = x * width / request.display_width
        y = y * height / request.display_height
    return x, y


def _draw_rectangle(selection: np.ndarray, bbox: List[float], request: MaskEditRequest) -> None:
    height, width = selection.shape
    if len(bbox) != 4:
        raise ValueError("rectangle region requires bbox [x1, y1, x2, y2]")

    x1, y1 = _scale_point(float(bbox[0]), float(bbox[1]), request, width, height)
    x2, y2 = _scale_point(float(bbox[2]), float(bbox[3]), request, width, height)
    left = max(0, min(width, math.floor(min(x1, x2))))
    right = max(0, min(width, math.ceil(max(x1, x2))))
    top = max(0, min(height, math.floor(min(y1, y2))))
    bottom = max(0, min(height, math.ceil(max(y1, y2))))
    if right > left and bottom > top:
        selection[top:bottom, left:right] = True


def _scaled_points(points: List[List[float]], request: MaskEditRequest, width: int, height: int) -> np.ndarray:
    scaled = []
    for point in points:
        if len(point) != 2:
            raise ValueError("points must be [x, y] pairs")
        scaled.append(_scale_point(float(point[0]), float(point[1]), request, width, height))
    return np.asarray(scaled, dtype=np.float32)


def _draw_polygon(selection: np.ndarray, points: List[List[float]], request: MaskEditRequest) -> None:
    height, width = selection.shape
    if len(points) < 3:
        raise ValueError("polygon region requires at least 3 points")
    scaled = _scaled_points(points, request, width, height)
    rr, cc = draw.polygon(scaled[:, 1], scaled[:, 0], shape=selection.shape)
    selection[rr, cc] = True


def _draw_brush(selection: np.ndarray, points: List[List[float]], radius: Optional[float], request: MaskEditRequest) -> None:
    height, width = selection.shape
    if not points:
        raise ValueError("brush region requires points")

    scaled = _scaled_points(points, request, width, height)
    brush_radius = float(radius if radius is not None else 8.0)
    if request.display_width and request.display_height:
        brush_radius *= (width / request.display_width + height / request.display_height) / 2
    brush_radius = max(1.0, brush_radius)

    for index, point in enumerate(scaled):
        if index == 0:
            line_rows = np.asarray([int(round(point[1]))])
            line_cols = np.asarray([int(round(point[0]))])
        else:
            prev = scaled[index - 1]
            line_rows, line_cols = draw.line(
                int(round(prev[1])),
                int(round(prev[0])),
                int(round(point[1])),
                int(round(point[0])),
            )

        sample_step = max(1, int(brush_radius // 2))
        for row, col in zip(line_rows[::sample_step], line_cols[::sample_step]):
            rr, cc = draw.disk((row, col), brush_radius, shape=selection.shape)
            selection[rr, cc] = True


def rasterize_edit_regions(request: MaskEditRequest, shape: tuple[int, int]) -> np.ndarray:
    selection = np.zeros(shape, dtype=bool)
    for region in request.regions:
        if region.type == "rectangle":
            if region.bbox is None:
                raise ValueError("rectangle region requires bbox")
            _draw_rectangle(selection, region.bbox, request)
        elif region.type == "polygon":
            if region.points is None:
                raise ValueError("polygon region requires points")
            _draw_polygon(selection, region.points, request)
        elif region.type == "brush":
            if region.points is None:
                raise ValueError("brush region requires points")
            _draw_brush(selection, region.points, region.radius, request)
        else:
            raise ValueError(f"unsupported edit region type: {region.type}")
    return selection


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def apply_mask_edit(task_id: str, request: MaskEditRequest) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise KeyError(task_id)

    task = TASKS[task_id]
    if task.status != JobStatus.succeeded:
        raise RuntimeError("task must be succeeded before mask editing")

    mask_tif = _output_file(task, "mask_tif", request.result_key_prefix)
    mask_png = _output_file(task, "mask_png", request.result_key_prefix)
    contours_json = _output_file(task, "contours_json", request.result_key_prefix)
    metadata_json = _output_file(task, "metadata_json", request.result_key_prefix)
    overlay_png = _output_file(task, "overlay_png", request.result_key_prefix)

    mask_path = mask_tif or mask_png
    if not mask_path:
        raise FileNotFoundError("task has no mask_tif or mask_png output to edit")

    mask = _read_binary_mask(mask_path)
    selection = rasterize_edit_regions(request, mask.shape)
    old_foreground = int(mask.sum())

    if request.operation == "remove":
        edited_mask = np.logical_and(mask.astype(bool), ~selection).astype(np.uint8)
    else:
        raise ValueError(f"unsupported mask edit operation: {request.operation}")

    new_foreground = int(edited_mask.sum())
    removed_pixels = old_foreground - new_foreground
    components = mask_to_components(edited_mask)

    reference = None
    image = None
    metadata = _load_json(metadata_json)
    input_path = metadata.get("input_path") or (task.input_paths[0] if task.input_paths else None)
    if input_path:
        try:
            image_data = read_image(input_path)
            reference = image_data
            image = image_data.array
        except Exception:
            reference = None
            image = None

    if mask_png:
        save_mask_png(edited_mask, mask_png)
    if mask_tif:
        save_tiff_like(edited_mask * 255, mask_tif, reference=reference, dtype=np.uint8)
    if contours_json:
        with Path(contours_json).open("w", encoding="utf-8") as f:
            json.dump(components, f, ensure_ascii=False, indent=2)
    if request.output_preview and overlay_png and image is not None:
        save_overlay_png(image, edited_mask, overlay_png)

    edited_at = utc_now()
    warnings = list(metadata.get("warnings", task.warnings))
    if removed_pixels == 0:
        warnings.append("manual edit did not overlap current foreground mask")

    metadata.update(
        {
            "image_width": int(edited_mask.shape[1]),
            "image_height": int(edited_mask.shape[0]),
            "foreground_area_ratio": float(edited_mask.mean()) if edited_mask.size else 0.0,
            "component_count": int(components["component_count"]),
            "warnings": warnings,
            "manual_edit_applied": True,
        }
    )
    history = list(metadata.get("manual_edit_history", []))
    history.append(
        {
            "operation": request.operation,
            "region_count": len(request.regions),
            "selected_pixel_count": int(selection.sum()),
            "removed_pixel_count": int(removed_pixels),
            "edited_at": edited_at,
        }
    )
    metadata["manual_edit_history"] = history
    if metadata_json:
        with Path(metadata_json).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    metrics = {
        "foreground_area_ratio": metadata["foreground_area_ratio"],
        "component_count": float(components["component_count"]),
        "selected_pixel_count": float(selection.sum()),
        "removed_pixel_count": float(removed_pixels),
        "foreground_pixel_count": float(new_foreground),
    }

    updated_task = update_task(
        task_id,
        message="foreground mask manually edited",
        metrics=metrics,
        warnings=warnings,
    )
    return {
        "task_id": task_id,
        "status": updated_task.status,
        "message": updated_task.message,
        "output_files": updated_task.output_files,
        "metrics": metrics,
        "warnings": warnings,
        "edited_at": edited_at,
    }
