from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import M11Config
from .correction import correct_core_box
from .geometry import estimate_box_angle
from .image_ops import contrast_stretch_uint8, draw_detection_overlay
from .io_envi import read_input_image, save_rgb_image
from .models import CoreBoxResult, M11BatchResult
from .quality import write_qa_report
from .segment_boxes import detect_core_boxes


def run_m11_rotation_correction(
    input_path: str,
    hdr_path: str | None = None,
    output_dir: str = "outputs/m1_1",
    config: M11Config | None = None,
) -> M11BatchResult:
    """Run the complete M1-1 detection, split, rotation-correction and export pipeline."""

    cfg = config or M11Config()
    output_root = Path(output_dir)
    corrected_dir = output_root / "corrected_boxes"
    masks_dir = output_root / "masks"
    previews_dir = output_root / "previews"
    review_sources_dir = output_root / "review_sources"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    review_sources_dir.mkdir(parents=True, exist_ok=True)

    image, _ = read_input_image(input_path, hdr_path)
    detection = detect_core_boxes(image, cfg)
    detection_preview: str | None = None
    if cfg.save_previews:
        detection_preview_path = previews_dir / "strip_detection_overlay.jpg"
        draw_detection_overlay(
            detection.thumbnail,
            detection.thumbnail_boxes,
            detection_preview_path,
            quality=cfg.preview_quality,
        )
        detection_preview = str(detection_preview_path)

    results: list[CoreBoxResult] = []
    for candidate in detection.candidates:
        crop, inner_bbox, source_crop_bbox = _crop_for_angle_with_bbox(image, candidate.bbox_xyxy_raw, cfg)
        angle = estimate_box_angle(crop, inner_bbox, cfg)
        corrected_image, corrected_mask, corrected_bbox, matrix, _ = correct_core_box(
            image,
            candidate.bbox_xyxy_raw,
            angle.angle_deg,
            cfg,
        )
        image_path = corrected_dir / f"{candidate.box_id}.png"
        mask_path = masks_dir / f"{candidate.box_id}_mask.png"
        review_source_path = review_sources_dir / f"{candidate.box_id}_source.jpg"
        save_rgb_image(image_path, corrected_image, quality=cfg.preview_quality)
        save_rgb_image(mask_path, corrected_mask, quality=cfg.preview_quality)
        save_rgb_image(review_source_path, crop, quality=cfg.preview_quality)

        preview_path: str | None = None
        if cfg.save_previews:
            preview_file = previews_dir / f"{candidate.box_id}_before_after.jpg"
            _save_before_after_preview(crop, corrected_image, preview_file, cfg.preview_quality)
            preview_path = str(preview_file)

        confidence = float(angle.confidence)
        needs_review = confidence < cfg.confidence_threshold
        results.append(
            CoreBoxResult(
                box_id=candidate.box_id,
                order_index=candidate.order_index,
                bbox_xyxy_raw=candidate.bbox_xyxy_raw,
                bbox_xyxy_corrected=corrected_bbox,
                angle_deg=float(angle.angle_deg),
                confidence=confidence,
                needs_manual_review=needs_review,
                rotation_matrix_2x3=matrix,
                output_image=str(image_path),
                output_mask=str(mask_path),
                preview_image=preview_path,
                review_source_image=str(review_source_path),
                source_crop_bbox_raw=source_crop_bbox,
            )
        )

    batch = M11BatchResult(
        module="M1-1",
        input_path=str(input_path),
        hdr_path=str(hdr_path) if hdr_path is not None else None,
        output_dir=str(output_root),
        box_count=len(results),
        boxes=results,
        detection_preview=detection_preview,
    )
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    batch.qa_report = write_qa_report(batch, output_root / "qa_report.md")

    if cfg.expected_box_count is not None and cfg.expected_box_count != batch.box_count:
        # Keep the outputs for diagnosis, but make the mismatch visible to callers.
        raise RuntimeError(
            f"Expected {cfg.expected_box_count} boxes, detected {batch.box_count}. "
            f"Inspect {detection_preview or metadata_path}."
        )

    return batch


def _crop_for_angle(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: M11Config,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    crop, inner, _ = _crop_for_angle_with_bbox(image, bbox, config)
    return crop, inner


def _crop_for_angle_with_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: M11Config,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    px0 = max(0, x0 - config.raw_bbox_margin_px)
    py0 = max(0, y0 - config.raw_bbox_margin_px)
    px1 = min(w - 1, x1 + config.raw_bbox_margin_px)
    py1 = min(h - 1, y1 + config.raw_bbox_margin_px)
    crop = np.asarray(image[py0 : py1 + 1, px0 : px1 + 1])
    inner = (
        max(0, x0 - px0),
        max(0, y0 - py0),
        min(px1 - px0, x1 - px0),
        min(py1 - py0, y1 - py0),
    )
    return crop, inner, (px0, py0, px1, py1)


def _save_before_after_preview(before: np.ndarray, after: np.ndarray, path: Path, quality: int) -> None:
    before_vis = _preview_array(before, max_height=720)
    after_vis = _preview_array(after, max_height=720)
    gap = 16
    height = max(before_vis.height, after_vis.height)
    width = before_vis.width + after_vis.width + gap
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    canvas.paste(before_vis, (0, 0))
    canvas.paste(after_vis, (before_vis.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "before", fill=(255, 240, 0))
    draw.text((before_vis.width + gap + 8, 8), "corrected", fill=(255, 240, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=quality)


def _preview_array(image: np.ndarray, max_height: int) -> Image.Image:
    arr = contrast_stretch_uint8(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    pil = Image.fromarray(arr[:, :, :3], mode="RGB")
    if pil.height > max_height:
        width = max(1, int(round(pil.width * max_height / pil.height)))
        pil = pil.resize((width, max_height), Image.Resampling.BILINEAR)
    return pil
