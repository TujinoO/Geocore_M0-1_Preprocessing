from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from geocore_mask.inference.merger import ProbabilityMerger
from geocore_mask.inference.tiler import generate_tiles
from geocore_mask.models.registry import build_model
from geocore_mask.postprocess.contour import mask_to_components
from geocore_mask.postprocess.refine_mask import refine_mask
from geocore_mask.utils.image_io import read_image, save_mask_png, save_overlay_png, save_tiff_like
from geocore_mask.utils.model_package import ModelPackage


class CoreMaskPredictor:
    def __init__(self, model_package: ModelPackage, device: Optional[str] = None):
        self.model_package = model_package
        self.manifest = model_package.manifest
        self.device = self._resolve_device(device or self.manifest.get("inference", {}).get("device", "auto"))
        self.model = self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self):
        input_cfg = self.manifest.get("input", {})
        input_channels = int(input_cfg.get("channels", 3))
        num_classes = int(self.manifest.get("num_classes", 2))
        model = build_model(self.manifest.get("model_name", "UNet"), input_channels=input_channels, num_classes=num_classes)

        weights_path = self.model_package.weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"model weights not found: {weights_path}")

        state = torch.load(str(weights_path), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state)
        except RuntimeError:
            cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(cleaned)
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        *,
        threshold: Optional[float] = None,
        enable_postprocess: Optional[bool] = None,
        output_preview: bool = True,
    ) -> Dict[str, Any]:
        started = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_data = read_image(input_path)
        image = image_data.array
        prob = self.predict_probability(image)

        inference_cfg = self.manifest.get("inference", {})
        post_cfg = self.manifest.get("postprocess", {})
        threshold_value = float(threshold if threshold is not None else inference_cfg.get("threshold", 0.5))
        mask = (prob >= threshold_value).astype(np.uint8)

        warnings: list[str] = []
        post_enabled = bool(post_cfg.get("enable", True) if enable_postprocess is None else enable_postprocess)
        if post_enabled:
            mask, warnings = refine_mask(mask, post_cfg)

        components = mask_to_components(mask, int(post_cfg.get("max_contour_points", 800)))

        mask_png = output_path / "mask.png"
        mask_tif = output_path / "mask.tif"
        probability_tif = output_path / "probability.tif"
        overlay_png = output_path / "overlay.png"
        contours_json = output_path / "contours.json"
        metadata_json = output_path / "metadata.json"

        save_mask_png(mask, mask_png)
        save_tiff_like(mask * 255, mask_tif, reference=image_data, dtype=np.uint8)
        save_tiff_like(np.clip(prob * 255, 0, 255), probability_tif, reference=image_data, dtype=np.uint8)
        if output_preview:
            save_overlay_png(image, mask, overlay_png)

        with contours_json.open("w", encoding="utf-8") as f:
            json.dump(components, f, ensure_ascii=False, indent=2)

        metadata = {
            "module": "M1-2 foreground mask",
            "model_name": self.manifest.get("model_name", "unknown"),
            "model_version": self.manifest.get("model_version", "unknown"),
            "input_path": str(input_path),
            "threshold": threshold_value,
            "postprocess": post_enabled,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "foreground_area_ratio": float(mask.mean()) if mask.size else 0.0,
            "component_count": int(components["component_count"]),
            "warnings": warnings,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with metadata_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        output_files = {
            "mask_png": str(mask_png),
            "mask_tif": str(mask_tif),
            "probability_tif": str(probability_tif),
            "contours_json": str(contours_json),
            "metadata_json": str(metadata_json),
        }
        if output_preview:
            output_files["overlay_png"] = str(overlay_png)

        return {
            "output_files": output_files,
            "metrics": {
                "foreground_area_ratio": metadata["foreground_area_ratio"],
                "component_count": metadata["component_count"],
                "elapsed_seconds": metadata["elapsed_seconds"],
            },
            "warnings": warnings,
        }

    @torch.no_grad()
    def predict_probability(self, image: np.ndarray) -> np.ndarray:
        image = self._prepare_image(image)
        h, w, _ = image.shape
        infer_cfg = self.manifest.get("inference", {})
        tile_size = int(infer_cfg.get("tile_size", self.manifest.get("input", {}).get("image_size", 512)))
        overlap = float(infer_cfg.get("overlap", 0.25))

        padded_h = max(h, tile_size)
        padded_w = max(w, tile_size)
        pad_h = padded_h - h
        pad_w = padded_w - w
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        merger = ProbabilityMerger(padded_h, padded_w, tile_size)
        for tile in generate_tiles(padded_w, padded_h, tile_size, overlap):
            patch = padded[tile.y : tile.y + tile.size, tile.x : tile.x + tile.size, :]
            probability = self._predict_patch(patch)
            merger.add(probability, tile.x, tile.y)

        return np.clip(merger.result()[:h, :w], 0, 1)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[2] == 4:
            image = image[:, :, :3]

        input_cfg = self.manifest.get("input", {})
        channels = int(input_cfg.get("channels", 3))
        if image.shape[2] < channels:
            repeats = channels - image.shape[2]
            image = np.concatenate([image] + [image[:, :, -1:]] * repeats, axis=2)
        image = image[:, :, :channels].astype(np.float32)

        mean = np.array(input_cfg.get("mean", [0.0] * channels), dtype=np.float32)[:channels]
        std = np.array(input_cfg.get("std", [1.0] * channels), dtype=np.float32)[:channels]
        std[std == 0] = 1.0
        return (image - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    def _predict_patch(self, patch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(patch.transpose(2, 0, 1)[None, ...]).float().to(self.device)
        logits = self.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits[:, 0])
        else:
            prob = torch.softmax(logits, dim=1)[:, 1]
        return prob.squeeze(0).detach().cpu().numpy().astype(np.float32)
