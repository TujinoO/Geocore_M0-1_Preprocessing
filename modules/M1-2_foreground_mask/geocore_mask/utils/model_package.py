from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelPackage:
    root: Path
    manifest: Dict[str, Any]

    @property
    def model_name(self) -> str:
        return str(self.manifest.get("model_name", "UNet"))

    @property
    def model_version(self) -> str:
        return str(self.manifest.get("model_version", "unknown"))

    @property
    def weights_path(self) -> Path:
        weights = Path(str(self.manifest["weights"]))
        return weights if weights.is_absolute() else self.root / weights


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_package(model_profile: str = "default", model_package: Optional[str] = None) -> ModelPackage:
    if model_package:
        root = Path(model_package)
    else:
        root = DEFAULT_MODELS_DIR / ("core_mask_unet_v4" if model_profile == "default" else model_profile)

    manifest_path = root / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"model manifest not found: {manifest_path}. "
            "Provide model_package or install the matching package under M1-2_foreground_mask/models."
        )

    manifest = _load_json(manifest_path)
    if "weights" not in manifest:
        raise ValueError(f"model manifest is missing required field 'weights': {manifest_path}")
    package = ModelPackage(root=root, manifest=manifest)
    if not package.weights_path.is_file():
        raise FileNotFoundError(f"model weights not found: {package.weights_path}")
    expected = manifest.get("weights_sha256")
    if expected:
        sha256 = hashlib.sha256()
        with package.weights_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(block)
        digest = sha256.hexdigest()
        if digest.lower() != str(expected).lower():
            raise ValueError(f"model weights SHA-256 mismatch: {package.weights_path}")
    return package


def build_ad_hoc_manifest(
    *,
    model_name: str,
    model_path: str,
    input_channels: int,
    num_classes: int,
    mean: Optional[list[float]],
    std: Optional[list[float]],
    tile_size: int,
    overlap: float,
    threshold: float,
) -> ModelPackage:
    root = Path(model_path).resolve().parent
    manifest = {
        "model_name": model_name,
        "model_version": "ad-hoc",
        "task": "M1-2 foreground mask",
        "weights": str(Path(model_path).resolve()),
        "input": {
            "channels": input_channels,
            "image_size": tile_size,
            "mean": mean or [0.0] * input_channels,
            "std": std or [1.0] * input_channels,
        },
        "inference": {
            "threshold": threshold,
            "tile_size": tile_size,
            "overlap": overlap,
            "device": "auto",
        },
        "postprocess": {
            "enable": True,
            "min_component_area": 100,
            "fill_hole_area": 500,
            "smooth_kernel_size": 3,
            "keep_top_k": None,
        },
    }
    return ModelPackage(root=root, manifest=manifest)
