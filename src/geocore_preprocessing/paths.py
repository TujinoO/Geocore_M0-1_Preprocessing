from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODULE_CONFIG = WORKSPACE_ROOT / "configs" / "module_paths.json"


def load_module_config() -> dict[str, Any]:
    return json.loads(MODULE_CONFIG.read_text(encoding="utf-8"))


def module_path(module_id: str) -> Path:
    config = load_module_config()
    module = config["modules"][module_id]
    return Path(config["workspace_root"]) / module["path"]


def module_python_path(module_id: str) -> Path:
    config = load_module_config()
    module = config["modules"][module_id]
    return Path(config["workspace_root"]) / module["python_path"]


def bootstrap_module_paths() -> list[Path]:
    """Add migrated module package roots to sys.path for in-process orchestration."""

    config = load_module_config()
    root = Path(config["workspace_root"])
    paths = [WORKSPACE_ROOT / "src"]
    for module in config["modules"].values():
        paths.append(root / module["python_path"])

    added: list[Path] = []
    for path in reversed(paths):
        resolved = path.resolve()
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
            added.append(resolved)
    return list(reversed(added))


def find_m12_model_package(explicit: str | None = None, profile: str = "core_mask_unet_v1") -> Path | None:
    """Resolve a foreground-mask model package.

    The code/docs were migrated into this workspace, while large model weights
    may remain in the original module directory. This resolver checks both
    places and still allows callers to provide an explicit package path.
    """

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            module_path("M1-2") / "models" / profile,
            Path("D:/Code/Geocore_M1-2_foreground_mask/models") / profile,
            module_path("M1-2") / "models" / "core_mask_unet_v2",
            Path("D:/Code/Geocore_M1-2_foreground_mask/models/core_mask_unet_v2"),
        ]
    )
    for candidate in candidates:
        if (candidate / "model_manifest.json").is_file():
            return candidate
    return None
