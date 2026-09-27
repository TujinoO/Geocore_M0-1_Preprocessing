from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MODULE_CONFIG = WORKSPACE_ROOT / "configs" / "module_paths.json"


def load_module_config() -> dict[str, Any]:
    return json.loads(MODULE_CONFIG.read_text(encoding="utf-8"))


def workspace_root(config: dict[str, Any] | None = None) -> Path:
    """Resolve the consolidated workspace independently of its drive or checkout name."""

    config = config or load_module_config()
    configured = Path(str(config.get("workspace_root", ".")))
    if not configured.is_absolute():
        configured = WORKSPACE_ROOT / configured
    return configured.resolve()


def module_path(module_id: str) -> Path:
    config = load_module_config()
    module = config["modules"][module_id]
    return workspace_root(config) / module["path"]


def module_python_path(module_id: str) -> Path:
    config = load_module_config()
    module = config["modules"][module_id]
    return workspace_root(config) / module["python_path"]


def bootstrap_module_paths() -> list[Path]:
    """Add migrated module package roots to sys.path for in-process orchestration."""

    config = load_module_config()
    root = workspace_root(config)
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


def find_m12_model_package(explicit: str | None = None, profile: str = "core_mask_unet_v4") -> Path | None:
    """Resolve a foreground-mask model package.

    Model weights, cards, manifests, and training assets are consolidated under
    the M1-2 module. Callers may still provide an explicit package path.
    """

    candidate = Path(explicit) if explicit else module_path("M1-2") / "models" / profile
    return candidate.resolve() if (candidate / "model_manifest.json").is_file() else None
