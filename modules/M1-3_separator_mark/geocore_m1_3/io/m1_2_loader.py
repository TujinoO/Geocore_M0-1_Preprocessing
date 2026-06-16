from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geocore_m1_3.utils.json_io import read_json


@dataclass
class M12Inputs:
    mask_path: Path
    metadata_path: Path | None
    contours_path: Path | None
    probability_path: Path | None
    overlay_path: Path | None
    metadata: dict


def load_m1_2_inputs(m1_2_output_dir: str | Path, mask_path: str | Path | None = None) -> M12Inputs:
    root = Path(m1_2_output_dir)
    resolved_mask = Path(mask_path) if mask_path else root / "mask.png"
    if not resolved_mask.exists():
        fallback = root / "mask.tif"
        if fallback.exists():
            resolved_mask = fallback
    if not resolved_mask.exists():
        raise FileNotFoundError(f"M1-2 mask file was not found: {resolved_mask}")

    metadata_path = root / "metadata.json"
    contours_path = root / "contours.json"
    probability_path = root / "probability.tif"
    overlay_path = root / "overlay.png"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    return M12Inputs(
        mask_path=resolved_mask,
        metadata_path=metadata_path if metadata_path.exists() else None,
        contours_path=contours_path if contours_path.exists() else None,
        probability_path=probability_path if probability_path.exists() else None,
        overlay_path=overlay_path if overlay_path.exists() else None,
        metadata=metadata,
    )
