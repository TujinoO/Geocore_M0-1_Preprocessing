from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ImageDataType(str, Enum):
    byte = "byte"
    uint16 = "uint16"


class ForegroundMaskRequest(BaseModel):
    """Request body for M1-2 rock core foreground mask extraction.

    The first group of fields is the release-facing API. The legacy fields are
    kept so older front-end demos and test JSON files can still run while the
    module migrates away from remote-sensing style parameters.
    """

    input_path: str = Field(..., description="Input image file or directory that the backend can access.")
    output_dir: Optional[str] = Field(default=None, description="Output directory. Defaults to api_runtime/outputs/{task_id}.")
    image_pattern: str = Field(default="*.tif", description="File pattern used when input_path is a directory.")
    recursive: bool = Field(default=False, description="Whether to recursively scan input_path when it is a directory.")

    model_profile: str = Field(default="default", description="Packaged model profile under models/.")
    model_package: Optional[str] = Field(default=None, description="Explicit model package directory containing model_manifest.json.")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Foreground probability threshold.")
    enable_postprocess: Optional[bool] = Field(default=None, description="Override packaged post-processing switch.")
    output_preview: bool = Field(default=True, description="Whether to output overlay.png for preview.")

    # Legacy compatibility fields.
    model_name: Optional[str] = Field(default=None, description="Legacy model name, e.g. UNet.")
    model_path: Optional[str] = Field(default=None, description="Legacy direct PyTorch weight path.")
    num_classes: int = Field(default=2, ge=1)
    band_num: int = Field(default=3, ge=1)
    label_norm: bool = Field(default=True)
    train_list_path: Optional[str] = None
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    target_size: int = Field(default=512, ge=16)
    unify_read_img: bool = Field(default=True)
    overlap_rate: float = Field(default=0.25, ge=0.0, lt=0.8)
    ignore_bandnum: int = Field(default=0, ge=0)
    img_data_type: ImageDataType = Field(default=ImageDataType.byte)
    use_mask: bool = Field(default=False)
    mask_path: Optional[str] = None
    use_vsimem: bool = Field(default=False)

    @validator("std")
    def std_must_match_mean(cls, value, values):
        mean = values.get("mean")
        if value is not None and mean is not None and len(value) != len(mean):
            raise ValueError("std and mean must have the same length")
        return value


class MaskEditRegion(BaseModel):
    """A frontend-selected area used to edit the final binary mask.

    Coordinates are image pixel coordinates by default. If the frontend sends
    coordinates from a scaled preview canvas, also send display_width and
    display_height in MaskEditRequest so the backend can map them back.
    """

    type: Literal["rectangle", "polygon", "brush"]
    bbox: Optional[List[float]] = Field(default=None, description="Rectangle as [x1, y1, x2, y2].")
    points: Optional[List[List[float]]] = Field(default=None, description="Polygon vertices or brush path points.")
    radius: Optional[float] = Field(default=None, gt=0, description="Brush radius in pixels.")


class MaskEditRequest(BaseModel):
    operation: Literal["remove"] = Field(default="remove", description="Remove selected regions from the foreground mask.")
    regions: List[MaskEditRegion] = Field(..., min_items=1)
    display_width: Optional[float] = Field(default=None, gt=0, description="Frontend preview canvas width.")
    display_height: Optional[float] = Field(default=None, gt=0, description="Frontend preview canvas height.")
    result_key_prefix: str = Field(
        default="",
        description="Optional prefix for batch outputs, e.g. 'sample.' when editing sample.mask_tif.",
    )
    output_preview: bool = Field(default=True, description="Whether to refresh overlay.png after editing.")


class MaskEditResponse(BaseModel):
    task_id: str
    status: JobStatus
    message: str
    output_files: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    edited_at: str


class TaskCreateResponse(BaseModel):
    task_id: str
    status: JobStatus
    status_url: str


class TaskInfo(BaseModel):
    task_id: str
    status: JobStatus
    message: str = ""
    input_paths: List[str] = Field(default_factory=list)
    output_dir: Optional[str] = None
    output_files: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: str
    updated_at: str


class SystemStatus(BaseModel):
    service: Literal["Geo-Core AI foreground mask API"]
    status: Literal["ok"]
    cuda_available: bool
    cuda_device_count: int
    gpu_names: List[str]
    memory: Optional[Dict[str, float]] = None
    default_model_profile: str = "default"
