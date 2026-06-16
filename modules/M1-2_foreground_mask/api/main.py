from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from geocore_mask.models.registry import available_models

from .schemas import (
    ForegroundMaskRequest,
    ImageDataType,
    JobStatus,
    MaskEditRequest,
    MaskEditResponse,
    SystemStatus,
    TaskCreateResponse,
    TaskInfo,
)
from .service import (
    TASKS,
    UPLOAD_DIR,
    create_task,
    ensure_runtime_dirs,
    apply_mask_edit,
    parse_csv_floats,
    run_foreground_mask,
    update_task,
)


app = FastAPI(
    title="Geo-Core AI M1-2 Foreground Mask API",
    description="Rock core foreground mask extraction service for Geo-Core AI.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_runtime_dirs()


@app.get("/api/system/status", response_model=SystemStatus, tags=["system"])
def system_status() -> SystemStatus:
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    memory = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        memory = {"total_gb": round(vm.total / 1024**3, 2), "available_gb": round(vm.available / 1024**3, 2)}
    except Exception:
        memory = None
    return SystemStatus(
        service="Geo-Core AI foreground mask API",
        status="ok",
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
        gpu_names=gpu_names,
        memory=memory,
    )


@app.get("/api/m1/foreground-mask/models", tags=["foreground-mask"])
def list_models():
    return {"models": available_models(), "default_profile": "default"}


@app.post(
    "/api/m1/foreground-mask/jobs",
    response_model=TaskCreateResponse,
    status_code=202,
    tags=["foreground-mask"],
)
def create_foreground_mask_job(request: ForegroundMaskRequest, background_tasks: BackgroundTasks):
    task = create_task()
    background_tasks.add_task(run_foreground_mask, task.task_id, request)
    return TaskCreateResponse(
        task_id=task.task_id,
        status=JobStatus.pending,
        status_url=f"/api/m1/foreground-mask/jobs/{task.task_id}",
    )


@app.post(
    "/api/preprocessing/foreground-mask",
    response_model=TaskCreateResponse,
    status_code=202,
    tags=["preprocessing"],
)
def create_preprocessing_foreground_mask_job(request: ForegroundMaskRequest, background_tasks: BackgroundTasks):
    return create_foreground_mask_job(request, background_tasks)


@app.post(
    "/api/m1/foreground-mask/upload",
    response_model=TaskCreateResponse,
    status_code=202,
    tags=["foreground-mask"],
)
async def upload_foreground_mask_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_path: Optional[str] = Form(None),
    model_profile: str = Form("default"),
    model_package: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
    enable_postprocess: Optional[bool] = Form(None),
    output_preview: bool = Form(True),
    model_name: str = Form("UNet"),
    num_classes: int = Form(2),
    band_num: int = Form(3),
    mean: Optional[str] = Form(None),
    std: Optional[str] = Form(None),
    target_size: int = Form(512),
    overlap_rate: float = Form(0.25),
    img_data_type: ImageDataType = Form(ImageDataType.byte),
):
    task = create_task()
    task_upload_dir = UPLOAD_DIR / task.task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = task_upload_dir / Path(file.filename).name

    with input_path.open("wb") as output:
        while chunk := await file.read(1024 * 1024):
            output.write(chunk)

    request = ForegroundMaskRequest(
        input_path=str(input_path),
        model_profile=model_profile,
        model_package=model_package,
        threshold=threshold,
        enable_postprocess=enable_postprocess,
        output_preview=output_preview,
        model_path=model_path,
        model_name=model_name,
        num_classes=num_classes,
        band_num=band_num,
        mean=parse_csv_floats(mean),
        std=parse_csv_floats(std),
        target_size=target_size,
        overlap_rate=overlap_rate,
        img_data_type=img_data_type,
    )
    update_task(task.task_id, input_paths=[str(input_path)])
    background_tasks.add_task(run_foreground_mask, task.task_id, request)
    return TaskCreateResponse(
        task_id=task.task_id,
        status=JobStatus.pending,
        status_url=f"/api/m1/foreground-mask/jobs/{task.task_id}",
    )


@app.get("/api/m1/foreground-mask/jobs/{task_id}", response_model=TaskInfo, tags=["foreground-mask"])
def get_foreground_mask_job(task_id: str) -> TaskInfo:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task_id not found") from exc


@app.get("/api/preprocessing/status/{task_id}", response_model=TaskInfo, tags=["preprocessing"])
def get_preprocessing_status(task_id: str) -> TaskInfo:
    return get_foreground_mask_job(task_id)


@app.post(
    "/api/m1/foreground-mask/jobs/{task_id}/mask-edits",
    response_model=MaskEditResponse,
    tags=["foreground-mask"],
)
def edit_foreground_mask(task_id: str, request: MaskEditRequest) -> MaskEditResponse:
    try:
        return MaskEditResponse(**apply_mask_edit(task_id, request))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task_id not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/api/preprocessing/foreground-mask/{task_id}/mask-edits",
    response_model=MaskEditResponse,
    tags=["preprocessing"],
)
def edit_preprocessing_foreground_mask(task_id: str, request: MaskEditRequest) -> MaskEditResponse:
    return edit_foreground_mask(task_id, request)


@app.get("/api/m1/foreground-mask/jobs/{task_id}/download", tags=["foreground-mask"])
def download_first_result(task_id: str):
    task = get_foreground_mask_job(task_id)
    if task.status != JobStatus.succeeded or not task.output_files:
        raise HTTPException(status_code=409, detail="task has no downloadable result yet")
    result_path = task.output_files.get("mask_tif") or next(iter(task.output_files.values()))
    return FileResponse(result_path, filename=Path(result_path).name)
