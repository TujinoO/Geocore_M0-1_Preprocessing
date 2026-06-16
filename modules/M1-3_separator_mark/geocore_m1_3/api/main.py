from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - depends on deployment environment
    FastAPI = None
    HTTPException = None

from geocore_m1_3 import __version__
from geocore_m1_3.api.schemas import DepthMetadataRequest, SegmentDepthRequest, TaskResponse
from geocore_m1_3.api.service import get_depth_metadata, get_task, save_depth_metadata, submit_segment_depth


def _dump_model(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def create_app():
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install the optional 'api' dependencies to run the HTTP service.")
    app = FastAPI(title="Geo-Core AI M1-3 API", version=__version__)

    @app.get("/api/system/status")
    def system_status():
        return {
            "service": "geocore-m1-3",
            "module": "M1-3 core segmentation and depth marking",
            "version": __version__,
            "status": "ok",
        }

    @app.post("/api/core-boxes/depth-metadata")
    def post_depth_metadata(request: DepthMetadataRequest):
        return save_depth_metadata(_dump_model(request))

    @app.get("/api/core-boxes/{core_box_id}/depth-metadata")
    def get_depth_metadata_endpoint(core_box_id: str):
        record = get_depth_metadata(core_box_id)
        if not record:
            raise HTTPException(status_code=404, detail="Depth metadata was not found.")
        return record

    @app.post("/api/preprocessing/segment-depth", response_model=TaskResponse)
    def post_segment_depth(request: SegmentDepthRequest):
        payload = _dump_model(request)
        try:
            result = submit_segment_depth(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return TaskResponse(**result)

    @app.get("/api/preprocessing/segment-depth/{task_id}", response_model=TaskResponse)
    def get_segment_depth(task_id: str):
        result = get_task(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Task was not found.")
        return TaskResponse(**result)

    return app


app = create_app() if FastAPI is not None else None
