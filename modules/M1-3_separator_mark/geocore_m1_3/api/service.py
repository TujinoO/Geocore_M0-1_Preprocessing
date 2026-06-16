from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from geocore_m1_3.pipeline import run_segment_depth


TASKS: dict[str, dict[str, Any]] = {}
DEPTH_METADATA: dict[str, dict[str, Any]] = {}


def save_depth_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    core_box_id = payload["core_box_id"]
    record = dict(payload)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    DEPTH_METADATA[core_box_id] = record
    return {"core_box_id": core_box_id, "status": "saved", **record}


def get_depth_metadata(core_box_id: str) -> dict[str, Any] | None:
    return DEPTH_METADATA.get(core_box_id)


def submit_segment_depth(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = payload.get("task_id")
    try:
        result = run_segment_depth(payload)
        TASKS[result["task_id"]] = result
        return result
    except Exception as exc:
        failed_id = task_id or "failed"
        result = {"task_id": failed_id, "status": "failed", "error_message": str(exc)}
        TASKS[failed_id] = result
        raise


def get_task(task_id: str) -> dict[str, Any] | None:
    return TASKS.get(task_id)
