# M1-3 API Change Requests

日期：2026-06-16  
模块：`geocore_m1_3`  
契约文件：`docs/openapi/m1-3.openapi.json`

## 目标

为前端 M1-3 页面提供稳定的前后端分离契约：页面先基于 OpenAPI 生成类型和 mock，再进入接口联调。本文记录需要后端确认或补齐的实现事项，避免前端开发直接修改后端算法代码。

## 当前已存在接口

- `GET /api/system/status`
- `POST /api/core-boxes/depth-metadata`
- `GET /api/core-boxes/{core_box_id}/depth-metadata`
- `POST /api/preprocessing/segment-depth`
- `GET /api/preprocessing/segment-depth/{task_id}`

## 需要后端确认的字段约束

- `depth_end_m` 必须大于 `depth_start_m`，否则返回 `400`。
- `segmentation.overlap_cm` 必须小于 `segmentation.segment_length_cm`，否则返回 `400`。
- `layout.lane_order` 仅允许 `left_to_right`、`right_to_left`。
- `layout.lane_direction` 仅允许 `top_to_bottom`、`bottom_to_top`。
- `gap_policy` 建议仅允许 `close_artificial_gaps`、`preserve_visual_gaps` 或 `null`。
- `missing_intervals` 必须落在 `[depth_start_m, depth_end_m]` 范围内，且每段 `depth_end_m > depth_start_m`。
- `depth_anchors` 的 `strip_y` 必须为非负数；如果后端启用锚点映射，建议至少包含首尾锚点。

## 权限行为

建议后端按 OpenAPI 的 `x-permissions` 实现权限检查：

| 接口 | 权限 |
| --- | --- |
| `POST /api/core-boxes/depth-metadata` | `m1_3.depth_metadata:write` |
| `GET /api/core-boxes/{core_box_id}/depth-metadata` | `m1_3.depth_metadata:read` |
| `POST /api/preprocessing/segment-depth` | `m1_3.task:run` |
| `GET /api/preprocessing/segment-depth/{task_id}` | `m1_3.task:read` |
| `GET /api/preprocessing/segment-depth/{task_id}/segments` | `m1_3.segment:read` |
| `GET /api/preprocessing/segment-depth/{task_id}/preview` | `m1_3.preview:read` |
| `PUT /api/preprocessing/segment-depth/{task_id}/corrections` | `m1_3.review:write` |
| `POST /api/preprocessing/segment-depth/{task_id}/rerun` | `m1_3.task:run` |
| `GET /api/files` | `m1_3.file:read` |

权限失败约定：

- 未登录或 token 无效：`401`，返回 `{ "detail": "Authentication credentials were not provided." }`。
- 已登录但缺权限：`403`，返回 `{ "detail": "Missing permission: <permission>" }`。
- 用户无权访问某个任务或文件时，后端可返回 `403`；如需隐藏资源存在性，也可返回 `404`，但需在全局 API 规范中统一。

## 需要新增或调整的接口

### 1. 图斑分页接口

`GET /api/preprocessing/segment-depth/{task_id}/segments`

用途：前端结果页图斑表格/网格分页加载，避免一次性读取完整 `segments.json`。

实现建议：

- 读取对应任务输出目录下的 `segments.json`。
- 支持 `page`、`page_size`、`quality_flag` 查询参数。
- 返回 `SegmentPage`，其中每个 `CoreSegment` 建议补充 `image_url` 和 `mask_url`。

### 2. 预览聚合接口

`GET /api/preprocessing/segment-depth/{task_id}/preview`

用途：前端结果页一次获取预览图 URL、质量报告和深度刻度数据。

实现建议：

- 读取 `review_manifest.json`、`quality_report.json`、`depth_mapping.json`。
- 将本地文件路径转换为可被浏览器访问的 URL。
- 返回 `needs_review`、`image_urls`、`quality_report`、`depth_mapping`、`warnings`。

### 3. 人工复核修正保存接口

`PUT /api/preprocessing/segment-depth/{task_id}/corrections`

用途：保存前端人工复核时调整的列方向、切分参数、缺失段、深度锚点和备注。

实现建议：

- 只保存 corrections，不直接重跑。
- 建议落库或写入任务目录下的 `corrections.json`。
- 返回 `{ "task_id": "...", "status": "saved", "updated_at": "..." }`。

### 4. 基于修正重跑接口

`POST /api/preprocessing/segment-depth/{task_id}/rerun`

用途：使用原任务输入和 corrections 创建新任务，支持前端从复核页一键重跑。

实现建议：

- 后端读取原始 `input_manifest.json` 和 corrections。
- 创建新的 `task_id`，不要覆盖原任务产物。
- 返回 `TaskResponse`，前端继续轮询新任务。

### 5. 产物文件访问接口

`GET /api/files?path=...&disposition=inline`

用途：让浏览器访问 M1-3 输出图片和 JSON 产物。

安全要求：

- 严禁允许任意系统路径读取。
- `path` 必须限制在当前用户有权限访问的 M1-3 任务输出根目录下。
- 建议支持产物 key 或相对路径，而不是直接暴露本机绝对路径。

## Mock 示例

已补充前端可直接使用的 mock 文件：

- `docs/mock/m1-3/system-status.ok.json`
- `docs/mock/m1-3/depth-metadata.saved.json`
- `docs/mock/m1-3/segment-depth.request.json`
- `docs/mock/m1-3/task.running.json`
- `docs/mock/m1-3/task.succeeded.json`
- `docs/mock/m1-3/segments.page.json`
- `docs/mock/m1-3/preview.response.json`
- `docs/mock/m1-3/corrections.request.json`

## 前端接入建议

- 前端优先使用 `docs/openapi/m1-3.openapi.json` 生成类型。
- 当前后端已有的 5 个接口可先联调最小闭环。
- 页面结果展示优先依赖 `preview`、`segments`、`files` 三类接口；如果后端暂未实现，前端可先用 mock。
- 不要在前端长期依赖本地绝对路径作为图片地址；正式部署必须使用后端返回的 URL。
