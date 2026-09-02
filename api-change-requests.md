# API Change Requests - M0/M1 Preprocessing

本文档记录给后端实现人员的接口实现说明。前端新增页面或字段前，应先更新：

1. `docs/openapi/preprocessing.openapi.json`
2. `docs/mocks/preprocessing/*.json`
3. 本文件

然后再基于 OpenAPI 生成类型并开发页面。

## 2026-06-16 - 建立统一预处理 API 契约

### 背景

四个算法模块已经汇总到：

```text
E:\Code\Geocore_M0&1_Preprocessing
```

并通过 `geocore_preprocessing.pipeline.run_preprocessing_pipeline` 打通：

```text
M0-1 manifest/Zarr
  -> M1-1 rotation correction
  -> M1-2 foreground mask
  -> M1-3 separator/depth marking
  -> preprocess_context.json + segments_with_cube_refs.json
```

现在需要后端提供面向前端的统一 HTTP API，不要求前端直接运行 CLI 或读取服务器本地路径。

### OpenAPI 文件

接口契约已定义在：

```text
docs/openapi/preprocessing.openapi.json
```

后端实现应以该文件为准。

### Mock 文件

前端联调用 mock 已补充在：

```text
docs/mocks/preprocessing/
```

包括：

- `system-status.ok.json`
- `modules.response.json`
- `permissions.response.json`
- `manifest-inspect.request.json`
- `manifest-inspect.response.json`
- `create-job.request.json`
- `create-job.response.json`
- `task.running.json`
- `task.succeeded.json`
- `task.failed.json`
- `context.response.json`
- `boxes.response.json`
- `segments.page.json`
- `review-action.request.json`
- `review-action.response.json`
- `error.validation.json`

## 一、需要后端实现的接口

### 1. 系统状态

```http
GET /api/preprocessing/system/status
```

用途：

- 前端初始化预处理页面。
- 展示 M0/M1 模块可用状态。
- 展示 GPU、内存、模型包状态。

实现要求：

- 普通用户可看 `status`、`modules.available`、`modules.health`。
- 只有 `preprocessing:admin` 可看完整服务器路径。
- 如果 M1-2 模型权重缺失，`M1-2.health` 应返回 `warning`，不要直接导致系统不可用。

### 2. 模块注册表

```http
GET /api/preprocessing/modules
```

用途：

- 前端展示当前预处理工程包含哪些模块。
- 调试模块路径和版本。

实现要求：

- 返回 M0-1、M1-1、M1-2、M1-3 四个模块。
- `available=false` 时必须给出 `warnings`。

### 3. 当前用户权限

```http
GET /api/preprocessing/permissions
```

用途：

- 前端控制按钮可见性和禁用原因。

权限枚举：

```text
preprocessing:read
preprocessing:create
preprocessing:cancel
preprocessing:review
preprocessing:export
preprocessing:admin
```

实现要求：

- 不要只返回字符串数组，应返回 `allowed` 和 `denied_reason`。
- 前端会用 `denied_reason` 做按钮 tooltip 或禁用说明。

### 4. Manifest 预检查

```http
POST /api/preprocessing/manifest/inspect
```

用途：

- 新建任务前检查 M0 manifest 是否可用。
- 解析 preview、Zarr、band metadata、registration model。

后端实现建议：

- 调用 `FusionResultAccessor` 或等价逻辑。
- 检查文件是否存在。
- 返回 `preview_to_reference_scale`，用于前端提示预览图是否为缩略图。

### 5. 创建预处理任务

```http
POST /api/preprocessing/jobs
```

用途：

- 创建完整 M0/M1 预处理异步任务。

后端实现建议：

- 请求体映射到 `run_preprocessing_pipeline(payload)`。
- 必须异步执行，不要阻塞 HTTP 请求。
- 返回 `202` 和 `task_id`。
- 支持从已有 `m0.manifest_path` 启动。
- 后续可支持从 RGB/NIR/SWIR 原始输入启动 M0 融合。

校验要求：

- `depth.depth_end_m > depth.depth_start_m`。
- `m1_2.engine=model` 时必须能解析模型包，否则返回 400。
- `m1_1.input_image` 为 `.dat` 时必须提供 `m1_1.hdr_path`。
- `output_dir` 必须在后端允许的工作目录内。

### 6. 查询任务列表

```http
GET /api/preprocessing/jobs
```

用途：

- 任务列表页。

实现要求：

- 支持 `status`、`project_id`、`hole_id` 过滤。
- 支持分页。
- 不要返回过大的 `preprocess_context`，只返回摘要。

### 7. 查询任务状态

```http
GET /api/preprocessing/jobs/{task_id}
```

用途：

- 任务详情页轮询。

实现要求：

- 返回 `status`、`stage`、`progress`、`message`。
- 返回 `metrics.box_count`、`metrics.segment_count`。
- 失败时返回标准 `ApiError`。
- `warnings` 使用结构化对象，不要只返回字符串。

### 8. 取消任务

```http
POST /api/preprocessing/jobs/{task_id}/cancel
```

用途：

- 取消 queued/running 任务。

权限：

- 需要 `preprocessing:cancel`。

实现要求：

- 对已完成任务返回 409。
- 对运行中任务尽量设置取消标记，允许算法阶段在安全点停止。

### 9. 查询预处理上下文

```http
GET /api/preprocessing/jobs/{task_id}/context
```

用途：

- 任务详情页。
- 前端读取跨模块产物引用。

实现要求：

- 返回 `preprocess_context.json` 的结构化内容。
- 本地绝对路径可保留在 JSON 中，但前端预览文件必须通过 artifact 接口。

### 10. 查询箱体列表

```http
GET /api/preprocessing/jobs/{task_id}/boxes
```

用途：

- M1-1 复核页。
- M1-2 掩膜复核页。

实现要求：

- 返回 `CoreBoxSummary[]`。
- 每个 box 需要包含 `corrected_image`、`m1_1_mask`、`m1_2.output_files`、`m1_3.metrics`。

### 11. 查询图斑列表

```http
GET /api/preprocessing/jobs/{task_id}/segments
```

用途：

- M1-3 深度图斑页。
- 后续 M2 模块选择图斑。

实现要求：

- 从每个 `segments_with_cube_refs.json` 汇总分页返回。
- 支持 `core_box_id`、`depth_from_m`、`depth_to_m` 过滤。
- 保留完整 `cube_ref`。

### 12. 产物预览/下载

```http
GET /api/preprocessing/jobs/{task_id}/artifacts/{artifact_id}
```

用途：

- 预览 PNG/JPG。
- 下载 JSON/CSV/report。

实现要求：

- 前端不直接访问服务器本地路径。
- 后端用 `artifact_id` 查表映射真实文件。
- `inline` 用于图片预览，`attachment` 用于下载。
- Zarr 大目录默认不允许直接下载，必要时后续新增导出任务。

### 13. 提交人工复核动作

```http
POST /api/preprocessing/jobs/{task_id}/review-actions
```

用途：

- M1-1 box 复核。
- M1-2 mask 编辑。
- M1-3 深度锚点修正。

权限：

- 需要 `preprocessing:review`。

实现要求：

- 先保存 review action，不一定立即重跑算法。
- 返回 `review_id`。
- 对会触发重算的动作，后端可创建子任务或将主任务状态置为 `needs_review`。

## 二、后端任务状态机

建议状态：

```text
queued
running
succeeded
failed
cancelled
needs_review
```

建议阶段：

```text
initializing
m0_fusion
m1_1_rotation
m1_2_foreground_mask
m1_3_separator_mark
finalizing
done
```

前端依赖这些枚举展示进度条，请不要随意改名。

## 三、Artifact 行为

后端需要建立 artifact 索引，不建议前端使用本地绝对路径。

建议 artifact 类型：

```text
json
image
mask
zarr
csv
report
other
```

图片类：

- `preview_rgb`
- `m1_1_corrected_box`
- `m1_2_overlay`
- `m1_3_reconstructed_strip`
- `segment_image`
- `segment_mask`

JSON 类：

- `preprocess_context`
- `pipeline_result`
- `m1_transform_stack`
- `segments_with_cube_refs`
- `quality_report`

## 四、权限行为

后端必须在接口层执行权限判断，前端按钮控制只是辅助。

权限建议：

| 权限 | 后端行为 |
|---|---|
| `preprocessing:read` | 允许读取任务、上下文、箱体、图斑、普通 artifact |
| `preprocessing:create` | 允许创建任务 |
| `preprocessing:cancel` | 允许取消任务 |
| `preprocessing:review` | 允许提交复核动作 |
| `preprocessing:export` | 允许下载大文件或导出 |
| `preprocessing:admin` | 允许查看服务器路径、模型路径、运行环境 |

## 五、安全与路径要求

必须避免前端传入任意路径导致后端越权读写。

实现要求：

- `output_dir` 必须限制在允许的工作目录。
- `manifest_path`、`input_image`、`model_package` 必须经过白名单或项目目录校验。
- artifact 下载必须通过 `artifact_id` 映射，不直接信任 query path。
- 错误信息可以包含文件名，但不要向普通用户暴露敏感系统目录。

## 六、验收清单

后端完成后，至少通过以下检查：

- 可以创建任务并返回 `202`。
- 可以轮询 running/succeeded/failed。
- 成功任务能返回 `preprocess_context`。
- `boxes` 能返回 M1-1/M1-2/M1-3 关联结果。
- `segments` 能分页返回 `CoreSegment`，且包含 `cube_ref`。
- artifact 接口可以预览 PNG/JPG。
- 无权限用户无法 create/cancel/review/export。
- OpenAPI 生成的前端类型与真实响应一致。

## 七、当前已知待实现项

- 将 CLI 管道封装为 FastAPI 异步任务服务。
- 建立任务持久化存储，至少保存 task metadata 和 output index。
- 建立 artifact id 到本地文件的映射表。
- 将 review action 持久化，并设计是否触发局部重算。
- 将 M1-2 模型包路径纳入后端配置，而不是让普通用户任意输入。
- 将 warnings 从字符串逐步升级为结构化 warning 对象。
