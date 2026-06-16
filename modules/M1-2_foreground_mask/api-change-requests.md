# API Change Requests

本文档记录 M1-2 岩心柱前景掩膜算法模块与前端联调相关的接口契约、权限边界和需要后端确认或实现的事项。前端开发时以 `docs/openapi.foreground-mask.json` 为当前静态契约，以 `docs/mocks/` 为 mock 数据来源。

## 当前契约基线

- OpenAPI 契约文件：`docs/openapi.foreground-mask.json`
- Mock 示例目录：`docs/mocks/`
- 推荐前端业务接口：
  - `POST /api/preprocessing/foreground-mask`
  - `GET /api/preprocessing/status/{task_id}`
  - `POST /api/preprocessing/foreground-mask/{task_id}/mask-edits`
- 模块级兼容接口：
  - `/api/m1/foreground-mask/...`
- 当前任务模式：异步任务创建后返回 `task_id`，前端轮询状态。
- 当前人工编辑能力：仅支持 `operation: "remove"` 删除误检前景区域。

## 权限行为约定

当前 standalone FastAPI 算法服务没有独立鉴权、用户隔离和任务持久化机制。集成到主系统时，建议由网关或宿主后端在转发到本算法服务前完成登录态和权限校验。

建议权限键：

| 权限键 | 行为 |
| --- | --- |
| `m1.foregroundMask.read` | 查看系统状态、模型列表、任务状态和结果路径 |
| `m1.foregroundMask.create` | 创建前景掩膜任务或上传图像 |
| `m1.foregroundMask.edit` | 提交人工掩膜编辑 |
| `m1.foregroundMask.download` | 下载掩膜结果 |
| `preprocessing.foregroundMask.read` | 在前处理业务流程中查看任务状态 |
| `preprocessing.foregroundMask.create` | 在前处理业务流程中创建任务 |
| `preprocessing.foregroundMask.edit` | 在前处理业务流程中编辑掩膜 |

如果主系统启用鉴权，建议未登录返回 `401`，权限不足返回 `403`。当前 standalone 服务不会主动返回这两个状态码，前端 mock 时可由主系统网关层模拟。

## 已确认字段

### 创建任务请求

前端正式场景优先使用：

- `input_path`
- `image_pattern`
- `recursive`
- `model_profile`
- `threshold`
- `enable_postprocess`
- `output_preview`

以下字段为旧版兼容或算法内部参数，前端默认不展示：

- `model_package`
- `model_name`
- `model_path`
- `num_classes`
- `band_num`
- `mean`
- `std`
- `target_size`
- `overlap_rate`
- `img_data_type`

### 任务状态响应

前端需要稳定依赖：

- `task_id`
- `status`
- `message`
- `input_paths`
- `output_dir`
- `output_files`
- `metrics`
- `warnings`
- `error`
- `created_at`
- `updated_at`

`output_files` 为动态 key 结构。单图常见 key：`mask_png`、`mask_tif`、`probability_tif`、`overlay_png`、`contours_json`、`metadata_json`。批量任务会使用 `{image_stem}.mask_png` 等前缀 key，并额外返回 `results_json`。

### 人工编辑请求

当前稳定字段：

- `operation`: 当前仅 `remove`
- `regions`: 至少 1 个区域
- `display_width` / `display_height`: 前端缩放画布尺寸
- `result_key_prefix`: 批量任务中指定待编辑图片
- `output_preview`: 是否刷新叠加预览图

坐标约定：左上角为 `(0, 0)`，x 向右，y 向下。未传 `display_width` 和 `display_height` 时，后端认为坐标为原始图像像素坐标。

## 后端待确认或实现事项

1. 静态 OpenAPI 发布方式  
   请后端确认是否将 `docs/openapi.foreground-mask.json` 作为契约源文件发布，或在 CI 中校验 FastAPI 自动生成 OpenAPI 与该静态契约保持一致。

2. 鉴权和权限落点  
   standalone 算法服务当前无鉴权。集成主系统时请确认由网关、主后端还是算法服务自身负责校验权限，并补齐 `401` / `403` 行为。

3. 结果文件访问方式  
   当前 `output_files` 返回后端文件路径。纯 Web 前端如果无法直接读取这些路径，需要后端或网关新增静态文件代理接口，例如按 `task_id` 和文件 key 获取图片/JSON。

4. 任务持久化  
   当前任务状态保存在内存中，服务重启后历史任务丢失。若前端需要历史任务列表、刷新页面后恢复任务或多人协作，需要后端增加任务持久化。

5. 进度信息  
   当前 `status` 和 `message` 可以表达粗粒度进度。若前端需要进度条，请后端新增 `progress` 字段，范围建议为 `0` 到 `1`。

6. 人工编辑扩展  
   当前仅支持删除误检区域。若前端需要补画漏检区域，请后端扩展 `operation: "add"`，并明确新增区域如何写入 `mask.tif`、`contours.json` 和 `metadata.json`。

7. 撤销/重做  
   当前没有撤销接口。若前端需要服务端撤销，请后端增加编辑历史快照或基于 `manual_edit_history` 的回滚接口。

8. 错误响应统一  
   当前沿用 FastAPI 默认 `{"detail": ...}`。如果主系统有统一错误结构，请后端在适配层转换，同时保持 OpenAPI 与前端类型同步。

## 前端 mock 使用建议

- 创建任务：`docs/mocks/create-job.request.json` 和 `docs/mocks/create-job.response.json`
- 轮询中：`docs/mocks/task-status.running.json`
- 任务成功：`docs/mocks/task-status.succeeded.json`
- 任务失败：`docs/mocks/task-status.failed.json`
- 人工编辑：`docs/mocks/mask-edit.request.json` 和 `docs/mocks/mask-edit.response.json`
- 轮廓文件：`docs/mocks/contours.sample.json`
- 元数据文件：`docs/mocks/metadata.sample.json`

前端生成类型时建议从 `docs/openapi.foreground-mask.json` 生成，避免直接从手写文档摘字段。
