# Geo-Core AI 预处理模块前端联调说明

本文档面向前端开发人员，说明 M0/M1 预处理汇总项目的业务功能、页面数据结构、接口契约、权限行为和 mock 使用方式。

## 一、前端开发边界

前端开发遵循严格的前后端分离原则：

- 前端只改前端项目代码，不直接修改后端算法实现。
- 页面字段必须来自 OpenAPI。
- 如果发现字段不够，先修改 OpenAPI 和 mock。
- 同步在 `api-change-requests.md` 记录给后端的实现说明。
- 再基于更新后的类型开发页面。

本项目的接口契约文件是：

```text
E:\Code\Geocore_M0&1_Preprocessing\docs\openapi\preprocessing.openapi.json
```

mock 示例目录是：

```text
E:\Code\Geocore_M0&1_Preprocessing\docs\mocks\preprocessing
```

## 二、业务功能总览

预处理模块把影像采集数据转换为可编录、可识别、可追溯的标准图斑。

前端需要理解四个阶段：

| 阶段 | 模块 | 前端关注点 | 主要输出 |
|---|---|---|---|
| M0-1 | 影像融合 | manifest、质量报告、预览图、Zarr 引用 | `manifest.json`、`fused_cube.zarr`、`band_metadata.csv` |
| M1-1 | 旋转校正 | 箱体列表、旋转角、置信度、校正图 | corrected box、rotation matrix、transform stack |
| M1-2 | 前景掩膜 | mask、overlay、置信度、人工编辑 | mask、probability、contours、metadata |
| M1-3 | 分割与深度标记 | 图斑列表、深度范围、质量标记 | CoreSegment、depth mapping、cube_ref |

总管道完成后，前端主要读取：

```text
preprocess_context.json
segments_with_cube_refs.json
```

## 三、推荐页面

### 1. 预处理任务列表页

用途：查看所有预处理任务。

建议展示字段：

- `task_id`
- `project.project_id`
- `project.hole_id`
- `status`
- `stage`
- `progress`
- `created_at`
- `updated_at`
- `metrics.box_count`
- `metrics.segment_count`
- `warnings.length`

操作：

- 新建任务：需要 `preprocessing:create`
- 查看详情：需要 `preprocessing:read`
- 取消任务：需要 `preprocessing:cancel`

### 2. 新建预处理任务页

用途：提交从 M0 manifest 到 M1-3 的完整任务。

表单字段：

- `manifest_path`：已有 M0 融合结果入口。
- `m1_1_input_image`：可选，全分辨率 RGB 或 ENVI dat。
- `m1_1_hdr_path`：当 `m1_1_input_image` 是 `.dat` 时必填。
- `project.project_id`
- `project.hole_id`
- `project.core_box_prefix`
- `depth.depth_start_m`
- `depth.depth_end_m`
- `m1_2.engine`：`model`、`auto`、`classical`。
- `m1_2.model_package`：生产模式建议必填。
- `m1_3.segment_length_cm`
- `m1_3.overlap_cm`
- `m1_3.lane_count`

前端校验：

- `depth_end_m` 必须大于 `depth_start_m`。
- `m1_2.engine=model` 时必须提供模型包或模型 profile。
- 如果输入 `.dat`，必须同时提供 `.hdr`。

### 3. 任务详情页

用途：查看阶段进度、质量指标、中间产物。

建议组件：

- 阶段进度条：`queued -> m0_fusion -> m1_1_rotation -> m1_2_mask -> m1_3_segments -> succeeded`
- 质量卡片：M0 配准、M1-1 角度、M1-2 前景比例、M1-3 图斑数量。
- warnings 面板。
- artifacts 下载入口。

### 4. M1-1 岩心箱复核页

数据来源：

- `GET /api/preprocessing/jobs/{task_id}/boxes`
- `preprocess_context.boxes`

展示字段：

- `box_id`
- `core_box_id`
- `order_index`
- `angle_deg`
- `confidence`
- `bbox_xyxy_m1_input`
- `bbox_xyxy_rgb_reference`
- `corrected_image`
- `m1_1_mask`

操作：

- 查看 corrected image。
- 查看 before/after preview。
- 标记人工复核意见。

### 5. M1-2 前景掩膜复核页

数据来源：

- `box.m1_2.output_files.mask_png`
- `box.m1_2.output_files.overlay_png`
- `box.m1_2.output_files.probability_tif`
- `box.m1_2.metrics.foreground_area_ratio`

操作：

- 查看 overlay。
- 切换 mask/probability。
- 提交 remove/add/replace 类型复核动作。

权限：

- 只读查看：`preprocessing:read`
- 提交复核：`preprocessing:review`

### 6. M1-3 深度图斑页

数据来源：

- `GET /api/preprocessing/jobs/{task_id}/segments`
- `segments_with_cube_refs.json`

展示字段：

- `segment_id`
- `core_box_id`
- `depth_start_m`
- `depth_end_m`
- `depth_center_m`
- `length_cm`
- `image_path`
- `mask_path`
- `core_coverage_ratio`
- `missing_ratio`
- `quality_flags`
- `cube_ref`

前端无需直接展示完整 `cube_ref`，但应保留该对象给后续 M2/M3 页面使用。

## 四、状态机

任务状态：

```text
queued
running
succeeded
failed
cancelled
needs_review
```

阶段枚举：

```text
initializing
m0_fusion
m1_1_rotation
m1_2_foreground_mask
m1_3_separator_mark
finalizing
done
```

页面行为建议：

- `queued/running`：显示进度条，禁用重复提交。
- `succeeded`：展示结果和下载入口。
- `failed`：展示 `error.code`、`error.message`、`error.detail`。
- `needs_review`：突出复核入口。
- `cancelled`：显示取消人和取消时间。

## 五、权限行为

前端应调用：

```http
GET /api/preprocessing/permissions
```

并根据返回值控制按钮：

| 权限 | 页面行为 |
|---|---|
| `preprocessing:read` | 可以查看任务、上下文、图斑和产物 |
| `preprocessing:create` | 显示新建任务按钮 |
| `preprocessing:cancel` | 显示取消任务按钮 |
| `preprocessing:review` | 显示人工复核、掩膜编辑、深度锚点编辑 |
| `preprocessing:export` | 显示下载/导出按钮 |
| `preprocessing:admin` | 显示运行环境、模型包路径、系统状态 |

如果无权限，按钮应禁用并展示 `denied_reason`，不要只隐藏关键操作。

## 六、接口与 mock

OpenAPI：

```text
docs/openapi/preprocessing.openapi.json
```

常用 mock：

```text
docs/mocks/preprocessing/system-status.ok.json
docs/mocks/preprocessing/modules.response.json
docs/mocks/preprocessing/create-job.request.json
docs/mocks/preprocessing/create-job.response.json
docs/mocks/preprocessing/task.running.json
docs/mocks/preprocessing/task.succeeded.json
docs/mocks/preprocessing/context.response.json
docs/mocks/preprocessing/boxes.response.json
docs/mocks/preprocessing/segments.page.json
docs/mocks/preprocessing/permissions.response.json
```

前端开发流程：

1. 用 OpenAPI 生成 TypeScript 类型。
2. 用 mock 驱动页面开发。
3. 页面字段不够时，先改 OpenAPI。
4. 在 `api-change-requests.md` 写清楚后端实现请求。
5. 后端实现完成后替换 mock 为真实 API。

## 七、重要数据结构

### PreprocessingJob

任务级数据。用于列表页和详情页。

核心字段：

- `task_id`
- `status`
- `stage`
- `progress`
- `input`
- `output_files`
- `metrics`
- `warnings`
- `error`

### PreprocessContext

完整预处理上下文。用于任务详情和跨模块追溯。

核心字段：

- `fusion`
- `m1_1`
- `m1_2`
- `m1_3`
- `boxes`
- `warnings`

### CoreBoxSummary

M1-1 之后的岩心箱对象。

核心字段：

- `box_id`
- `core_box_id`
- `bbox_xyxy_rgb_reference`
- `corrected_image`
- `m1_2`
- `m1_3`
- `segments_with_cube_refs`

### CoreSegment

M1-3 后的图斑对象。

核心字段：

- `segment_id`
- `depth_start_m`
- `depth_end_m`
- `image_path`
- `mask_path`
- `cube_ref`
- `transform_stack_ref`

## 八、前端注意事项

- 不要尝试在浏览器中直接加载完整 `fused_cube.zarr`。
- 图斑页面只展示预览图、掩膜和 metadata。
- `cube_ref` 作为后续算法请求参数保存，不作为大文件下载入口。
- 所有路径字段由后端返回，前端不要拼接服务器本地路径。
- 文件预览应通过 artifact 下载/预览接口访问，不直接暴露本地绝对路径。
- 对 `warnings`、`quality_flags`、`needs_review` 要给明显视觉反馈。

## 九、接口变更纪律

任何接口变更必须同时更新：

1. `docs/openapi/preprocessing.openapi.json`
2. `docs/mocks/preprocessing/*.json`
3. `api-change-requests.md`
4. 前端类型和页面逻辑

后端未实现前，前端可以基于 mock 开发，但不能私自约定 OpenAPI 之外的字段。
