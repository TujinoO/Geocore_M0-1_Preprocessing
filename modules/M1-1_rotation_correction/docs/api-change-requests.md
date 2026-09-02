# M1-1 API Change Requests

本文记录前端接入 M1-1 岩心箱旋转校正页面前，需要后端确认、补充或在主系统集成层实现的接口事项。

## 已新增契约与 mock

- OpenAPI: `docs/openapi/m1-1.openapi.yaml`
- Mock 请求与响应: `mocks/m1-1/`
- 覆盖接口:
  - `GET /api/m1-1/health`
  - `POST /api/m1-1/run`
  - `GET /api/m1-1/results`
  - `GET /api/m1-1/review-items`
  - `POST /api/m1-1/manual-correction`
  - `GET /api/m1-1/file`

## 后端需要确认的字段契约

1. `POST /api/m1-1/run`
   - 请求字段以 OpenAPI `RunRequest` 为准。
   - `input_path` 当前为后端本机可访问路径，不是上传流。
   - `hdr_path` 在 ENVI `dat` 输入时必填，普通图片可为空。
   - `expected_box_count` 如果与实际检测数量不一致，后端应返回结构化错误。
   - 返回体必须稳定符合 `BatchResult`，尤其是 `boxes[]` 内的路径、角度、置信度、复核状态字段。

2. `GET /api/m1-1/results`
   - 返回 `metadata.json` 的完整内容。
   - `output_dir` 不传时当前默认读取 `outputs/m1_1`。
   - 集成主系统后建议改为 `task_id` 或 `run_id` 查询，避免前端直接传服务器路径。

3. `GET /api/m1-1/review-items`
   - 默认只返回 `needs_manual_review=true` 的箱体。
   - `only_needs_review=false` 时返回全部箱体。
   - 返回项结构与 `CoreBoxResult` 保持一致，不要返回简化结构导致前端另写类型。

4. `POST /api/m1-1/manual-correction`
   - 前端提交的标注坐标基于 `review_source_image` 的原始像素坐标。
   - 支持矩形 `annotation.type=rectangle` 和多边形 `annotation.type=polygon`。
   - 成功后返回更新后的单个 `CoreBoxResult`，并将 `needs_manual_review=false`、`correction_status=manual`。
   - 后端应保存 `manual_annotation`，便于审计和复现人工校正。

5. `GET /api/m1-1/file`
   - 当前开发服务使用 `path` 读取本机文件。
   - 接入主系统时建议改成 `task_id + relative_path` 或文件资源 ID。
   - 如果短期保留 `path`，必须做路径白名单和项目权限校验。

## 权限与安全要求

当前独立算法服务不做认证授权，仅用于本地联调。主系统接入时后端或网关必须实现以下权限行为：

| 接口 | 所需权限 | 行为要求 |
| --- | --- | --- |
| `GET /api/m1-1/health` | `public:health` 或登录态 | 如对普通用户开放，至少要求登录态 |
| `POST /api/m1-1/run` | `m1_1:execute`, `project:file:read`, `project:file:write` | 只能处理用户有权访问的输入文件，并写入授权输出目录 |
| `GET /api/m1-1/results` | `m1_1:read`, `project:file:read` | 只能读取当前用户所属项目的结果 |
| `GET /api/m1-1/review-items` | `m1_1:read`, `project:file:read` | 只能查看有权项目的复核项 |
| `POST /api/m1-1/manual-correction` | `m1_1:review`, `project:file:read`, `project:file:write` | 只能复核有权项目的箱体，并记录标注人和标注时间 |
| `GET /api/m1-1/file` | `m1_1:read`, `project:file:read` | 文件路径必须限制在授权任务输出目录或项目文件目录内 |

## 建议后端补充

1. 增加任务标识
   - 建议在 `POST /run` 响应中增加 `task_id` 或 `run_id`。
   - 后续查询、文件读取、人工复核优先使用该标识，减少前端传绝对路径。

2. 增加异步任务接口
   - 当前 `POST /run` 同步阻塞。
   - 大图处理进入主系统后建议拆为：创建任务、查询任务状态、查询任务结果。

3. 增加统一错误码
   - 当前错误主要是 `status/error/exception_type`。
   - 建议增加 `code`，例如 `FILE_NOT_FOUND`、`INVALID_ANNOTATION`、`BOX_NOT_FOUND`、`PERMISSION_DENIED`。

4. 限制文件读取路径
   - `/api/m1-1/file?path=...` 当前能读取服务进程可访问的任意文件。
   - 主系统集成前必须限制到项目数据目录和当前 M1-1 输出目录。

5. 记录人工复核审计信息
   - 建议在 metadata 或主系统数据库中记录 `reviewed_by`、`reviewed_at`、`annotation`、`previous_box_state`。

## 前端接入约定

- 前端基于 `docs/openapi/m1-1.openapi.yaml` 生成请求和响应类型。
- 页面开发先使用 `mocks/m1-1/` 中的 JSON 示例。
- 图片展示统一通过 `GET /api/m1-1/file?path=${encodeURIComponent(path)}` 加载。
- 人工标注提交前必须把页面显示坐标换算回图片原始像素坐标。
- 人工校正成功后应重新调用 `GET /api/m1-1/results` 获取最新 metadata。
