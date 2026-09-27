# Geo-Core AI M0/M1 预处理汇总工程

面向岩心 RGB/NIR/SWIR 数据的统一预处理与可审计编排工具，串联影像融合、旋转校正、前景掩膜、柱体分割和深度标记四个模块。

> **项目状态：研究与工程集成阶段。** 仓库提供可维护源码、接口契约、mock、测试和轻量模型元数据；原始影像、训练集、模型权重及运行结果不进入常规 Git 历史。链路 smoke test 通过不等同于生产验收或科学有效性验证。

本工程是 Geo-Core AI 岩心智能编录软件的数据预处理总入口，统一归纳并串联四个算法模块：

- `M0-1_image_fusion`：高光谱-光学影像融合。
- `M1-1_rotation_correction`：岩心箱旋转校正。
- `M1-2_foreground_mask`：岩心柱体前景掩膜。
- `M1-3_separator_mark`：岩心柱体分割与深度标记。

当前目标不是把四个模块揉成一个不可维护的大文件，而是在汇总项目中建立统一的数据契约、调用入口、OpenAPI、mock 和前后端协作规范。四个算法模块仍保留清晰边界，预处理总管道负责把它们接成一条稳定流程。

## 2026-09-02 模块合并状态

本地完整工作区已将四个旧独立目录中的必要源码、模型、训练资产、回归样例和轻量 QA 按功能归入 `modules/`。GitHub 源码仓库只跟踪源码、配置、文档、测试及轻量模型元数据；大型资产继续由独立数据存储管理。各模块仍可单独运行、测试或训练，统一入口只负责编排，不改变模块内部算法边界：

| 模块 | 独立维护目录 | 可独立执行内容 |
|---|---|---|
| M0-1 | `modules/M0-1_image_fusion` | 四种融合模式、流式 ENVI、配准诊断与回归 QA |
| M1-1 | `modules/M1-1_rotation_correction` | 旋转校正、人工复核 API、完整 ENVI 回归样例 |
| M1-2 | `modules/M1-2_foreground_mask` | 推理、API、两套模型包、标注数据和重新训练 |
| M1-3 | `modules/M1-3_separator_mark` | 掩膜修正、列重建、深度映射与图斑切分 |

具体命令、资产位置和训练边界见 [四模块独立维护指南](docs/FOUR_MODULE_MAINTENANCE_GUIDE.md)。旧目录迁移证据见 [迁移审计](docs/migration_20260902/README_迁移与旧目录删除说明.md)，综合项目生成物与缓存的复核记录见 [清理复核](docs/cleanup_review_20260902/README_清理复核说明.md)。审计文档记录的是特定本地工作区快照，不代表公开仓库包含其中列出的二进制资产。

## 一、工程定位

预处理阶段负责把原始 RGB/NIR/SWIR 岩心影像转化为后续 M2/M3 可直接使用的标准图斑和数据引用。

完整流程如下：

```text
M0-1 影像融合
  输入 RGB/NIR/SWIR .dat + .hdr
  输出 manifest.json + fused_cube.zarr + band_metadata.csv
        |
        v
M1-1 旋转校正
  输入 manifest.json 解析出的 RGB 参考图或原始 RGB/ENVI 图
  输出 corrected_boxes、旋转矩阵、m1_transform_stack.json
        |
        v
M1-2 前景掩膜
  输入 M1-1 corrected box
  输出 mask.png、mask.tif、probability.tif、contours.json、metadata.json
        |
        v
M1-3 分割与深度标记
  输入 corrected box + M1-2 mask + 深度参数
  输出 CoreSegment、reconstructed strip、depth mapping
        |
        v
preprocess_context.json + segments_with_cube_refs.json
  后续 M2/M3 按 CoreSegment 引用读取 Zarr 小块
```

## 二、目录结构

```text
Geocore_M0-1_Preprocessing/
  README.md
  api-change-requests.md
  pyproject.toml

  configs/
    module_paths.json

  docs/
    M0_M1_unified_preprocessing_pipeline.md
    frontend_preprocessing_integration.md
    openapi/
      preprocessing.openapi.json
    mocks/
      preprocessing/

  modules/
    M0-1_image_fusion/
    M1-1_rotation_correction/
    M1-2_foreground_mask/
    M1-3_separator_mark/

  scripts/
    set_pythonpath.ps1

  src/
    geocore_preprocessing/
      cli.py
      pipeline.py
      fusion_accessor.py
      classical_mask.py
      paths.py
      json_utils.py
```

说明：

- `modules/` 下保存四个算法模块迁移后的代码、配置、文档和测试代码。
- `src/geocore_preprocessing/` 是汇总工程新增的统一编排层。
- `docs/openapi/preprocessing.openapi.json` 是前后端协同开发的接口单一来源。
- `docs/mocks/preprocessing/` 保存前端页面开发可直接使用的 mock 示例。
- `api-change-requests.md` 记录需要后端实现或调整的接口说明。
- 大型样例数据、模型权重、运行输出、训练集和缓存不纳入 Git 跟踪，避免源码仓库失控膨胀。

## 三、核心输出契约

总管道成功后，输出目录中至少包含：

```text
preprocess_context.json
m1_transform_stack.json
pipeline_result.json
m1_1_rotation/
m1_2_foreground_mask/
m1_3_separator_mark/
```

最关键的后续交付物是：

```text
m1_3_separator_mark/<core_box_id>/segments_with_cube_refs.json
```

每个 `CoreSegment` 会包含：

- `segment_id`：图斑 ID。
- `depth_start_m`、`depth_end_m`、`depth_center_m`：深度信息。
- `image_path`、`mask_path`：图斑预览和掩膜。
- `cube_ref.manifest_path`：M0 融合 manifest。
- `cube_ref.cube_path`：融合高光谱 Zarr。
- `cube_ref.band_metadata`：波段元数据。
- `transform_stack_ref`：M1-1 坐标变换栈。
- `bbox_rgb_reference_parent_box`：图斑所在父箱体在 M0 RGB 参考网格中的范围。
- `source_refs`：M1-1/M1-2/M1-3 中间产物引用。

后续 M2-6 矿物识别、M2-5 岩性识别等模块应读取这些引用，而不是复制整块高光谱数据。

## 四、运行环境

需要 Python 3.10 或更高版本。克隆并安装统一编排包：

```powershell
git clone https://github.com/TujinoO/Geocore_M0-1_Preprocessing.git
Set-Location Geocore_M0-1_Preprocessing
python -m pip install -e .
```

在仓库根目录设置四个子模块的 Python 路径：

```powershell
. .\scripts\set_pythonpath.ps1
```

脚本根据自身位置解析仓库根目录，不依赖盘符或检出目录名称。若只检查统一编排包，也可临时指定：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
```

查看模块路径：

```powershell
python -m geocore_preprocessing.cli modules
```

检查 M0 manifest：

```powershell
python -m geocore_preprocessing.cli inspect-manifest `
  --manifest D:\path\to\fusion_result\manifest.json
```

## 五、运行完整预处理管道

从已有 M0 融合结果启动：

```powershell
python -m geocore_preprocessing.cli run `
  --manifest D:\path\to\fusion_result\manifest.json `
  --output-dir D:\path\to\preprocessing_run `
  --hole-id ZKH3 `
  --core-box-prefix ZKH3_132_140 `
  --depth-start-m 132.0 `
  --depth-end-m 140.0 `
  --m1-2-engine model `
  --m1-2-model-package .\modules\M1-2_foreground_mask\models\core_mask_unet_v4 `
  --segment-length-cm 10
```

如果 M0 预览图只是缩略图，M1-1 应使用全分辨率 RGB/ENVI 输入：

```powershell
python -m geocore_preprocessing.cli run `
  --manifest D:\path\to\fusion_result\manifest.json `
  --m1-1-input-image D:\path\to\RGB-20230909_141858-00000.dat `
  --m1-1-hdr-path D:\path\to\RGB-20230909_141858-00000.hdr `
  --output-dir D:\path\to\preprocessing_run `
  --depth-start-m 132.0 `
  --depth-end-m 140.0 `
  --m1-2-engine model `
  --m1-2-model-package .\modules\M1-2_foreground_mask\models\core_mask_unet_v4
```

调试时可使用轻量兜底掩膜：

```powershell
python -m geocore_preprocessing.cli run `
  --manifest D:\path\to\fusion_result\manifest.json `
  --output-dir D:\path\to\preprocessing_run `
  --depth-start-m 132.0 `
  --depth-end-m 140.0 `
  --m1-2-engine classical
```

`classical` 仅用于链路测试，不是生产模型。

## 六、前后端协作原则

本项目严格采用前后端分离协作模式：

1. 前端开发人员主要修改前端代码，不直接改后端算法代码。
2. 页面需要的数据结构、接口字段、权限行为必须先在 OpenAPI 中定义。
3. 如需新增或调整字段，先更新 `docs/openapi/preprocessing.openapi.json`。
4. 同步补充 `docs/mocks/preprocessing/` 中的 mock 示例。
5. 在 `api-change-requests.md` 记录给后端的实现说明。
6. 前端基于 OpenAPI 生成类型或手写类型后，再开发页面。
7. 后端实现以 OpenAPI 和 mock 为验收口径。

## 七、前端页面建议

建议前端围绕以下页面组织：

- 预处理任务列表页：查看任务状态、进度、阶段、失败原因。
- 新建预处理任务页：选择 manifest 或 M0 输入，填写深度、模型参数和处理策略。
- 任务详情页：展示 M0/M1-1/M1-2/M1-3 各阶段状态和产物。
- 岩心箱复核页：查看 M1-1 corrected box、旋转角、置信度、是否需人工复核。
- 前景掩膜复核页：查看 mask、overlay、confidence/probability，提交编辑建议。
- 深度图斑页：浏览 CoreSegment、深度范围、质量标记和 cube_ref。
- 运行日志页：展示 warnings、metrics、artifact 下载入口。

详见：

```text
docs/frontend_preprocessing_integration.md
docs/openapi/preprocessing.openapi.json
docs/mocks/preprocessing/
```

## 八、权限模型

OpenAPI 中将权限行为明确为以下动作：

- `preprocessing:read`：查看任务、上下文、产物、mock 结构。
- `preprocessing:create`：创建预处理任务。
- `preprocessing:cancel`：取消运行中的任务。
- `preprocessing:review`：提交人工复核、掩膜编辑、深度锚点修正。
- `preprocessing:export`：下载或导出产物。
- `preprocessing:admin`：查看系统状态、模型路径、底层运行配置。

前端需要根据 `/api/preprocessing/permissions` 返回值决定按钮是否可见、是否禁用、以及禁用原因。

## 九、开发与校验

基础导入校验：

```powershell
. .\scripts\set_pythonpath.ps1
python -B -m geocore_preprocessing.cli --help
python -B -m geocore_preprocessing.cli run --help
```

运行现有模块测试：

```powershell
python -B -m pytest modules\M1-1_rotation_correction\tests modules\M1-3_separator_mark\tests
python -B modules\M0-1_image_fusion\tests\run_tests.py
```

已完成过一次端到端 smoke run。为避免源码项目继续充当过程成果仓库，原始生成结果已可逆移出源码树；搬移清单、逐文件哈希和恢复说明见 `docs/cleanup_review_20260902/`。

该 smoke run 使用 `M1-2 classical` 兜底模式验证数据链路，生成了：

- `preprocess_context.json`
- `m1_transform_stack.json`
- `segments_with_cube_refs.json`

当前 M0&1 默认固定为 `core_mask_unet_v4`；模型权重不在 Git 中，必须先安装同版权重并通过 SHA-256 校验。未安装时默认 `model` 模式显式报错，不会悄悄改用 V1/V2 或 classical。`classical` 仅供明确指定的调试/回退使用。V4 目前只有开发与工程验证证据，新的独立钻孔验收仍待完成；应用到新场景时须人工复核掩膜。

## 十、后续维护建议

- `manifest.json` 和 `preprocess_context.json` 是跨模块边界，修改前必须同步 OpenAPI。
- `segments_with_cube_refs.json` 是 M2/M3 的核心输入，字段调整要记录兼容策略。
- 大型数据、模型权重、运行输出不要提交到源码仓库；模型权重应放置在本机 `modules/M1-2_foreground_mask/models/<profile>/`，或通过显式参数指向外部模型包。
- 新增前端页面前，先确认 OpenAPI、mock 和权限行为是否完整。
- 新增后端接口前，先在 `api-change-requests.md` 记录变更动机、字段、兼容策略和验收样例。
