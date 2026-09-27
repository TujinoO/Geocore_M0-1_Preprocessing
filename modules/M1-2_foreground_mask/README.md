# Geo-Core AI M1-2 岩心柱前景掩膜算法模块

本目录已经从原始 `AGRS_semantic_segmentation` 遥感语义分割工程整理为
Geo-Core AI 岩心智能编录软件的 **M1-2 岩心柱前景掩膜提取模块**。

旧工程中的 `networks/`、`train.py`、`predict.py` 等文件仅作为历史实验流程保留。新的软件发布主流程位于：

```text
geocore_mask/
api/
configs/
models/
```

## 模块定位

M1-2 模块负责从岩心箱或岩心柱影像中提取岩心实体区域，输出二值掩膜、概率图、预览图、轮廓和质量元数据。其结果将作为后续 M1-3 岩心柱体分割与深度标记、M2 属性识别和前端可视化的基础。

## 推荐使用方式

发布前由算法团队完成：

1. 人工勾画岩心柱前景样本；
2. 训练并评估模型；
3. 将权重放入模型包；
4. 更新模型卡和模型清单。

当前 M0/M1 统一入口固定调用 V4 模型包；以下 V1 目录仅为旧版资料，不能当作当前默认部署版本。发布后软件调用的 V4 模型包为：

```text
models/core_mask_unet_v4/
  weights.pth
  model_manifest.json
  model_card.md
```

当前模型清单路径：

```text
models/core_mask_unet_v4/model_manifest.json
```

Git 源码不包含约 69 MB 的 V4 权重。新克隆仓库先在项目根目录运行 `./scripts/install_v4_model.ps1` 下载权重并校验 SHA-256；模型清单标明这是需人工复核的开发验证版本，不代表跨场景无人值守验收。

## API 启动

在本目录下执行：

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

打开接口文档：

```text
http://127.0.0.1:8000/docs
```

## 推荐请求

```json
{
  "input_path": "examples/input",
  "image_pattern": "*.tif",
  "model_profile": "default",
  "threshold": 0.5,
  "enable_postprocess": true,
  "output_preview": true
}
```

## 输出结果

单图任务默认输出：

```text
mask.png
mask.tif
probability.tif
overlay.png
contours.json
metadata.json
```

其中：

- `mask.png`：前端快速预览二值掩膜；
- `mask.tif`：后续算法使用的原尺寸掩膜；
- `probability.tif`：前景概率图；
- `overlay.png`：原图与掩膜叠加预览；
- `contours.json`：连通域轮廓、包围盒和面积；
- `metadata.json`：模型版本、阈值、耗时、面积比例和质量警告。

## 新代码结构

```text
geocore_mask/
  inference/
    predictor.py
    tiler.py
    merger.py

  postprocess/
    refine_mask.py
    contour.py

  models/
    registry.py

  utils/
    image_io.py
    metrics.py
    model_package.py
```

## 兼容说明

为了降低迁移风险，API 仍兼容旧的 ad-hoc 请求字段，例如 `model_path`、`model_name`、`mean`、`std`、`target_size` 和 `overlap_rate`。正式发布时建议使用 `model_profile` 或 `model_package`，不要让前端传入训练细节。

当前默认发布模型 `CoreMaskUNet` 已内置在 `geocore_mask/models/core_unet.py`，不再依赖旧 `networks/` 目录。

## 合并后的训练与 QA 资产

- 模型源码：`geocore_mask/models/`。
- V1/V2 模型包：`models/core_mask_unet_v1/`、`models/core_mask_unet_v2/`。
- 标注、SHP 来源、训练划分和候选图像：`datasets/`。
- 历史批量验证：`qa/legacy_outputs_20260714/`。
- API 烟测：`qa/legacy_api_runtime_outputs_20260714/`。
- 小型独立推理输入：`examples/input/`。

在本模块目录中独立重新训练：

```powershell
$RunRoot = "E:\Experiment_data\GeoCore_Preprocessing_Runs"
python tools/train_core_mask.py `
  --config configs/train_core_mask_v2.json `
  --base-package models/core_mask_unet_v1 `
  --output-package "$RunRoot\M1-2\models\core_mask_unet_v2_candidate"
```

独立批量推理：

```powershell
python tools/predict_core_mask_batch.py `
  --input-dir datasets/corrected_boxes `
  --output-dir "$RunRoot\M1-2\core_mask_v2_regression" `
  --model-package models/core_mask_unet_v2
```

现有数据集只有少量岩心箱，适合工程复现和继续标注，不足以证明跨钻孔泛化；重新训练后必须按钻孔/岩心箱独立划分测试集，不能仅凭训练集或同箱切片指标替换正式模型。
