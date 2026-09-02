# Geo-Core M0/M1 四模块独立维护指南

本指南适用于合并后的 `E:\Code\Geocore_M0&1_Preprocessing`。统一管道位于 `src/geocore_preprocessing`，四个功能模块仍在 `modules/` 内独立维护。

## 1. 通用环境

在仓库根目录执行：

```powershell
. .\scripts\set_pythonpath.ps1
python -m geocore_preprocessing.cli modules
```

`configs/module_paths.json` 使用相对工作区根目录，不再依赖旧独立项目或固定磁盘盘符。

过程数据、实验结果和候选权重默认写到源码仓库之外。以下示例统一使用：

```powershell
$RunRoot = "E:\Experiment_data\GeoCore_Preprocessing_Runs"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
```

## 2. M0-1 影像融合

目录：`modules/M0-1_image_fusion`

- 源码：`src/geocore_m01_fusion/`
- 测试：`tests/run_tests.py`
- 独有配准脚本：`scripts/legacy_registration_20260717/`
- 对齐回归输入：`assets/legacy_zkh3_132_140/aligned_envi/`
- 历史指标和预览：`qa/legacy_registration_20260717/`

```powershell
Set-Location modules/M0-1_image_fusion
$env:PYTHONPATH = (Resolve-Path src)
python tests/run_tests.py
python -m geocore_m01_fusion.cli --demo --mode classical --output "$RunRoot\M0-1\demo_classical"
```

四套历史 Zarr 融合立方体没有迁入，它们是约 3.38 GiB 的可复算成果；输入、配置、manifest、指标和预览已经保留。

## 3. M1-1 旋转校正

目录：`modules/M1-1_rotation_correction`

- 源码：`geocore_m1_1/`
- 测试：`tests/`
- 完整 ENVI 回归样例：`assets/legacy_rgb_20230909/`
- 历史 QA：`qa/legacy_outputs_20260714/`

```powershell
Set-Location modules/M1-1_rotation_correction
python -m unittest discover -s tests
python -m geocore_m1_1.cli `
  --input assets/legacy_rgb_20230909/RGB-20230909_141858-00000.dat `
  --hdr assets/legacy_rgb_20230909/RGB-20230909_141858-00000.hdr `
  --output "$RunRoot\M1-1\regression_20230909" `
  --expected-box-count 9
```

该模块不含训练模型；调整检测、旋转或复核逻辑后应比较历史元数据、矩阵、箱数和 QA 预览。

## 4. M1-2 前景掩膜

目录：`modules/M1-2_foreground_mask`

- 推理/API：`geocore_mask/`、`api/`
- 模型源码：`geocore_mask/models/`
- V1/V2 模型包：`models/`
- 训练资产：`datasets/`
- 训练和批量推理：`tools/`
- 历史 QA：`qa/`

本机已验证的训练环境为 `geo_env2`（PyTorch `2.8.0+cu128`，CUDA 可用）。先激活环境：

```powershell
conda activate geo_env2
```

```powershell
Set-Location modules/M1-2_foreground_mask
python tools/train_core_mask.py `
  --config configs/train_core_mask_v2.json `
  --base-package models/core_mask_unet_v1 `
  --output-package "$RunRoot\M1-2\models\core_mask_unet_v2_candidate"
```

```powershell
python tools/predict_core_mask_batch.py `
  --input-dir datasets/corrected_boxes `
  --output-dir "$RunRoot\M1-2\core_mask_v2_regression" `
  --model-package models/core_mask_unet_v2
```

两套权重与数据已迁入，但现有样本规模不能证明跨钻孔泛化；正式替换模型前仍需按钻孔/岩心箱划分独立测试集并记录 Precision、Recall、Dice、IoU 和空白区误报。

## 5. M1-3 分割与深度标记

目录：`modules/M1-3_separator_mark`

- 源码：`geocore_m1_3/`
- 测试：`tests/`
- 历史烟测 QA：`qa/legacy_outputs_20260714/smoke_box_0008/`

```powershell
Set-Location modules/M1-3_separator_mark
python -m unittest discover -s tests
```

该模块不含训练权重；调整列检测、掩膜修正、重建或深度映射后，应与历史 QA 中的结构化 JSON 和预览进行回归比较。

## 6. 输出与版本控制

- 新运行、过程数据、实验结果和候选权重统一写入外部任务目录；不要把综合项目或四个模块目录作为成果仓库。
- `assets/`、`datasets/`、`qa/`、模型权重和运行结果为本地大资产，默认不纳入 Git；迁移审计提供逐文件 SHA-256。
- 源码、配置、模型卡、模型清单、训练历史、文档和测试应纳入版本控制。
- 工程回归或自动 QA 通过不等同模型泛化、地质有效性或科学真值验证。
