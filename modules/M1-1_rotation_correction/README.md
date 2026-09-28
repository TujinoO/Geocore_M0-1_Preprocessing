# Geocore M1-1 岩心箱旋转校正

本仓库实现 Geo-Core AI V2.1 后端核心算法模块 `M1-1 岩心箱旋转校正`。当前实现基于 Python、NumPy 和 Pillow，不依赖 OpenCV。

## 2026-09-28 跨箱型候选更新

- 先保留已验证橙色宽框配对路径；窄两槽橙色箱可由长向橙色侧壁收紧左右范围，避免旧固定 17%–86% 裁幅把大量传送带纳入箱体。
- 对 7520 像素大幅面采用更长的箱体周期与跨宽度横梁证据，降低岩石断口导致的半箱切分；普通横向梯度候选均设置 `geometry_review_required=true`，即使角度置信度高也不能自动验收。
- 散落岩石/无箱画面仍可能产生候选框；必须结合箱数、叠加图和人工箱界标签进行最终判定。这里没有跨场景准确率或漏检率结论。
- 原始及待标注数据盘点、最终空白 SHP 队列见 `H:\Data_M1-1&M1-3_box_identification_and_reassembly`。不得把候选框当作人工真值。

第 2 轮人工 SHP 回归修正了 7520 像素宽暗色箱：不再用橙褐色岩石列充当箱体外壁；放宽宽箱的左右搜寻范围，并对明显的箱间暗带前缀做保守裁剪。16 个已标注箱原始 ENVI 上的最佳框平均 bbox IoU 从 0.8655 提至 0.9129（开发集回归）；两个无箱负裁片仍各有假候选，未通过跨场景验收。详见综合项目的 `docs/M11_M13_标注驱动优化第2轮_20260928.md`。

## 功能

1. 读取 ENVI `dat + hdr` 长条 RGB 影像。
2. 自动检测竖向连续采集图中的多个岩心箱。
3. 按岩心箱水平边框拆分独立箱体。
4. 基于左右竖边拟合估计旋转角。
5. 使用最近邻采样旋转，保护像元值。
6. 输出校正影像、掩膜、元数据和 QA 预览图。

## 运行

```powershell
$RunRoot = "E:\Experiment_data\GeoCore_Preprocessing_Runs"
python -m geocore_m1_1.cli `
  --input assets/legacy_rgb_20230909/RGB-20230909_141858-00000.dat `
  --hdr assets/legacy_rgb_20230909/RGB-20230909_141858-00000.hdr `
  --output "$RunRoot\M1-1\m1_1" `
  --expected-box-count 9
```

如果系统默认 `python -c` 不可用，可使用 Codex 捆绑 Python：

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m geocore_m1_1.cli `
  --input assets/legacy_rgb_20230909/RGB-20230909_141858-00000.dat `
  --hdr assets/legacy_rgb_20230909/RGB-20230909_141858-00000.hdr `
  --output "$RunRoot\M1-1\m1_1" `
  --expected-box-count 9
```

## 输出

```text
outputs/m1_1/
  corrected_boxes/
  masks/
  review_sources/
  previews/
  metadata.json
  qa_report.md
```

## 后端 API 服务

启动服务：

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m geocore_m1_1.server `
  --host 127.0.0.1 `
  --port 8765
```

### 1. 健康检查

```http
GET /api/m1-1/health
```

### 2. 自动执行 M1-1

```http
POST /api/m1-1/run
Content-Type: application/json
```

```json
{
  "input_path": "E:/Code/Geocore_M0&1_Preprocessing/modules/M1-1_rotation_correction/assets/legacy_rgb_20230909/RGB-20230909_141858-00000.dat",
  "hdr_path": "E:/Code/Geocore_M0&1_Preprocessing/modules/M1-1_rotation_correction/assets/legacy_rgb_20230909/RGB-20230909_141858-00000.hdr",
  "output_dir": "E:/Experiment_data/GeoCore_Preprocessing_Runs/M1-1/m1_1",
  "expected_box_count": 9,
  "save_preview": true
}
```

返回 `metadata.json` 等价结构，包含每个岩心箱的 `bbox_xyxy_raw`、`angle_deg`、`confidence`、`needs_manual_review`、`review_source_image`、`source_crop_bbox_raw`、`output_image` 和 `output_mask`。

### 3. 查询处理结果

```http
GET /api/m1-1/results?output_dir=E:/Experiment_data/GeoCore_Preprocessing_Runs/M1-1/m1_1
```

### 4. 查询待人工复核项

```http
GET /api/m1-1/review-items?output_dir=E:/Experiment_data/GeoCore_Preprocessing_Runs/M1-1/m1_1
```

如需返回全部箱体：

```http
GET /api/m1-1/review-items?output_dir=...&only_needs_review=false
```

### 5. 提交人工矩形/多边形校正

前端或主软件应在 `review_source_image` 上进行标注，坐标单位为该复核图的像素坐标。

矩形提交：

```http
POST /api/m1-1/manual-correction
Content-Type: application/json
```

```json
{
  "output_dir": "E:/Experiment_data/GeoCore_Preprocessing_Runs/M1-1/m1_1",
  "input_path": "E:/Code/Geocore_M0&1_Preprocessing/modules/M1-1_rotation_correction/assets/legacy_rgb_20230909/RGB-20230909_141858-00000.dat",
  "hdr_path": "E:/Code/Geocore_M0&1_Preprocessing/modules/M1-1_rotation_correction/assets/legacy_rgb_20230909/RGB-20230909_141858-00000.hdr",
  "box_id": "box_0006",
  "annotation": {
    "type": "rectangle",
    "rect": {
      "x": 40,
      "y": 40,
      "width": 1320,
      "height": 2300
    }
  }
}
```

多边形提交：

```json
{
  "output_dir": "E:/Experiment_data/GeoCore_Preprocessing_Runs/M1-1/m1_1",
  "input_path": "E:/Code/Geocore_M0&1_Preprocessing/modules/M1-1_rotation_correction/assets/legacy_rgb_20230909/RGB-20230909_141858-00000.dat",
  "hdr_path": "E:/Code/Geocore_M0&1_Preprocessing/modules/M1-1_rotation_correction/assets/legacy_rgb_20230909/RGB-20230909_141858-00000.hdr",
  "box_id": "box_0007",
  "annotation": {
    "type": "polygon",
    "points": [
      {"x": 42, "y": 55},
      {"x": 1370, "y": 38},
      {"x": 1392, "y": 2200},
      {"x": 51, "y": 2220}
    ]
  }
}
```

后端会将标注坐标映射回原始长条影像，重新执行最近邻旋转校正，覆盖对应的 `corrected_boxes/{box_id}.png` 和 `masks/{box_id}_mask.png`，并同步更新 `metadata.json`、`qa_report.md`。

### 6. 读取输出文件

```http
GET /api/m1-1/file?path=E:/Experiment_data/GeoCore_Preprocessing_Runs/M1-1/m1_1/review_sources/box_0006_source.jpg
```

## 测试

```powershell
python -m unittest discover -s tests
```

## 合并后的独立维护资产

- 完整 ENVI 回归样例：`assets/legacy_rgb_20230909/`。
- 既有自动校正与人工复核 QA：`qa/legacy_outputs_20260714/`。
- 新运行输出统一写入综合项目外的任务目录，不覆盖 `qa/`。

独立回归示例：

```powershell
python -m geocore_m1_1.cli `
  --input assets/legacy_rgb_20230909/RGB-20230909_141858-00000.dat `
  --hdr assets/legacy_rgb_20230909/RGB-20230909_141858-00000.hdr `
  --output "$RunRoot\M1-1\regression_20230909" `
  --expected-box-count 9
```

本模块是确定性几何校正算法，不包含需要训练的模型；日后调整应复跑单元测试和上述 ENVI 回归样例，并与 `qa/legacy_outputs_20260714/metadata.json`、预览和复核结果比较。
