# ArcGIS SHP 标注到岩心前景训练样本自动化指南

本文档说明如何在拿到一批新的岩心箱影像后，用最少的人工操作完成样本标注，并通过脚本自动生成 M1-2 岩心柱前景掩膜模型所需的训练数据。

## 1. 总体目标

人工只负责一件事：

```text
在 ArcGIS 中沿岩心实体边界勾画 polygon，并保存为同名 .shp。
```

其余步骤由脚本自动完成：

```text
影像 + ArcGIS shp
  -> 自动栅格化为 0/255 二值 mask
  -> 自动生成 overlay 质检图
  -> 自动划分 train / val / test
  -> 自动生成训练配置
  -> 后续用于模型训练或微调
```

## 2. 推荐目录结构

建议每批新数据建立一个独立数据集目录，例如：

```text
D:\Code\Geocore_M1-2_foreground_mask\datasets\core_mask_v2\
  images\
    box_0001.png
    box_0002.png
    box_0003.png

  shp\
    box_0001.shp
    box_0001.shx
    box_0001.dbf
    box_0001.prj
    box_0002.shp
    box_0002.shx
    box_0002.dbf
    box_0002.prj

  masks\
  overlays\
  splits\
```

注意：

- 每张影像推荐对应一个同名 shapefile。
- `.shp` 不是单个文件，至少要保留 `.shp`、`.shx`、`.dbf`。
- 如果 ArcGIS 生成了 `.prj`、`.cpg`，也要一起保留。
- 不要把新数据覆盖到旧数据集里，建议用 `core_mask_v2`、`core_mask_202606` 这类带版本含义的目录名。

## 3. ArcGIS 人工标注步骤

对每张影像执行以下操作：

1. 在 ArcGIS 中加载岩心箱影像，例如 `box_0001.png`。
2. 新建 polygon 类型的 shapefile。
3. shapefile 文件名与影像主文件名保持一致，例如：

```text
box_0001.png -> box_0001.shp
```

4. 在同一个 ArcGIS 工程中确认影像和 shapefile 完全重叠。
5. 开始编辑，沿岩心实体外边界勾画 polygon。
6. 保存编辑结果。
7. 将 `.shp` 及其附属文件放入数据集的 `shp\` 目录。

## 4. 标注规则

只需要一个类别：岩心前景。

需要标为前景：

```text
完整圆柱岩心
半圆柱岩心
破碎岩块
明显属于岩心的碎块
岩心断面
岩心侧面
```

不要标为前景：

```text
岩心箱边框
箱内隔板
黑色背景
箱外区域
岩心之间的阴影缝隙
标签纸
胶带
反光塑料膜
文字牌
松散粉尘和泥渣
```

如果一张影像里有很多块岩心，可以画多个 polygon。它们都属于同一个前景类别，不需要给每块岩心编号。

## 5. 自动生成训练样本

在仓库根目录执行：

```powershell
python tools\prepare_arcgis_shp_dataset.py `
  --dataset-root datasets\core_mask_v2 `
  --config-out configs\train_core_mask_v2.json `
  --experiment-name core_mask_unet_v2 `
  --overwrite
```

脚本会自动完成：

```text
1. 扫描 images\ 中的影像
2. 查找 shp\ 中的同名 shapefile
3. 将 polygon 栅格化为 mask
4. 生成 overlay 质检图
5. 生成 train.txt / val.txt / test.txt
6. 生成训练配置 JSON
7. 输出 prepare_report.json 质检报告
```

默认输出：

```text
datasets\core_mask_v2\
  masks\
    box_0001.png
    box_0002.png

  overlays\
    box_0001_overlay.png
    box_0002_overlay.png

  splits\
    train.txt
    val.txt
    test.txt

  prepare_report.json

configs\
  train_core_mask_v2.json
```

## 6. 常用命令参数

只处理 PNG：

```powershell
python tools\prepare_arcgis_shp_dataset.py `
  --dataset-root datasets\core_mask_v2 `
  --image-pattern *.png `
  --config-out configs\train_core_mask_v2.json `
  --experiment-name core_mask_unet_v2
```

处理 TIF：

```powershell
python tools\prepare_arcgis_shp_dataset.py `
  --dataset-root datasets\core_mask_v2 `
  --image-pattern *.tif `
  --config-out configs\train_core_mask_v2.json `
  --experiment-name core_mask_unet_v2
```

允许部分影像暂时没有标注：

```powershell
python tools\prepare_arcgis_shp_dataset.py `
  --dataset-root datasets\core_mask_v2 `
  --allow-missing-shp
```

边界像素尽量纳入前景：

```powershell
python tools\prepare_arcgis_shp_dataset.py `
  --dataset-root datasets\core_mask_v2 `
  --all-touched
```

自定义训练集比例：

```powershell
python tools\prepare_arcgis_shp_dataset.py `
  --dataset-root datasets\core_mask_v2 `
  --train-ratio 0.7 `
  --val-ratio 0.2 `
  --test-ratio 0.1
```

## 7. 质检方法

脚本生成 `overlays\` 后，优先检查 overlay 图，而不是直接看 mask。

重点检查：

```text
岩心有没有明显漏标
箱格和边框有没有被误标
标签纸和胶带有没有被误标
polygon 是否整体错位
破碎岩块是否覆盖合理
```

如果发现问题：

1. 回到 ArcGIS 修改对应 `.shp`。
2. 重新运行脚本。
3. 再检查对应 overlay。

## 8. 输出 mask 规范

自动生成的 mask 是模型训练需要的二值图：

```text
岩心前景 = 255
背景 = 0
```

mask 文件名与影像主文件名一致：

```text
images\box_0001.png
masks\box_0001.png
```

mask 尺寸必须与原图完全一致。脚本会按原图尺寸生成 mask。

## 9. split 文件格式

脚本会生成：

```text
splits\train.txt
splits\val.txt
splits\test.txt
```

每一行格式为：

```text
images/box_0001.png masks/box_0001.png
```

这表示一张训练影像及其对应的二值 mask。

## 10. 常见问题

### shapefile 缺少附属文件

如果只有 `.shp`，没有 `.shx` 或 `.dbf`，脚本可能无法读取。

正确做法是复制完整 shapefile 文件组：

```text
box_0001.shp
box_0001.shx
box_0001.dbf
box_0001.prj
box_0001.cpg
```

### overlay 整体偏移

说明 ArcGIS 中的 shapefile 与影像坐标没有正确对齐。

处理方法：

```text
回到 ArcGIS，确认影像和 shp 在同一坐标空间内完全重叠。
```

### 生成的 mask 全黑

常见原因：

```text
shp 没有 polygon
polygon 不在影像范围内
影像没有地理参考且 shp 坐标不是像素坐标
```

处理方法：

```text
打开 prepare_report.json 查看 warnings
检查 overlay
回 ArcGIS 修正 shp
```

### 标签纸被模型识别成岩心

优先补充包含标签纸、胶带、反光塑料膜的困难样本，并确保这些区域在人工标注中保持为背景。

## 11. 推荐批处理流程

每批新影像建议按以下节奏推进：

```text
1. 先标 10-20 张代表性困难样本
2. 运行脚本生成 mask 和 overlay
3. 人工检查 overlay
4. 修正问题 shp
5. 扩展到 50-100 张样本
6. 进入模型微调
7. 用新模型预测更多未标注影像
8. 人工只修正预测错误，再形成下一批训练样本
```

这样可以逐步降低人工标注量，最终让人工从“完整勾画”转为“检查和修正”。

