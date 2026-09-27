# Geo-Core AI M1-3 岩心柱体分割与深度标记

本仓库实现 M1-3 后端算法模块：承接 M1-1 旋转校正影像和 M1-2 前景掩膜输出，将岩心箱中多列岩心按深度顺序重建为长条状岩心影像，并切分为带深度信息的标准 CoreSegment 图斑。

## 快速运行

```powershell
python -m geocore_m1_3.cli segment-depth `
  --image D:\data\corrected_boxes\box_0008.png `
  --m1-2-output-dir D:\data\m1_2_outputs\box_0008 `
  --output-dir D:\data\m1_3_outputs `
  --hole-id DH001 `
  --core-box-id DH001_BOX_0008 `
  --depth-start-m 70.0 `
  --depth-end-m 75.0
```

主要输出包括：

- `refined_mask.png`
- `lane_detection.json`
- `reconstructed_strip.png`
- `segments/`
- `segments.json`
- `quality_report.json`

## 当前实现特性

- 纯 `numpy + Pillow` 核心算法，不依赖 OpenCV。
- 支持 M1-2 标准输出目录输入。
- 默认从前景掩膜逐箱判断槽数；3 槽和 5 槽真实样例已通过几何回归。判定不可靠时停止，显式槽数与可靠判定冲突时也停止。
- 橙色箱体的上下横梁若能可靠识别，会从掩膜中排除；不影响其他箱型。默认保留 V4 掩膜，不做未经验证的二次形态学修改。
- 可配置列顺序和列方向。默认值只是一种排列约定，真实浅→深方向仍须通过箱号/野外记录核对；未确认时输出 `depth_order_unverified`。
- 支持箱外误检剔除、岩心列重建、线性深度映射、固定长度/重叠图斑切分。
- 提供 CLI 和可选 FastAPI 适配层。

## 合并后的独立维护资产

- 历史完整烟测输出：`qa/legacy_outputs_20260714/smoke_box_0008/`。
- 新运行输出应写入综合项目外任务目录，不覆盖既有 QA。
- 本模块为确定性布局、重建和深度映射算法，不包含训练权重；修改后应运行：

```powershell
python -m unittest discover -s tests
```

并用历史 `smoke_box_0008` 的 `lane_detection.json`、`depth_mapping.json`、`segments.json` 和 `quality_report.json` 做结构及行为回归比较。
