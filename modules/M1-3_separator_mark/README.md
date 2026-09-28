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
- 新增 RGB 物理隔板 2–8 槽格架候选，与前景掩膜的槽数分开记录；两种证据冲突或 RGB 候选不明确时输出人工复核警告，不把 RGB 候选直接当真值。
- 另有 RGB 长向暗隔板候选，输出 `rgb_dark_rail_candidate` 与 `rgb_dark_rail_suggestion.jpg`。红线仅供逐箱复核；自动掩膜分槽失败时仍停止，并保留 `lane_preflight_review.json` 与 `review_manifest.json`，经人工确认后才能把内部隔板传给 `lane_dividers_x`。
- 如已逐箱人工确认槽位，可在 `layout` 中提交 `lane_dividers_x: [x1, x2, ...]`（M1-1 校正后图像的内部隔板 x 像素坐标）；槽数即隔板数加一，空槽也保留。`lane_count` 可为 `0` 或与其一致的数值。不允许未确认的自动候选直接写入该字段。
- 独立 CLI 可用 `--lane-dividers-json` 读取上述 JSON 列表；综合流程可用 `--lane-dividers-by-box-json` 提供如 `{"box_0001":[1200,2400]}` 的逐箱映射。标注队列 PNG 与 M1-1 校正图并非同一坐标系，不能直接复制 SHP 顶点 x 值。
- 综合项目的 `tools/export_shp_dividers_for_m11.py` 可在已有 M1-1 `metadata.json` 后，将匹配的人工 SHP 槽位按旋转矩阵、裁剪偏移转换为同一次运行的逐箱隔板 JSON；未填写连续 `SLOT_IDX=1…N` 的样本会被跳过。`sample_005` 已修正并通过第 3 轮实际转换。
- 综合流程可显式传入 `--source-slot-priors-json`。同来源多箱人工标注提供的槽数只作冲突预警；没有逐箱复核隔板时，结果写 `physical_slot_count_unverified`，不得自动当作槽数真值。
- 支持 M1-2 标准输出目录输入。
- 默认从前景掩膜逐箱判断槽数；判定不可靠时停止，显式槽数与可靠判定冲突时也停止。旧 ZKZ4-5 15 箱曾在特定原生运行中自动判为 3 槽，但对本轮 16 个标注箱的固定 V4 掩膜，只有 3/16 自动判对且通过置信检查，不能将旧场景结果外推。详见综合项目第 3 轮报告。
- 橙色箱体的上下横梁若能可靠识别，会从掩膜中排除；不影响其他箱型。默认保留 V4 掩膜，不做未经验证的二次形态学修改。
- 可配置列顺序和列方向。默认值只是一种排列约定，真实浅→深方向仍须通过箱号/野外记录核对；未确认时输出 `depth_order_unverified`。
- 默认保留槽内空白并对大空隙提示人工复核；确认是人工摆放间隔后才显式使用 `--gap-policy close_artificial_gaps`。15 箱的保留空隙 QA 在综合项目工程验证报告中，不能作真实深度成果。
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
