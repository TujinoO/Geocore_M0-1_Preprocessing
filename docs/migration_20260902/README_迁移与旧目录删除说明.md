# Geo-Core M0/M1 四项目合并与旧目录删除说明

日期：2026-09-02
目标项目：`E:\Code\Geocore_M0&1_Preprocessing`

## 1. 迁移结论

- 40 项一对一映射已经完成。
- 465 个必要文件、536,853,133 字节（约 511.98 MiB）已迁入综合项目对应模块。
- 移动前后逐文件 SHA-256 一致：不匹配 0、缺失目标 0、残留源路径 0。
- 没有删除任何文件或目录。
- 四个旧项目剩余文件均已分类为可复算融合立方体、生成型测试输出或 Python 字节码缓存；未解释残留 0。

## 2. 各模块迁入内容

| 模块 | 映射数 | 文件数 | 约 MiB | 迁入内容 |
|---|---:|---:|---:|---|
| M0-1 影像融合 | 28 | 177 | 92.09 | 3 个独有源码脚本、对齐 ENVI 样例、配准/融合配置、指标、预览和 QA |
| M1-1 旋转校正 | 3 | 48 | 174.08 | 唯一完整 RGB ENVI 样例及自动/人工复核 QA |
| M1-2 前景掩膜 | 8 | 114 | 226.59 | 模型源码、V1/V2 权重、模型卡、训练历史、标注数据、划分和验证输出 |
| M1-3 分割标记 | 1 | 126 | 19.23 | 完整 smoke_box_0008 结构化结果与预览 QA |

## 3. 为什么没有迁移 M0-1 的大型 Zarr

旧 M0-1 中四套 `fused_cube.zarr` 合计约 3.38 GiB，是同一对齐输入在 `classical`、`deep_unsupervised`、`fast_preview`、`upsample_only` 四种模式下生成的结果。为了避免综合源码项目再次膨胀，本次保留并迁入了：

- 对齐后的 RGB/NIR/SWIR ENVI 输入；
- 每种模式的 manifest、处理配置、配准/光谱模型、质量指标；
- 结果预览和跨模式比较图；
- ROI manifest 与 tie-point 会话。

因此大型 Zarr 具备复算条件，作为旧目录剩余的可删除生成成果处理。该判断是工程可复算性判断，不代表四种融合方法的科学效果已被独立验证。

## 4. 可直接人工删除的旧目录

以下四个旧目录当前状态均为 `READY_FOR_MANUAL_DELETE`：

1. `E:\Code\Geocore_M0-1_image_fusion`：剩余 1,183 个文件、3,626,829,277 字节；仅可复算 Zarr、生成测试输出和 `.pyc`。
2. `E:\Code\Geocore_M1-1_rotation_correction`：剩余 19 个 `.pyc`，103,197 字节。
3. `E:\Code\Geocore_M1-2_foreground_mask`：剩余 40 个 `.pyc`，165,698 字节。
4. `E:\Code\Geocore_M1-3_separator_mark`：剩余 35 个 `.pyc`，102,432 字节。

本次未替用户删除目录。确认审计清单后，可在文件管理器中手动删除上述四个完整文件夹。

## 5. 恢复

在尚未手动删除旧目录、且原源路径没有新同名内容时，可以执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\Code\Geocore_M0&1_Preprocessing\docs\migration_20260902\RESTORE.ps1"
```

恢复脚本会先校验当前迁入文件 SHA-256，发现缺失、变化或路径冲突即停止；不会覆盖或删除内容。手动删除旧目录后，不再建议执行恢复脚本。

## 6. 审计文件

- `MOVE_PLAN.csv`：迁移来源、目标与理由。
- `PREMOVE_FILE_INVENTORY.csv`：移动前哈希清单。
- `MIGRATED_FILE_MANIFEST_SHA256.csv`：移动完成时的双向哈希比较。
- `POST_NORMALIZATION_FILE_MANIFEST_SHA256.csv`：路径规范化后的当前哈希。
- `MOVE_SUMMARY.csv`：每项映射的数量、体量和验证结果。
- `LEGACY_REMAINDER_MANIFEST_SHA256.csv`：旧目录全部剩余文件及分类。
- `OLD_FOLDER_DELETE_READINESS.csv`：四个旧目录的删除就绪状态。
- `AUDIT_STATUS.json`、`POST_NORMALIZATION_VERIFICATION.json`：机器可读总状态。
- `VERIFY_POST_NORMALIZATION.ps1`：只读重算当前迁入文件哈希并刷新迁移后验证材料。
