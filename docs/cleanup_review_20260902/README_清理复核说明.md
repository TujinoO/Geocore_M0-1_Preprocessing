# 综合项目生成物与缓存清理复核说明

日期：2026-09-02
源码项目：`E:\Code\Geocore_M0&1_Preprocessing`
待手动删除目录：`E:\Code\_DELETE_REVIEW_Geocore_M0and1_Preprocessing_20260902`

## 结论

- 已可逆移出 20 项、270 个文件、72,525,143 字节（约 69.17 MiB）。
- 其中历史 `runs/smoke_pipeline` 生成结果 175 个文件，Python 缓存 95 个文件。
- 搬移前后逐文件 SHA-256 不匹配 0，缺失目标 0，源路径残留 0。
- 没有删除任何文件或目录。
- 待删除目录状态为 `ReadyForManualDelete: true`，确认后可在文件管理器中手动删除整个目录。

这些内容不属于可维护源码、模型权重、训练数据或保留型 QA；历史 smoke run 的工程结论仍在项目 README 中保留，但原始生成图像和中间结果不再占用源码树。

## 审计文件

- `MOVE_PLAN.csv`：20 项互斥搬移映射。
- `PREMOVE_FILE_MANIFEST_SHA256.csv`：搬移前逐文件清单和 SHA-256。
- `MOVED_FILE_MANIFEST_SHA256.csv`：搬移后路径及哈希核验。
- `STATUS.json`：机器可读清理状态。
- `EXECUTE_REVERSIBLE_CLEANUP.ps1`：本次执行脚本，仅移动、不删除。
- `RESTORE.ps1`：待删除目录仍存在且内容未变化时的恢复脚本。

## 恢复

在尚未手动删除待删除目录、且原路径没有新同名内容时执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\Code\Geocore_M0&1_Preprocessing\docs\cleanup_review_20260902\RESTORE.ps1"
```

恢复脚本会先校验逐文件 SHA-256；发现路径冲突、文件缺失或哈希变化时立即停止，不覆盖、不删除。
