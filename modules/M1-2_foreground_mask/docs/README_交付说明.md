# Geo-Core AI M1-2 岩心柱前景掩膜 API 交付说明

本文档面向前端开发工程师和软件集成工程师。前端不需要直接调用 Python 算法函数，只需要启动 FastAPI 服务，然后通过 HTTP 接口调用岩心柱前景掩膜提取能力。

## 目录说明

```text
api/                         FastAPI 服务层
geocore_mask/                岩心前景掩膜新主流程
models/core_mask_unet_v1/    默认模型包目录
configs/                     训练和推理配置示例
examples/                    请求样例和测试输入
networks/                    旧 AGRS 网络结构，继续作为模型库使用
api_runtime/                 运行时上传和输出目录
```

## 模型权重放置

正式发布前，将训练好的权重放到：

```text
models/core_mask_unet_v1/weights.pth
```

并检查：

```text
models/core_mask_unet_v1/model_manifest.json
```

其中记录了模型名称、输入通道、均值方差、滑窗尺寸、重叠率、阈值和后处理参数。

## 环境准备

建议使用 Python 3.9 或 3.10。

```powershell
conda create -n geocore-mask-api python=3.10
conda activate geocore-mask-api
```

根据机器情况安装 PyTorch。GPU 机器安装 CUDA 版本，普通机器安装 CPU 版本。

Windows 上建议使用 conda 安装 GDAL：

```powershell
conda install -c conda-forge gdal
```

然后安装项目依赖：

```powershell
pip install -r requirements.txt
```

## 启动服务

方式一：双击 `run_api_server.bat`。

方式二：在本目录执行：

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

启动后打开：

```text
http://127.0.0.1:8000/docs
```

## 前端调用流程

1. 请求系统状态：

```text
GET /api/system/status
```

2. 创建前景掩膜任务：

```text
POST /api/preprocessing/foreground-mask
```

推荐请求体：

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

3. 从响应中读取 `task_id`。

4. 轮询任务状态：

```text
GET /api/preprocessing/status/{task_id}
```

5. 当状态为 `succeeded` 时，从 `output_files` 中读取：

```text
mask_png
mask_tif
probability_tif
overlay_png
contours_json
metadata_json
```

也可以调用：

```text
GET /api/m1/foreground-mask/jobs/{task_id}/download
```

该接口默认下载 `mask.tif`。

## PowerShell 调用示例

```powershell
$body = Get-Content .\examples\request_example.json -Raw
$job = Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/preprocessing/foreground-mask `
  -ContentType 'application/json' `
  -Body $body

$job
Invoke-RestMethod "http://127.0.0.1:8000/api/preprocessing/status/$($job.task_id)"
```

## 输出文件说明

```text
api_runtime/outputs/{task_id}/
  mask.png
  mask.tif
  probability.tif
  overlay.png
  contours.json
  metadata.json
```

- `mask.png`：0/255 二值掩膜，用于浏览器快速预览。
- `mask.tif`：原尺寸掩膜，用于后续 M1-3 和 M2 算法。
- `probability.tif`：前景概率图，用于模型分析和阈值检查。
- `overlay.png`：原图叠加掩膜预览。
- `contours.json`：连通域轮廓、面积、中心点和包围盒。
- `metadata.json`：模型版本、阈值、前景面积比例、连通域数量、耗时和质量警告。

## 验收要点

- `GET /api/system/status` 返回 `status: ok`。
- `http://127.0.0.1:8000/docs` 可访问。
- 使用 `examples/request_example.json` 能创建任务并返回 `task_id`。
- 轮询任务状态最终返回 `succeeded`。
- 输出目录中生成 `mask.png`、`mask.tif`、`probability.tif`、`overlay.png`、`contours.json` 和 `metadata.json`。
