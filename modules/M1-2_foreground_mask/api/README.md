# Geo-Core AI M1-2 Foreground Mask API

This API exposes the rock core foreground mask extraction module as a FastAPI
service. The public API is release-oriented: the frontend submits an image path,
chooses a packaged model profile, and receives mask files, preview files,
contours and metadata.

The old remote-sensing style fields such as `model_path`, `mean`, `std`,
`band_num` and `target_size` are still accepted for temporary compatibility, but
the recommended release mode is to use a model package under `models/`.

## Install

Install PyTorch and GDAL for the target machine first, then install Python
dependencies:

```powershell
pip install -r requirements.txt
```

GDAL is still useful for reading/writing TIF files and preserving spatial
metadata when the input image has it.

## Model Package

The default profile reads:

```text
models/core_mask_unet_v1/model_manifest.json
```

Before formal deployment, put the trained model weights here:

```text
models/core_mask_unet_v1/weights.pth
```

The model manifest controls the model type, normalization statistics, tile size,
overlap, threshold and post-processing parameters. The frontend does not need to
send these internal parameters.

## Start

Run from `AGRS_semantic_segmentation-main`:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

OpenAPI docs:

```text
http://127.0.0.1:8000/docs
```

Static frontend contract and mock data:

```text
docs/openapi.foreground-mask.json
docs/mocks/
api-change-requests.md
```

Use the static OpenAPI file when the frontend needs stable generated types or
mock-server setup. `api-change-requests.md` records permission behavior and
backend follow-up items for host-system integration.

## Endpoints

- `GET /api/system/status`
- `GET /api/m1/foreground-mask/models`
- `POST /api/m1/foreground-mask/jobs`
- `POST /api/preprocessing/foreground-mask`
- `POST /api/m1/foreground-mask/upload`
- `GET /api/m1/foreground-mask/jobs/{task_id}`
- `GET /api/preprocessing/status/{task_id}`
- `POST /api/m1/foreground-mask/jobs/{task_id}/mask-edits`
- `POST /api/preprocessing/foreground-mask/{task_id}/mask-edits`
- `GET /api/m1/foreground-mask/jobs/{task_id}/download`

## Recommended Request

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

PowerShell example:

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

## Output

For a single image, outputs are written to `api_runtime/outputs/{task_id}` by
default:

```text
mask.png
mask.tif
probability.tif
overlay.png
contours.json
metadata.json
```

The status response includes:

- `output_files`: generated file paths;
- `metrics`: foreground area ratio, component count and elapsed time;
- `warnings`: quality warnings such as unusually small foreground area.

## Manual Mask Edit

After the model returns the first foreground mask, the frontend can send user
selected regions back to the backend. The backend removes those regions from the
final binary mask and refreshes `mask.png`, `mask.tif`, `overlay.png`,
`contours.json` and `metadata.json`. `probability.tif` is kept unchanged because
it represents the original model probability map.

Rectangle example:

```json
{
  "operation": "remove",
  "regions": [
    {
      "type": "rectangle",
      "bbox": [120, 80, 260, 180]
    }
  ],
  "output_preview": true
}
```

Polygon plus brush example from a scaled preview canvas:

```json
{
  "operation": "remove",
  "display_width": 1024,
  "display_height": 512,
  "regions": [
    {
      "type": "polygon",
      "points": [[300, 120], [430, 126], [420, 230], [290, 220]]
    },
    {
      "type": "brush",
      "radius": 12,
      "points": [[520, 200], [540, 210], [565, 220]]
    }
  ]
}
```

PowerShell example:

```powershell
$edit = @{
  operation = "remove"
  regions = @(
    @{ type = "rectangle"; bbox = @(120, 80, 260, 180) }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/m1/foreground-mask/jobs/$($job.task_id)/mask-edits" `
  -ContentType "application/json" `
  -Body $edit
```
