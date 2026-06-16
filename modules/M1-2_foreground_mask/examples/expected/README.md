# Expected Output

The API writes generated foreground-mask GeoTIFF files to `api_runtime/outputs/{task_id}`
by default.

This directory is reserved for manually verified reference outputs. It is left
empty because reference masks should be regenerated with the same runtime,
weights, GDAL version, and PyTorch version used for delivery acceptance.
