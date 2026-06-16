$WorkspaceRoot = "D:\Code\Geocore_M0and1_Preprocessing"
$ModulePaths = @(
  "$WorkspaceRoot\modules\M0-1_image_fusion\src",
  "$WorkspaceRoot\modules\M1-1_rotation_correction",
  "$WorkspaceRoot\modules\M1-2_foreground_mask",
  "$WorkspaceRoot\modules\M1-3_separator_mark",
  "$WorkspaceRoot\src"
)
$existing = @()
if ($env:PYTHONPATH) {
  $existing = $env:PYTHONPATH -split ';' | Where-Object { $_ }
}
$env:PYTHONPATH = (($ModulePaths + $existing) | Select-Object -Unique) -join ';'
Write-Host "PYTHONPATH updated for Geo-Core M0/M1 preprocessing modules."
