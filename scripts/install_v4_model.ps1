$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$modelDir = Join-Path $projectRoot 'modules\M1-2_foreground_mask\models\core_mask_unet_v4'
$target = Join-Path $modelDir 'weights.pth'
$expected = '896495B3FACAE1369E95DC5E05DEF561B1CB7DF08B15F2B214B7BCC0D747A32D'
if (-not (Test-Path -LiteralPath $modelDir -PathType Container)) { throw "Model package is missing: $modelDir" }
if (Test-Path -LiteralPath $target) {
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $expected) { throw "Existing V4 weights have the wrong SHA-256: $target" }
    Write-Output "V4 weights already verified: $target"
    return
}
$download = Join-Path $modelDir 'weights.pth.download'
if (Test-Path -LiteralPath $download) { throw "An unfinished download already exists; review it manually: $download" }
$url = 'https://github.com/TujinoO/Geocore_M0-1_Preprocessing/releases/download/m0m1-v4-20260928/core_mask_unet_v4_weights.pth'
Invoke-WebRequest -Uri $url -OutFile $download
if ((Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash -ne $expected) { throw "Downloaded V4 weights failed SHA-256 verification; review: $download" }
Move-Item -LiteralPath $download -Destination $target
Write-Output "Installed and verified V4 weights: $target"
