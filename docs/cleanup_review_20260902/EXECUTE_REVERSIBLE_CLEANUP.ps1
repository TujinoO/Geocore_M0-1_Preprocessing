$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = "E:\Code\Geocore_M0&1_Preprocessing"
$reviewRoot = "E:\Code\_DELETE_REVIEW_Geocore_M0and1_Preprocessing_20260902"
$auditRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function New-MappingRow {
    param([string]$Id, [string]$Category, [string]$Source, [string]$Target)
    [pscustomobject]@{
        MappingId = $Id
        Category = $Category
        Source = $Source
        Target = $Target
    }
}

if (-not (Test-Path -LiteralPath $reviewRoot -PathType Container)) {
    throw "Review directory does not exist: $reviewRoot"
}

$mappings = @()
$runSource = Join-Path $workspace "runs"
if (Test-Path -LiteralPath $runSource -PathType Container) {
    $mappings += New-MappingRow "C001" "GENERATED_SMOKE_RUN" $runSource (Join-Path $reviewRoot "payload\workspace\runs")
}

$cacheDirs = @(Get-ChildItem -LiteralPath $workspace -Recurse -Force -Directory | Where-Object {
    $_.Name -in @("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache")
} | Sort-Object FullName)

$cacheIndex = 1
foreach ($cacheDir in $cacheDirs) {
    $relative = $cacheDir.FullName.Substring($workspace.TrimEnd('\').Length + 1)
    $mappings += New-MappingRow ("C{0:D3}" -f ($cacheIndex + 1)) "REGENERABLE_CACHE" $cacheDir.FullName (Join-Path $reviewRoot ("payload\workspace\" + $relative))
    $cacheIndex++
}

if ($mappings.Count -eq 0) {
    throw "No cleanup candidates found; nothing was moved."
}

$duplicateSources = @($mappings | Group-Object Source | Where-Object Count -gt 1)
$duplicateTargets = @($mappings | Group-Object Target | Where-Object Count -gt 1)
$missingSources = @($mappings | Where-Object { -not (Test-Path -LiteralPath $_.Source) })
$targetConflicts = @($mappings | Where-Object { Test-Path -LiteralPath $_.Target })
if ($duplicateSources.Count -or $duplicateTargets.Count -or $missingSources.Count -or $targetConflicts.Count) {
    throw "Cleanup plan rejected: duplicate source=$($duplicateSources.Count), duplicate target=$($duplicateTargets.Count), missing source=$($missingSources.Count), target conflict=$($targetConflicts.Count)"
}

$overlaps = for ($i = 0; $i -lt $mappings.Count; $i++) {
    for ($j = $i + 1; $j -lt $mappings.Count; $j++) {
        $left = [System.IO.Path]::GetFullPath($mappings[$i].Source).TrimEnd('\')
        $right = [System.IO.Path]::GetFullPath($mappings[$j].Source).TrimEnd('\')
        if ($left.StartsWith($right + '\', [System.StringComparison]::OrdinalIgnoreCase) -or $right.StartsWith($left + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            [pscustomobject]@{ Left = $left; Right = $right }
        }
    }
}
if (@($overlaps).Count) {
    throw "Cleanup plan rejected: overlapping sources=$(@($overlaps).Count)"
}

$mappings | Export-Csv -LiteralPath (Join-Path $auditRoot "MOVE_PLAN.csv") -NoTypeInformation -Encoding UTF8

$preInventory = foreach ($mapping in $mappings) {
    $sourceItem = Get-Item -LiteralPath $mapping.Source -Force
    $files = if ($sourceItem.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $sourceItem.FullName -Recurse -Force -File)
    } else {
        @($sourceItem)
    }
    foreach ($file in $files) {
        $relative = if ($sourceItem.PSIsContainer) {
            $file.FullName.Substring($sourceItem.FullName.TrimEnd('\').Length + 1)
        } else {
            $file.Name
        }
        $targetFile = if ($sourceItem.PSIsContainer) {
            Join-Path $mapping.Target $relative
        } else {
            $mapping.Target
        }
        [pscustomobject]@{
            MappingId = $mapping.MappingId
            Category = $mapping.Category
            SourceFile = $file.FullName
            TargetFile = $targetFile
            RelativeWithinMapping = $relative
            SizeBytes = [int64]$file.Length
            PreMoveSHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
}
$preInventory | Export-Csv -LiteralPath (Join-Path $auditRoot "PREMOVE_FILE_MANIFEST_SHA256.csv") -NoTypeInformation -Encoding UTF8

foreach ($mapping in $mappings) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $mapping.Target) -Force | Out-Null
    Move-Item -LiteralPath $mapping.Source -Destination $mapping.Target
}

$postInventory = foreach ($row in $preInventory) {
    $exists = Test-Path -LiteralPath $row.TargetFile -PathType Leaf
    $hash = if ($exists) { (Get-FileHash -LiteralPath $row.TargetFile -Algorithm SHA256).Hash } else { $null }
    [pscustomobject]@{
        MappingId = $row.MappingId
        Category = $row.Category
        OriginalSourceFile = $row.SourceFile
        CurrentReviewFile = $row.TargetFile
        SizeBytes = [int64]$row.SizeBytes
        PreMoveSHA256 = $row.PreMoveSHA256
        CurrentSHA256 = $hash
        TargetExists = $exists
        HashMatches = ($exists -and $hash -eq $row.PreMoveSHA256)
    }
}
$postInventory | Export-Csv -LiteralPath (Join-Path $auditRoot "MOVED_FILE_MANIFEST_SHA256.csv") -NoTypeInformation -Encoding UTF8

$sourceStillExists = @($mappings | Where-Object { Test-Path -LiteralPath $_.Source })
$missingTargets = @($mappings | Where-Object { -not (Test-Path -LiteralPath $_.Target) })
$hashFailures = @($postInventory | Where-Object { -not $_.HashMatches })
$status = if ($sourceStillExists.Count -eq 0 -and $missingTargets.Count -eq 0 -and $hashFailures.Count -eq 0) { "PASS" } else { "FAIL" }

$summary = [ordered]@{
    SchemaVersion = "1.0"
    CleanupStatus = $status
    CompletedAt = (Get-Date).ToString("o")
    Workspace = $workspace
    ReviewFolder = $reviewRoot
    MappingCount = $mappings.Count
    MovedFileCount = $postInventory.Count
    MovedBytes = [int64](($postInventory | Measure-Object SizeBytes -Sum).Sum)
    GeneratedSmokeRunFileCount = @($postInventory | Where-Object Category -eq "GENERATED_SMOKE_RUN").Count
    RegenerableCacheFileCount = @($postInventory | Where-Object Category -eq "REGENERABLE_CACHE").Count
    HashMismatchCount = $hashFailures.Count
    SourceStillExistsCount = $sourceStillExists.Count
    MissingTargetCount = $missingTargets.Count
    DirectDeletionCount = 0
    ReadyForManualDelete = ($status -eq "PASS")
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $auditRoot "STATUS.json") -Encoding UTF8
$summary | ConvertTo-Json -Depth 5

if ($status -ne "PASS") {
    throw "Reversible cleanup verification failed; do not delete the review folder."
}
