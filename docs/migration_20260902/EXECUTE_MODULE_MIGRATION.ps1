$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = "E:\Code\Geocore_M0&1_Preprocessing"
$auditRoot = Join-Path $workspace "docs\migration_20260902"
$legacyM01 = "E:\Code\Geocore_M0-1_image_fusion"
$legacyM11 = "E:\Code\Geocore_M1-1_rotation_correction"
$legacyM12 = "E:\Code\Geocore_M1-2_foreground_mask"
$legacyM13 = "E:\Code\Geocore_M1-3_separator_mark"

function New-MappingRow {
    param([string]$Id, [string]$Module, [string]$Source, [string]$Target, [string]$Rationale)
    [pscustomobject]@{
        MappingId = $Id
        Module = $Module
        Source = $Source
        Target = $Target
        Rationale = $Rationale
    }
}

$m01Module = Join-Path $workspace "modules\M0-1_image_fusion"
$m11Module = Join-Path $workspace "modules\M1-1_rotation_correction"
$m12Module = Join-Path $workspace "modules\M1-2_foreground_mask"
$m13Module = Join-Path $workspace "modules\M1-3_separator_mark"
$m01Qa = Join-Path $m01Module "qa\legacy_registration_20260717"
$m01Asset = Join-Path $m01Module "assets\legacy_zkh3_132_140"
$m01Roi = Join-Path $legacyM01 "roi_outputs\ZKH3_132_140_roi_768x512_final_hsi_rgb_warp"

$mappings = @(
    New-MappingRow "M001" "M0-1" (Join-Path $legacyM01 "scripts\diagnose_current_geocorefusion_registration.py") (Join-Path $m01Module "scripts\legacy_registration_20260717\diagnose_current_geocorefusion_registration.py") "Unique maintainable diagnostic source"
    New-MappingRow "M002" "M0-1" (Join-Path $legacyM01 "scripts\prototype_local_registration.py") (Join-Path $m01Module "scripts\legacy_registration_20260717\prototype_local_registration.py") "Unique local-registration prototype source"
    New-MappingRow "M003" "M0-1" (Join-Path $legacyM01 "scripts\prototype_tiepoint_registration_v4.py") (Join-Path $m01Module "scripts\legacy_registration_20260717\prototype_tiepoint_registration_v4.py") "Unique guarded tie-point prototype source"
    New-MappingRow "M004" "M0-1" (Join-Path $legacyM01 "registration_diagnostics") (Join-Path $m01Qa "registration_diagnostics") "Compact cross-project registration diagnostics"
    New-MappingRow "M005" "M0-1" (Join-Path $legacyM01 "local_registration_prototype") (Join-Path $m01Qa "local_registration_prototype") "Prototype metrics and visual QA"
    New-MappingRow "M006" "M0-1" (Join-Path $legacyM01 "tiepoint_registration_v4") (Join-Path $m01Qa "tiepoint_registration_v4") "Tie-point reports and visual QA"
    New-MappingRow "M007" "M0-1" (Join-Path $legacyM01 "roi_outputs\alignment_diagnosis_ZKH3_132_140") (Join-Path $m01Qa "alignment_diagnosis_ZKH3_132_140") "Final ROI alignment diagnosis"
    New-MappingRow "M008" "M0-1" (Join-Path $m01Roi "aligned_envi") (Join-Path $m01Asset "aligned_envi") "Compact aligned ENVI inputs needed to reproduce fusion variants"
    New-MappingRow "M009" "M0-1" (Join-Path $m01Roi "fusion_v2_comparison") (Join-Path $m01Qa "fusion_v2_comparison") "Cross-method comparison metrics and previews"
    New-MappingRow "M010" "M0-1" (Join-Path $m01Roi "previews") (Join-Path $m01Qa "roi_previews") "ROI and registration previews"
    New-MappingRow "M011" "M0-1" (Join-Path $m01Roi "tiepoint_sessions") (Join-Path $m01Qa "tiepoint_sessions") "Tie-point session and context previews"
    New-MappingRow "M012" "M0-1" (Join-Path $m01Roi "roi_manifest.json") (Join-Path $m01Qa "roi_manifest.json") "ROI identity and source mapping manifest"
    New-MappingRow "M013" "M0-1" (Join-Path $m01Roi "fusion_v2_classical\manifest.json") (Join-Path $m01Qa "fusion_v2_classical\manifest.json") "Classical run manifest without large generated cube"
    New-MappingRow "M014" "M0-1" (Join-Path $m01Roi "fusion_v2_classical\metadata") (Join-Path $m01Qa "fusion_v2_classical\metadata") "Classical run configuration and registration metadata"
    New-MappingRow "M015" "M0-1" (Join-Path $m01Roi "fusion_v2_classical\metrics") (Join-Path $m01Qa "fusion_v2_classical\metrics") "Classical run quality metrics"
    New-MappingRow "M016" "M0-1" (Join-Path $m01Roi "fusion_v2_classical\previews") (Join-Path $m01Qa "fusion_v2_classical\previews") "Classical run previews"
    New-MappingRow "M017" "M0-1" (Join-Path $m01Roi "fusion_v2_deep_unsupervised\manifest.json") (Join-Path $m01Qa "fusion_v2_deep_unsupervised\manifest.json") "Deep-unsupervised run manifest without large generated cube"
    New-MappingRow "M018" "M0-1" (Join-Path $m01Roi "fusion_v2_deep_unsupervised\metadata") (Join-Path $m01Qa "fusion_v2_deep_unsupervised\metadata") "Deep-unsupervised configuration and registration metadata"
    New-MappingRow "M019" "M0-1" (Join-Path $m01Roi "fusion_v2_deep_unsupervised\metrics") (Join-Path $m01Qa "fusion_v2_deep_unsupervised\metrics") "Deep-unsupervised quality metrics"
    New-MappingRow "M020" "M0-1" (Join-Path $m01Roi "fusion_v2_deep_unsupervised\previews") (Join-Path $m01Qa "fusion_v2_deep_unsupervised\previews") "Deep-unsupervised previews"
    New-MappingRow "M021" "M0-1" (Join-Path $m01Roi "fusion_v2_fast_preview\manifest.json") (Join-Path $m01Qa "fusion_v2_fast_preview\manifest.json") "Fast-preview run manifest without large generated cube"
    New-MappingRow "M022" "M0-1" (Join-Path $m01Roi "fusion_v2_fast_preview\metadata") (Join-Path $m01Qa "fusion_v2_fast_preview\metadata") "Fast-preview configuration and registration metadata"
    New-MappingRow "M023" "M0-1" (Join-Path $m01Roi "fusion_v2_fast_preview\metrics") (Join-Path $m01Qa "fusion_v2_fast_preview\metrics") "Fast-preview quality metrics"
    New-MappingRow "M024" "M0-1" (Join-Path $m01Roi "fusion_v2_fast_preview\previews") (Join-Path $m01Qa "fusion_v2_fast_preview\previews") "Fast-preview previews"
    New-MappingRow "M025" "M0-1" (Join-Path $m01Roi "fusion_v2_upsample_only\manifest.json") (Join-Path $m01Qa "fusion_v2_upsample_only\manifest.json") "Upsample-only run manifest without large generated cube"
    New-MappingRow "M026" "M0-1" (Join-Path $m01Roi "fusion_v2_upsample_only\metadata") (Join-Path $m01Qa "fusion_v2_upsample_only\metadata") "Upsample-only configuration and registration metadata"
    New-MappingRow "M027" "M0-1" (Join-Path $m01Roi "fusion_v2_upsample_only\metrics") (Join-Path $m01Qa "fusion_v2_upsample_only\metrics") "Upsample-only quality metrics"
    New-MappingRow "M028" "M0-1" (Join-Path $m01Roi "fusion_v2_upsample_only\previews") (Join-Path $m01Qa "fusion_v2_upsample_only\previews") "Upsample-only previews"

    New-MappingRow "M029" "M1-1" (Join-Path $legacyM11 "RGB-20230909_141858-00000.dat") (Join-Path $m11Module "assets\legacy_rgb_20230909\RGB-20230909_141858-00000.dat") "Unique full-resolution ENVI regression sample"
    New-MappingRow "M030" "M1-1" (Join-Path $legacyM11 "RGB-20230909_141858-00000.hdr") (Join-Path $m11Module "assets\legacy_rgb_20230909\RGB-20230909_141858-00000.hdr") "Header for the ENVI regression sample"
    New-MappingRow "M031" "M1-1" (Join-Path $legacyM11 "outputs") (Join-Path $m11Module "qa\legacy_outputs_20260714") "Rotation-correction reference outputs and QA"

    New-MappingRow "M032" "M1-2" (Join-Path $legacyM12 "models") (Join-Path $m12Module "models") "V1 and V2 model packages, weights, cards, manifests, and training history"
    New-MappingRow "M033" "M1-2" (Join-Path $legacyM12 "datasets") (Join-Path $m12Module "datasets") "Annotated masks, SHP provenance, splits, and corrected-box training candidates"
    New-MappingRow "M034" "M1-2" (Join-Path $legacyM12 "outputs") (Join-Path $m12Module "qa\legacy_outputs_20260714") "Model validation outputs and batch summary"
    New-MappingRow "M035" "M1-2" (Join-Path $legacyM12 "api_runtime\outputs") (Join-Path $m12Module "qa\legacy_api_runtime_outputs_20260714") "API edit and standalone smoke outputs"
    New-MappingRow "M036" "M1-2" (Join-Path $legacyM12 "examples\input") (Join-Path $m12Module "examples\input") "Small standalone inference sample"
    New-MappingRow "M037" "M1-2" (Join-Path $legacyM12 "geocore_mask\models\__init__.py") (Join-Path $m12Module "geocore_mask\models\__init__.py") "Missing model package initializer source"
    New-MappingRow "M038" "M1-2" (Join-Path $legacyM12 "geocore_mask\models\core_unet.py") (Join-Path $m12Module "geocore_mask\models\core_unet.py") "Missing U-Net architecture source required for retraining"
    New-MappingRow "M039" "M1-2" (Join-Path $legacyM12 "geocore_mask\models\registry.py") (Join-Path $m12Module "geocore_mask\models\registry.py") "Missing model registry source"

    New-MappingRow "M040" "M1-3" (Join-Path $legacyM13 "outputs") (Join-Path $m13Module "qa\legacy_outputs_20260714") "Separator/depth-mapping smoke outputs and review evidence"
)

New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null

$duplicateSources = @($mappings | Group-Object Source | Where-Object Count -gt 1)
$duplicateTargets = @($mappings | Group-Object Target | Where-Object Count -gt 1)
$missingSources = @($mappings | Where-Object { -not (Test-Path -LiteralPath $_.Source) })
$targetConflicts = @($mappings | Where-Object { Test-Path -LiteralPath $_.Target })
if ($duplicateSources.Count -or $duplicateTargets.Count -or $missingSources.Count -or $targetConflicts.Count) {
    throw "Move plan rejected: duplicate source=$($duplicateSources.Count), duplicate target=$($duplicateTargets.Count), missing source=$($missingSources.Count), target conflict=$($targetConflicts.Count)"
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
    throw "Move plan rejected: overlapping sources=$(@($overlaps).Count)"
}

foreach ($mapping in $mappings) {
    $targetFull = [System.IO.Path]::GetFullPath($mapping.Target)
    if (-not $targetFull.StartsWith([System.IO.Path]::GetFullPath($workspace).TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes consolidated workspace: $targetFull"
    }
}

$mappings | Export-Csv -LiteralPath (Join-Path $auditRoot "MOVE_PLAN.csv") -NoTypeInformation -Encoding UTF8

$preInventory = foreach ($mapping in $mappings) {
    $sourceItem = Get-Item -LiteralPath $mapping.Source -Force
    $files = if ($sourceItem.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $sourceItem.FullName -Force -File -Recurse)
    } else {
        @($sourceItem)
    }
    foreach ($file in $files) {
        $relative = if ($sourceItem.PSIsContainer) { $file.FullName.Substring($sourceItem.FullName.TrimEnd('\').Length + 1) } else { $file.Name }
        $targetFile = if ($sourceItem.PSIsContainer) { Join-Path $mapping.Target $relative } else { $mapping.Target }
        [pscustomobject]@{
            MappingId = $mapping.MappingId
            Module = $mapping.Module
            SourceFile = $file.FullName
            TargetFile = $targetFile
            RelativeWithinMapping = $relative
            SizeBytes = [int64]$file.Length
            LastWriteTimeUtc = $file.LastWriteTimeUtc.ToString("o")
            PreMoveSHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
}
$preInventory | Export-Csv -LiteralPath (Join-Path $auditRoot "PREMOVE_FILE_INVENTORY.csv") -NoTypeInformation -Encoding UTF8

foreach ($mapping in $mappings) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $mapping.Target) -Force | Out-Null
    Move-Item -LiteralPath $mapping.Source -Destination $mapping.Target
}

$postInventory = foreach ($row in $preInventory) {
    $exists = Test-Path -LiteralPath $row.TargetFile -PathType Leaf
    $postHash = if ($exists) { (Get-FileHash -LiteralPath $row.TargetFile -Algorithm SHA256).Hash } else { $null }
    [pscustomobject]@{
        MappingId = $row.MappingId
        Module = $row.Module
        OriginalSourceFile = $row.SourceFile
        CurrentTargetFile = $row.TargetFile
        SizeBytes = $row.SizeBytes
        SHA256 = $postHash
        PreMoveSHA256 = $row.PreMoveSHA256
        TargetExists = $exists
        HashMatches = ($exists -and $postHash -eq $row.PreMoveSHA256)
    }
}
$postInventory | Export-Csv -LiteralPath (Join-Path $auditRoot "MIGRATED_FILE_MANIFEST_SHA256.csv") -NoTypeInformation -Encoding UTF8

$moveSummary = foreach ($mapping in $mappings) {
    $rows = @($postInventory | Where-Object MappingId -eq $mapping.MappingId)
    [pscustomobject]@{
        MappingId = $mapping.MappingId
        Module = $mapping.Module
        Source = $mapping.Source
        Target = $mapping.Target
        FileCount = $rows.Count
        Bytes = [int64](($rows | Measure-Object SizeBytes -Sum).Sum)
        SourceAbsent = (-not (Test-Path -LiteralPath $mapping.Source))
        TargetPresent = (Test-Path -LiteralPath $mapping.Target)
        HashMismatchCount = @($rows | Where-Object { -not $_.HashMatches }).Count
        Rationale = $mapping.Rationale
    }
}
$moveSummary | Export-Csv -LiteralPath (Join-Path $auditRoot "MOVE_SUMMARY.csv") -NoTypeInformation -Encoding UTF8

$legacyRoots = @(
    [pscustomobject]@{ Module = "M0-1"; Root = $legacyM01 },
    [pscustomobject]@{ Module = "M1-1"; Root = $legacyM11 },
    [pscustomobject]@{ Module = "M1-2"; Root = $legacyM12 },
    [pscustomobject]@{ Module = "M1-3"; Root = $legacyM13 }
)

$remainder = foreach ($legacy in $legacyRoots) {
    foreach ($file in @(Get-ChildItem -LiteralPath $legacy.Root -Force -File -Recurse)) {
        $relative = $file.FullName.Substring($legacy.Root.TrimEnd('\').Length + 1)
        $category = if ($legacy.Module -eq "M0-1" -and $relative -match 'fused_cube\.zarr') {
            "REGENERABLE_FUSION_CUBE"
        } elseif ($legacy.Module -eq "M0-1" -and $relative.StartsWith("test_outputs\", [System.StringComparison]::OrdinalIgnoreCase)) {
            "GENERATED_TEST_OUTPUT"
        } elseif ($file.Extension -eq ".pyc" -or $relative -match '(^|\\)__pycache__(\\|$)') {
            "PYTHON_BYTECODE_CACHE"
        } else {
            "UNEXPECTED_REMAINDER"
        }
        [pscustomobject]@{
            Module = $legacy.Module
            LegacyRoot = $legacy.Root
            RelativePath = $relative
            FullPath = $file.FullName
            Category = $category
            SizeBytes = [int64]$file.Length
            SHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        }
    }
}
$remainder | Export-Csv -LiteralPath (Join-Path $auditRoot "LEGACY_REMAINDER_MANIFEST_SHA256.csv") -NoTypeInformation -Encoding UTF8

$deleteReadiness = foreach ($legacy in $legacyRoots) {
    $rows = @($remainder | Where-Object Module -eq $legacy.Module)
    $unexpected = @($rows | Where-Object Category -eq "UNEXPECTED_REMAINDER")
    [pscustomobject]@{
        Module = $legacy.Module
        LegacyFolder = $legacy.Root
        RemainingFileCount = $rows.Count
        RemainingBytes = [int64](($rows | Measure-Object SizeBytes -Sum).Sum)
        RegenerableFusionCubeFiles = @($rows | Where-Object Category -eq "REGENERABLE_FUSION_CUBE").Count
        GeneratedTestOutputFiles = @($rows | Where-Object Category -eq "GENERATED_TEST_OUTPUT").Count
        PythonCacheFiles = @($rows | Where-Object Category -eq "PYTHON_BYTECODE_CACHE").Count
        UnexpectedRemainderFiles = $unexpected.Count
        Status = if ($unexpected.Count -eq 0) { "READY_FOR_MANUAL_DELETE" } else { "REVIEW_REQUIRED" }
    }
}
$deleteReadiness | Export-Csv -LiteralPath (Join-Path $auditRoot "OLD_FOLDER_DELETE_READINESS.csv") -NoTypeInformation -Encoding UTF8

$hashMismatchCount = @($postInventory | Where-Object { -not $_.HashMatches }).Count
$sourceStillExistsCount = @($mappings | Where-Object { Test-Path -LiteralPath $_.Source }).Count
$missingTargetCount = @($mappings | Where-Object { -not (Test-Path -LiteralPath $_.Target) }).Count
$unexpectedRemainderCount = @($remainder | Where-Object Category -eq "UNEXPECTED_REMAINDER").Count
$readyFolderCount = @($deleteReadiness | Where-Object Status -eq "READY_FOR_MANUAL_DELETE").Count
$status = if ($hashMismatchCount -eq 0 -and $sourceStillExistsCount -eq 0 -and $missingTargetCount -eq 0 -and $unexpectedRemainderCount -eq 0 -and $readyFolderCount -eq 4) { "PASS" } else { "FAIL" }

$auditStatus = [ordered]@{
    SchemaVersion = "1.0"
    AuditStatus = $status
    CompletedAt = (Get-Date).ToString("o")
    ConsolidatedWorkspace = $workspace
    MoveMappingCount = $mappings.Count
    MigratedFileCount = $postInventory.Count
    MigratedBytes = [int64](($postInventory | Measure-Object SizeBytes -Sum).Sum)
    HashMismatchCount = $hashMismatchCount
    SourceStillExistsCount = $sourceStillExistsCount
    MissingMappingTargetCount = $missingTargetCount
    LegacyRemainderFileCount = $remainder.Count
    LegacyRemainderBytes = [int64](($remainder | Measure-Object SizeBytes -Sum).Sum)
    UnexpectedRemainderCount = $unexpectedRemainderCount
    ReadyForManualDeleteFolderCount = $readyFolderCount
    DirectDeletionCount = 0
    Boundary = "Maintainable source, unique models, training assets, compact QA, and regression samples were migrated. Large Zarr fusion cubes, generated test outputs, and bytecode caches remain only in legacy folders and are documented as regenerable/redundant."
}
$auditStatus | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $auditRoot "AUDIT_STATUS.json") -Encoding UTF8

if ($status -ne "PASS") {
    throw "Migration verification failed; inspect audit files before editing paths or deleting any legacy folder."
}

$auditStatus | ConvertTo-Json -Depth 5
