$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$auditRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$migrationManifestPath = Join-Path $auditRoot "MIGRATED_FILE_MANIFEST_SHA256.csv"
$movePlanPath = Join-Path $auditRoot "MOVE_PLAN.csv"
$postManifestPath = Join-Path $auditRoot "POST_NORMALIZATION_FILE_MANIFEST_SHA256.csv"
$verificationPath = Join-Path $auditRoot "POST_NORMALIZATION_VERIFICATION.json"

if (-not (Test-Path -LiteralPath $migrationManifestPath -PathType Leaf)) {
    throw "Missing migration manifest: $migrationManifestPath"
}
if (-not (Test-Path -LiteralPath $movePlanPath -PathType Leaf)) {
    throw "Missing move plan: $movePlanPath"
}

$expectedNormalizedMappings = @("M001", "M002", "M003")
$migrationRows = @(Import-Csv -LiteralPath $migrationManifestPath)
$movePlan = @(Import-Csv -LiteralPath $movePlanPath)

$postRows = foreach ($row in $migrationRows) {
    $exists = Test-Path -LiteralPath $row.CurrentTargetFile -PathType Leaf
    $item = if ($exists) { Get-Item -LiteralPath $row.CurrentTargetFile -Force } else { $null }
    $currentHash = if ($exists) {
        (Get-FileHash -LiteralPath $row.CurrentTargetFile -Algorithm SHA256).Hash
    } else {
        $null
    }
    $matchesMigrationHash = $exists -and $currentHash -eq $row.SHA256
    $classification = if (-not $exists) {
        "MISSING"
    } elseif ($matchesMigrationHash) {
        "UNCHANGED"
    } elseif ($expectedNormalizedMappings -contains $row.MappingId) {
        "INTENTIONAL_PATH_NORMALIZATION"
    } else {
        "UNEXPECTED_CHANGE"
    }

    [pscustomobject]@{
        MappingId = $row.MappingId
        Module = $row.Module
        OriginalSourceFile = $row.OriginalSourceFile
        CurrentTargetFile = $row.CurrentTargetFile
        MigratedSizeBytes = [int64]$row.SizeBytes
        CurrentSizeBytes = if ($exists) { [int64]$item.Length } else { $null }
        MigratedSHA256 = $row.SHA256
        CurrentSHA256 = $currentHash
        TargetExists = $exists
        MatchesMigrationHash = $matchesMigrationHash
        ChangeClassification = $classification
    }
}

$postRows | Export-Csv -LiteralPath $postManifestPath -NoTypeInformation -Encoding UTF8

$missingRows = @($postRows | Where-Object ChangeClassification -eq "MISSING")
$unchangedRows = @($postRows | Where-Object ChangeClassification -eq "UNCHANGED")
$intentionalRows = @($postRows | Where-Object ChangeClassification -eq "INTENTIONAL_PATH_NORMALIZATION")
$unexpectedRows = @($postRows | Where-Object ChangeClassification -eq "UNEXPECTED_CHANGE")
$missingExpectedMappings = @($expectedNormalizedMappings | Where-Object { $_ -notin $intentionalRows.MappingId })
$sourceStillExists = @($movePlan | Where-Object { Test-Path -LiteralPath $_.Source })
$missingMappingTargets = @($movePlan | Where-Object { -not (Test-Path -LiteralPath $_.Target) })

$normalizedScriptRows = @($postRows | Where-Object MappingId -in $expectedNormalizedMappings)
$legacyPathReferences = foreach ($row in $normalizedScriptRows) {
    if (-not $row.TargetExists) {
        continue
    }
    $matches = @(Select-String -LiteralPath $row.CurrentTargetFile -Pattern @(
        "Geocore_M0-1_image_fusion",
        "Geocore_M1-1_rotation_correction",
        "Geocore_M1-2_foreground_mask",
        "Geocore_M1-3_separator_mark"
    ) -SimpleMatch)
    foreach ($match in $matches) {
        [pscustomobject]@{
            Path = $row.CurrentTargetFile
            LineNumber = $match.LineNumber
            Line = $match.Line.Trim()
        }
    }
}

$status = if (
    $missingRows.Count -eq 0 -and
    $unexpectedRows.Count -eq 0 -and
    $intentionalRows.Count -eq $expectedNormalizedMappings.Count -and
    $missingExpectedMappings.Count -eq 0 -and
    $sourceStillExists.Count -eq 0 -and
    $missingMappingTargets.Count -eq 0 -and
    @($legacyPathReferences).Count -eq 0
) { "PASS" } else { "FAIL" }

$verification = [ordered]@{
    SchemaVersion = "1.0"
    VerificationStatus = $status
    CompletedAt = (Get-Date).ToString("o")
    MigratedFileCount = $postRows.Count
    CurrentFileCount = @($postRows | Where-Object TargetExists).Count
    CurrentBytes = [int64](($postRows | Measure-Object CurrentSizeBytes -Sum).Sum)
    UnchangedFileCount = $unchangedRows.Count
    IntentionalPathNormalizationCount = $intentionalRows.Count
    IntentionalPathNormalizationMappings = @($intentionalRows.MappingId)
    MissingFileCount = $missingRows.Count
    UnexpectedChangeCount = $unexpectedRows.Count
    MissingExpectedNormalizationCount = $missingExpectedMappings.Count
    SourceStillExistsCount = $sourceStillExists.Count
    MissingMappingTargetCount = $missingMappingTargets.Count
    LegacyPathReferenceCountInNormalizedScripts = @($legacyPathReferences).Count
    RestoreManifestReady = ($status -eq "PASS")
    DirectDeletionCount = 0
    Boundary = "Three migrated M0-1 scripts were intentionally normalized after their move. All other migrated files must remain byte-identical to the migration-time SHA-256 manifest."
}

$verification | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $verificationPath -Encoding UTF8
$verification | ConvertTo-Json -Depth 5

if ($status -ne "PASS") {
    throw "Post-normalization verification failed; inspect the generated manifest and JSON report."
}
