$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$auditRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$planPath = Join-Path $auditRoot "MOVE_PLAN.csv"
$manifestPath = Join-Path $auditRoot "MOVED_FILE_MANIFEST_SHA256.csv"

if (-not (Test-Path -LiteralPath $planPath -PathType Leaf)) {
    throw "Missing move plan: $planPath"
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Missing moved-file manifest: $manifestPath"
}

$plan = @(Import-Csv -LiteralPath $planPath)
$manifest = @(Import-Csv -LiteralPath $manifestPath)
$sourceConflicts = @($plan | Where-Object { Test-Path -LiteralPath $_.Source })
$missingTargets = @($plan | Where-Object { -not (Test-Path -LiteralPath $_.Target) })
if ($sourceConflicts.Count -or $missingTargets.Count) {
    throw "Restore precheck failed: source conflicts=$($sourceConflicts.Count), missing targets=$($missingTargets.Count)"
}

$hashFailures = foreach ($row in $manifest) {
    if (-not (Test-Path -LiteralPath $row.CurrentReviewFile -PathType Leaf)) {
        [pscustomobject]@{ Path = $row.CurrentReviewFile; Reason = "missing" }
        continue
    }
    $actual = (Get-FileHash -LiteralPath $row.CurrentReviewFile -Algorithm SHA256).Hash
    if ($actual -ne $row.CurrentSHA256) {
        [pscustomobject]@{ Path = $row.CurrentReviewFile; Reason = "sha256_mismatch" }
    }
}
if (@($hashFailures).Count) {
    $hashFailures | Format-Table -AutoSize
    throw "Restore rejected: $(@($hashFailures).Count) review files failed SHA-256 verification"
}

foreach ($row in $plan) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $row.Source) -Force | Out-Null
    Move-Item -LiteralPath $row.Target -Destination $row.Source
}

$missingRestoredSources = @($plan | Where-Object { -not (Test-Path -LiteralPath $_.Source) })
$targetsStillPresent = @($plan | Where-Object { Test-Path -LiteralPath $_.Target })
if ($missingRestoredSources.Count -or $targetsStillPresent.Count) {
    throw "Restore verification failed: missing restored sources=$($missingRestoredSources.Count), targets still present=$($targetsStillPresent.Count)"
}

[pscustomobject]@{
    RestoreStatus = "PASS"
    RestoredMappingCount = $plan.Count
    RestoredFileCount = $manifest.Count
    CompletedAt = (Get-Date).ToString("o")
    DirectDeletionCount = 0
} | ConvertTo-Json
