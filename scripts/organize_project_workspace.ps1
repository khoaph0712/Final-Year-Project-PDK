param(
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ArchiveRel = "_archive\legacy_cleanup_2026_05_30"

function Get-FullPathInRoot {
    param([Parameter(Mandatory=$true)][string]$RelativePath)

    $fullPath = [System.IO.Path]::GetFullPath((Join-Path $Root $RelativePath))
    $rootWithSlash = $Root.TrimEnd("\") + "\"
    if (($fullPath -ne $Root) -and (-not $fullPath.StartsWith($rootWithSlash, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "Path escapes workspace root: $RelativePath -> $fullPath"
    }
    return $fullPath
}

function Move-WorkspaceItem {
    param(
        [Parameter(Mandatory=$true)][string]$SourceRel,
        [Parameter(Mandatory=$true)][string]$TargetRel
    )

    $source = Get-FullPathInRoot $SourceRel
    $target = Get-FullPathInRoot $TargetRel

    if (-not (Test-Path -LiteralPath $source)) {
        Write-Host "SKIP missing: $SourceRel"
        return
    }
    if (Test-Path -LiteralPath $target) {
        Write-Host "SKIP target exists: $TargetRel"
        return
    }

    $targetParent = Split-Path -Parent $target
    if ($WhatIfOnly) {
        Write-Host "MOVE $SourceRel -> $TargetRel"
        return
    }

    New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
    Move-Item -LiteralPath $source -Destination $target -Force
    Write-Host "MOVED $SourceRel -> $TargetRel"
}

function Remove-WorkspaceItem {
    param(
        [Parameter(Mandatory=$true)][string]$RelativePath,
        [switch]$RequireEmpty
    )

    $target = Get-FullPathInRoot $RelativePath
    if (-not (Test-Path -LiteralPath $target)) {
        Write-Host "SKIP missing: $RelativePath"
        return
    }

    if ($RequireEmpty) {
        $children = @(Get-ChildItem -LiteralPath $target -Force -ErrorAction SilentlyContinue)
        if ($children.Count -gt 0) {
            Write-Host "SKIP not empty: $RelativePath"
            return
        }
    }

    if ($WhatIfOnly) {
        Write-Host "REMOVE $RelativePath"
        return
    }

    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Host "REMOVED $RelativePath"
}

function Get-WorkspaceFileSha256 {
    param([Parameter(Mandatory=$true)][string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            return [System.BitConverter]::ToString($sha.ComputeHash($stream)).Replace("-", "")
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Remove-WorkspaceFileIfDuplicate {
    param(
        [Parameter(Mandatory=$true)][string]$SourceRel,
        [Parameter(Mandatory=$true)][string]$CanonicalRel
    )

    $source = Get-FullPathInRoot $SourceRel
    $canonical = Get-FullPathInRoot $CanonicalRel

    if (-not (Test-Path -LiteralPath $source)) {
        Write-Host "SKIP missing: $SourceRel"
        return
    }
    if (-not (Test-Path -LiteralPath $canonical)) {
        Write-Host "SKIP canonical missing: $CanonicalRel"
        return
    }

    $sourceHash = Get-WorkspaceFileSha256 -Path $source
    $canonicalHash = Get-WorkspaceFileSha256 -Path $canonical
    if ($sourceHash -ne $canonicalHash) {
        Write-Host "SKIP not duplicate: $SourceRel differs from $CanonicalRel"
        return
    }

    if ($WhatIfOnly) {
        Write-Host "REMOVE duplicate $SourceRel (canonical: $CanonicalRel)"
        return
    }

    Remove-Item -LiteralPath $source -Force
    Write-Host "REMOVED duplicate $SourceRel"
}

$archiveMoves = @(
    @{ Source = "scratch"; Target = "$ArchiveRel\scratch" },
    @{ Source = "assets\internet_test_images"; Target = "$ArchiveRel\assets\internet_test_images_raw" },
    @{ Source = "assets\internet_test_images_rejected"; Target = "$ArchiveRel\assets\internet_test_images_rejected" },
    @{ Source = "runs\detect\demo_beach_and_grass"; Target = "$ArchiveRel\runs\detect\demo_beach_and_grass" },
    @{ Source = "runs\manual_tests"; Target = "$ArchiveRel\runs\manual_tests" },
    @{ Source = "runs\dl\classification_to_localization"; Target = "$ArchiveRel\runs\dl\stage2_localization_first_trial" },
    @{ Source = "runs\dl\classification_to_localization_yolo_smoke"; Target = "$ArchiveRel\runs\dl\stage2_localization_yolo_smoke" },
    @{ Source = "runs\dl\classification_to_localization_yolo_stratified60"; Target = "$ArchiveRel\runs\dl\stage2_localization_yolo_conf025_early" }
)

$clearerMoves = @(
    @{ Source = "runs\dl\classification_to_localization_stratified60"; Target = "runs\dl\localization_rework\gradcam_baseline_stratified60" },
    @{ Source = "runs\dl\classification_to_localization_yolo_stratified60_improved"; Target = "runs\dl\localization_rework\yolo_conf025_stratified60" },
    @{ Source = "runs\dl\classification_to_localization_yolo_stratified60_conf035"; Target = "runs\dl\localization_rework\yolo_conf035_stratified60_final" }
)

foreach ($move in $archiveMoves) {
    Move-WorkspaceItem -SourceRel $move.Source -TargetRel $move.Target
}

foreach ($move in $clearerMoves) {
    Move-WorkspaceItem -SourceRel $move.Source -TargetRel $move.Target
}

Remove-WorkspaceItem -RelativePath ".antigravitycli" -RequireEmpty
Remove-WorkspaceItem -RelativePath "data\external_datasets" -RequireEmpty
Remove-WorkspaceItem -RelativePath "docs\01_final_report\rendered_tracking_report" -RequireEmpty
Remove-WorkspaceItem -RelativePath "data\convnext_training_crops"
Remove-WorkspaceItem -RelativePath "scripts\__pycache__"
Remove-WorkspaceItem -RelativePath "scripts\archive\__pycache__"
Remove-WorkspaceItem -RelativePath "web\__pycache__"
Remove-WorkspaceItem -RelativePath "docs\figma_pipelines" -RequireEmpty
Remove-WorkspaceItem -RelativePath "runs\detect\runs"
Remove-WorkspaceItem -RelativePath "runs\dl\convnextv2_material_stage1_smoke"
Remove-WorkspaceItem -RelativePath "runs\dl\convnextv2_material_stage1_candidate" -RequireEmpty
Remove-WorkspaceItem -RelativePath "external_datasets\yolo26_balanced_realworld_v1"
Remove-WorkspaceItem -RelativePath "open-design"

Move-WorkspaceItem -SourceRel "yolo26n.pt" -TargetRel "models\pretrained\yolo26n.pt"
Remove-WorkspaceFileIfDuplicate -SourceRel "yolo26n.pt" -CanonicalRel "models\pretrained\yolo26n.pt"
Remove-WorkspaceFileIfDuplicate -SourceRel "yolo11n.pt" -CanonicalRel "models\pretrained\yolo11n.pt"

Write-Host "Workspace organization pass complete."
