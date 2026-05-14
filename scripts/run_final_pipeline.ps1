param(
    [string]$Python = ".\.venv311\Scripts\python.exe",
    [string]$Data = "merged_dataset_v3\data.yaml",
    [string]$DatasetRoot = "merged_dataset_v3",
    [string]$MlOut = "runs\ml\feature_ml_enhanced_6class_4k",
    [string]$ComparisonOut = "runs\comparisons\model_comparison",
    [string]$YoloWeights = "runs\dl\trash_yolov8n_v3\weights\best.pt",
    [string]$ManualSource = "",
    [switch]$SkipFeatureMl,
    [switch]$SkipTinyCnn,
    [switch]$SkipYoloEval,
    [switch]$SkipExport,
    [switch]$SkipManualPredict
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Command
}

Run-Step "Class balance" {
    & $Python scripts\check_class_balance.py $DatasetRoot
}

if (-not $SkipFeatureMl) {
    Run-Step "Enhanced feature extraction + classical ML" {
        & $Python scripts\feature_ml_analysis.py `
            --data $Data `
            --out $MlOut `
            --exclude-classes other `
            --max-per-class-train 4000 `
            --max-per-class-test 800
    }
}

if (-not $SkipTinyCnn) {
    Run-Step "Tiny CNN baseline" {
        & $Python scripts\deep_learning_baseline.py --data $Data
    }
}

Run-Step "ML-vs-DL comparison" {
    & $Python scripts\compare_ml_dl.py `
        --ml-metrics (Join-Path $MlOut "metrics_summary.json") `
        --out $ComparisonOut
}

if (-not $SkipYoloEval) {
    Run-Step "YOLO quality check" {
        & $Python scripts\evaluate.py `
            --weights $YoloWeights `
            --data $Data `
            --split both
    }
}

if (-not $SkipExport) {
    Run-Step "Export YOLO model" {
        & $Python scripts\export_model.py --imgsz 640
    }
}

if (-not $SkipManualPredict) {
    if ([string]::IsNullOrWhiteSpace($ManualSource)) {
        Write-Host ""
        Write-Host "=== Manual prediction skipped ===" -ForegroundColor Yellow
        Write-Host "Pass -ManualSource C:\path\to\test_images to run no-UI predictions."
    } else {
        Run-Step "Manual no-UI prediction" {
            & $Python scripts\predict_images.py --source $ManualSource --conf 0.10
        }
    }
}

Write-Host ""
Write-Host "[OK] Final project pipeline complete." -ForegroundColor Green
