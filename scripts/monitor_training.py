#!/usr/bin/env python
"""
WasteWise Training Monitor and Auto-Restart Daemon
Resumes YOLO26s training from the last checkpoint, monitors progress, auto-restarts on failure,
and reports detailed metrics upon completion.
"""

import sys
import os
import subprocess
import time
import csv
from pathlib import Path

# Reconfigure stdout/stderr to use UTF-8 and handle encoding errors gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = ROOT / ".venv311" / "Scripts" / "python.exe"
TRAIN_SCRIPT = ROOT / "scripts" / "train_yolo26_hardcase.py"
RUN_DIR = ROOT / "runs" / "detect" / "yolo26s_hardcase_v1"
RESULTS_CSV = RUN_DIR / "results.csv"
MONITOR_LOG = RUN_DIR / "monitor_run.log"
SUMMARY_MD = RUN_DIR / "TRAINING_SUMMARY.md"
LAST_CHECKPOINT = RUN_DIR / "weights" / "last.pt"

# Command configuration
COMMAND = [
    str(PYTHON_EXE),
    str(TRAIN_SCRIPT),
    "--resume",
    "--name", "yolo26s_hardcase_v1",
    "--model", "runs/detect/yolo26s_hardcase_v1/weights/last.pt",
    "--epochs", "100",
    "--batch", "16",
    "--workers", "4"
]

def log_message(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    sys.stdout.flush()
    
    # Append to log file
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(MONITOR_LOG, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def get_completed_epochs() -> int:
    if not RESULTS_CSV.exists():
        return 0
    try:
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return 0
            # Get the epoch from the last valid row
            last_row = rows[-1]
            # Strip whitespace from keys
            cleaned_row = {k.strip(): v.strip() for k, v in last_row.items() if k is not None}
            epoch_str = cleaned_row.get("epoch", "0")
            return int(epoch_str)
    except Exception as e:
        log_message(f"Warning: Failed to read completed epochs from results.csv: {e}")
        return 0

def compile_metrics():
    log_message("Compiling final training metrics...")
    if not RESULTS_CSV.exists():
        log_message("Error: results.csv not found, cannot compile metrics.")
        return
        
    try:
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            # Ultralytics results.csv headers contain whitespace
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader)]
            rows = []
            for r in reader:
                if not r or len(r) < len(headers):
                    continue
                rows.append(dict(zip(headers, [val.strip() for val in r])))
                
        if not rows:
            log_message("Error: results.csv is empty.")
            return

        # Find best epoch based on metrics/mAP50(B)
        best_row = None
        best_map50 = -1.0
        
        for r in rows:
            try:
                map50 = float(r.get("metrics/mAP50(B)", 0.0))
                if map50 > best_map50:
                    best_map50 = map50
                    best_row = r
            except ValueError:
                continue
                
        last_row = rows[-1]
        
        # Format a markdown table of epoch history
        history_rows = []
        for r in rows[-5:]:  # Last 5 epochs
            history_rows.append(
                f"| {r.get('epoch', '-')} "
                f"| {float(r.get('train/box_loss', 0.0)):.4f} "
                f"| {float(r.get('train/cls_loss', 0.0)):.4f} "
                f"| {float(r.get('metrics/precision(B)', 0.0)):.4f} "
                f"| {float(r.get('metrics/recall(B)', 0.0)):.4f} "
                f"| {float(r.get('metrics/mAP50(B)', 0.0)):.4f} "
                f"| {float(r.get('metrics/mAP50-95(B)', 0.0)):.4f} |"
            )
        history_joined = "\n".join(history_rows)
        log_rel_path = str(MONITOR_LOG.relative_to(ROOT)).replace("\\", "/")
        summary_content = f"""# YOLO26s Training Summary Report

Completed training YOLO26s on the hard-case dataset.

## Summary Table

| Metric | Best Epoch ({best_row.get('epoch', 'N/A')}) | Final Epoch ({last_row.get('epoch', 'N/A')}) |
| :--- | :---: | :---: |
| **Precision** | {float(best_row.get('metrics/precision(B)', 0.0)):.4f} | {float(last_row.get('metrics/precision(B)', 0.0)):.4f} |
| **Recall** | {float(best_row.get('metrics/recall(B)', 0.0)):.4f} | {float(last_row.get('metrics/recall(B)', 0.0)):.4f} |
| **mAP@0.5** | **{float(best_row.get('metrics/mAP50(B)', 0.0)):.4f}** | {float(last_row.get('metrics/mAP50(B)', 0.0)):.4f} |
| **mAP@0.5:0.95** | {float(best_row.get('metrics/mAP50-95(B)', 0.0)):.4f} | {float(last_row.get('metrics/mAP50-95(B)', 0.0)):.4f} |
| **Train Box Loss** | {float(best_row.get('train/box_loss', 0.0)):.4f} | {float(last_row.get('train/box_loss', 0.0)):.4f} |
| **Train Class Loss** | {float(best_row.get('train/cls_loss', 0.0)):.4f} | {float(last_row.get('train/cls_loss', 0.0)):.4f} |
| **Val Box Loss** | {float(best_row.get('val/box_loss', 0.0)):.4f} | {float(last_row.get('val/box_loss', 0.0)):.4f} |
| **Val Class Loss** | {float(best_row.get('val/cls_loss', 0.0)):.4f} | {float(last_row.get('val/cls_loss', 0.0)):.4f} |

## Recent Training History (Last 5 Epochs)

| Epoch | Train Box Loss | Train Cls Loss | Precision | Recall | mAP50 | mAP50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{history_joined}

*Log outputs saved to `{log_rel_path}`*
"""
        with open(SUMMARY_MD, "w", encoding="utf-8") as f:
            f.write(summary_content)
        log_message(f"Summary written to {SUMMARY_MD}")
        print("\n" + summary_content)
        sys.stdout.flush()
    except Exception as e:
        log_message(f"Error compiling metrics: {e}")

def main():
    log_message("Starting WasteWise Training Monitor...")
    log_message(f"Running command: {' '.join(COMMAND)}")
    
    total_restarts = 0
    max_restarts = 10
    no_progress_count = 0
    target_epochs = 100
    
    while True:
        current_epoch = get_completed_epochs()
        log_message(f"Current training progress: {current_epoch}/{target_epochs} epochs completed.")
        
        if current_epoch >= target_epochs:
            log_message("Target epochs reached! Training is complete.")
            compile_metrics()
            break
            
        log_message(f"Launching training process (Restart #{total_restarts})...")
        start_time = time.time()
        
        # Launch subprocess
        process = subprocess.Popen(
            COMMAND,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(ROOT)
        )
        
        # Stream stdout/stderr in real-time
        for line in process.stdout:
            # Print to stdout and log file
            print(line, end="")
            sys.stdout.flush()
            with open(MONITOR_LOG, "a", encoding="utf-8") as f:
                f.write(line)
                
        process.wait()
        exit_code = process.returncode
        elapsed = time.time() - start_time
        
        log_message(f"Process terminated with exit code: {exit_code} (Duration: {elapsed:.2f} seconds)")
        
        # Check progress
        new_epoch = get_completed_epochs()
        if new_epoch > current_epoch:
            log_message(f"Progress made: Epochs completed increased from {current_epoch} to {new_epoch}.")
            no_progress_count = 0
        else:
            no_progress_count += 1
            log_message(f"No epoch progress made during this run. No-progress run count: {no_progress_count}/3.")
            
        if exit_code == 0:
            log_message("Training process finished successfully.")
            compile_metrics()
            break
        else:
            # Process crashed
            if no_progress_count >= 3:
                log_message("CRITICAL: Training process crashed 3 times in a row without making any epoch progress. Stopping monitor.")
                sys.exit(1)
                
            total_restarts += 1
            if total_restarts > max_restarts:
                log_message(f"CRITICAL: Maximum restart limit ({max_restarts}) reached. Stopping monitor.")
                sys.exit(1)
                
            log_message("Training process failed. Waiting 10 seconds before restarting...")
            time.sleep(10)

if __name__ == "__main__":
    main()
