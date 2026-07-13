# Stage 2 Localization Threshold Sweep, 300 Images

This comparison uses the revised DL workflow: Stage 1 classification/gating first, then Stage 2 YOLO localization only. YOLO is used for box localization, not as the final class decision.

| Setting | Images | GT boxes | Pred boxes | TP | FP | FN | Precision | Recall | F1 | Mean IoU | Gate hit-rate | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO conf=0.30 | 300 | 1152 | 943 | 660 | 283 | 492 | 0.6999 | 0.5729 | 0.6301 | 0.9057 | 0.8533 | Promoted final balanced setting. |
| YOLO conf=0.35 | 300 | 1152 | 815 | 617 | 198 | 535 | 0.7571 | 0.5356 | 0.6274 | 0.9043 | 0.8533 | Balanced precision setting. |
| YOLO conf=0.40 | 300 | 1152 | 738 | 593 | 145 | 559 | 0.8035 | 0.5148 | 0.6275 | 0.9050 | 0.8533 | Highest precision setting. |

## Recommendation

Use `--localizer yolo --yolo-conf 0.30` for the final balanced Stage 2 localization report. It gives the best F1 and recall among the 300-image threshold checks while keeping mean matched IoU above 0.90.

Use `--yolo-conf 0.40` only when the discussion needs to show a high-precision trade-off.

## Artifact Paths

- Final balanced run: `runs/dl/localization_rework/yolo_conf030_stratified300_final/REPORT.md`
- Balanced precision sweep: `runs/dl/localization_rework/yolo_conf035_stratified300_final/REPORT.md`
- High precision sweep: `runs/dl/localization_rework/yolo_conf040_stratified300_sweep/REPORT.md`
