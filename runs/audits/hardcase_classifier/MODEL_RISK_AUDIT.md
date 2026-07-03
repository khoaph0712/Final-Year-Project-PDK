# Model Risk Audit

Static provenance (verified by code reading, see plan/session notes):

- **Model inbreeding: CLEAR** - all labels trace to human annotations (TACO COCO via `export_taco_yolo_hardcase.py`, Roboflow-source annotations via `archive/build_super_yolo_dataset.py`); no script converts model predictions into labels. Upstream Roboflow community annotation quality is unverifiable.
- **Split handling: CLEAR** - manifest-driven / source-preserving; no random re-splitting by default.
- **Filename-marker cross-split contamination (merged_v5): CLEAR** - 0 in both directions.

## Dataset: hardcase_classifier

- Cross-split duplicates: **516 exact**, **2775 near (Hamming<=4)**
- Undecodable images: **0**
  - EXACT test<->train: `baseline_test_d7382a518c.jpg` == `baseline_train_04be80318a.jpg`
  - EXACT test<->train: `baseline_test_5659b9efad.jpg` == `baseline_train_05cabec63c.jpg`
  - EXACT test<->train: `baseline_test_423808c9a2.jpg` == `baseline_train_199a135b7b.jpg`
  - EXACT test<->train: `baseline_test_4600458ccc.jpg` == `baseline_train_206ae33213.jpg`
  - EXACT test<->train: `baseline_test_6074f71007.jpg` == `baseline_train_2c9aefebbf.jpg`
  - EXACT test<->train: `baseline_test_4458d7b259.jpg` == `baseline_train_2de90dd033.jpg`
  - EXACT test<->train: `baseline_test_513a3f8cfb.jpg` == `baseline_train_356d4fbf9d.jpg`
  - EXACT test<->train: `baseline_test_2441e2c477.jpg` == `baseline_train_4060dc94a6.jpg`
  - EXACT test<->train: `baseline_test_7e7af0b301.jpg` == `baseline_train_4a23200484.jpg`
  - EXACT test<->train: `baseline_test_8cec07b870.jpg` == `baseline_train_583296e037.jpg`
  - NEAR ham=3 test<->train: `baseline_test_55804c6c0d.jpg` ~ `baseline_train_00fe5a3bce.jpg`
  - NEAR ham=4 test<->train: `baseline_test_5b6f82e9c4.jpg` ~ `baseline_train_00fe5a3bce.jpg`
  - NEAR ham=3 test<->train: `baseline_test_00578a3800.jpg` ~ `baseline_train_00fe5a3bce.jpg`
  - NEAR ham=4 test<->train: `realwaste_fded7f271d.jpg` ~ `baseline_train_00fe5a3bce.jpg`
  - NEAR ham=4 test<->train: `baseline_test_26f2c23ee5.jpg` ~ `baseline_train_00fe5a3bce.jpg`
  - NEAR ham=2 test<->train: `baseline_test_2351bee43c.jpg` ~ `baseline_train_053a20270c.jpg`
  - NEAR ham=3 test<->train: `baseline_test_bdd645325b.jpg` ~ `baseline_train_053a20270c.jpg`
  - NEAR ham=3 test<->train: `baseline_test_8be49be0fd.jpg` ~ `baseline_train_1265873d12.jpg`
  - NEAR ham=2 test<->train: `baseline_test_e117bcf74c.jpg` ~ `baseline_train_1265873d12.jpg`
  - NEAR ham=4 test<->train: `baseline_test_ce713c9216.jpg` ~ `baseline_train_15dfd7d5a8.jpg`

| split | class | total | field | studio |
|---|---|---:|---:|---:|
| test | Background | 300 | 300 | 0 |
| test | cardboard | 374 | 300 | 74 |
| test | glass | 356 | 300 | 56 |
| test | metal | 438 | 300 | 138 |
| test | organic | 427 | 300 | 127 |
| test | paper | 376 | 300 | 76 |
| test | plastic | 425 | 300 | 125 |
| train | Background | 1200 | 1200 | 0 |
| train | cardboard | 1517 | 1200 | 317 |
| train | glass | 1501 | 1200 | 301 |
| train | metal | 1736 | 1200 | 536 |
| train | organic | 1806 | 1200 | 606 |
| train | paper | 1548 | 1200 | 348 |
| train | plastic | 1855 | 1200 | 655 |
| val | Background | 300 | 300 | 0 |
| val | cardboard | 370 | 300 | 70 |
| val | glass | 363 | 300 | 63 |
| val | metal | 416 | 300 | 116 |
| val | organic | 414 | 300 | 114 |
| val | paper | 376 | 300 | 76 |
| val | plastic | 441 | 300 | 141 |

## Overfitting signals (yolo26n_hardcase_v2_long, may be mid-training)

- Epochs done: 100; best mAP50-95 0.5364 at epoch 100
- Last epoch: P 0.791 R 0.630 mAP50 0.707
- val-train cls-loss gap: 0.368 (mild gap normal; watch for val loss rising while train falls)
