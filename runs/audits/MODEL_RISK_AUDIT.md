# Model Risk Audit

Static provenance (verified by code reading, see plan/session notes):

- **Model inbreeding: CLEAR** - all labels trace to human annotations (TACO COCO via `export_taco_yolo_hardcase.py`, Roboflow-source annotations via `archive/build_super_yolo_dataset.py`); no script converts model predictions into labels. Upstream Roboflow community annotation quality is unverifiable.
- **Split handling: CLEAR** - manifest-driven / source-preserving; no random re-splitting by default.
- **Filename-marker cross-split contamination (merged_v5): CLEAR** - 0 in both directions.

## Dataset: hardcase

- Cross-split duplicates: **1 exact**, **4808 near (Hamming<=4)**
- Undecodable images: **0**
  - EXACT train<->val: `super_yolo_273253d888.jpg` == `super_yolo_823c20fd37.jpg`
  - NEAR ham=4 test<->train: `super_yolo_f44800e091.jpg` ~ `super_yolo_001e2cd2ad.jpg`
  - NEAR ham=3 test<->train: `super_yolo_ff8556589d.jpg` ~ `super_yolo_00853c353e.jpg`
  - NEAR ham=4 test<->train: `super_yolo_c7afedef25.jpg` ~ `super_yolo_00853c353e.jpg`
  - NEAR ham=4 test<->train: `super_yolo_ea3998cd70.jpg` ~ `super_yolo_00a83f48cd.jpg`
  - NEAR ham=4 test<->train: `super_yolo_f1f73897be.jpg` ~ `super_yolo_00a83f48cd.jpg`
  - NEAR ham=0 test<->train: `taco_hardcase_bcad3d0832.jpg` ~ `super_yolo_00ebda7525.jpg`
  - NEAR ham=4 test<->train: `super_yolo_ff8556589d.jpg` ~ `super_yolo_016b691452.jpg`
  - NEAR ham=4 test<->train: `super_yolo_d06940fac3.jpg` ~ `super_yolo_0225668c83.jpg`
  - NEAR ham=2 test<->train: `super_yolo_ff8556589d.jpg` ~ `super_yolo_0225668c83.jpg`
  - NEAR ham=3 test<->train: `super_yolo_c7afedef25.jpg` ~ `super_yolo_0225668c83.jpg`

| split | class | boxes | median area | tiny <1% img |
|---|---|---:|---:|---:|
| test | plastic | 2052 | 3.33% | 27.6% |
| test | glass | 77 | 0.15% | 75.3% |
| test | metal | 603 | 7.43% | 19.2% |
| test | paper | 1424 | 17.0% | 15.8% |
| test | cardboard | 59 | 3.8% | 30.5% |
| test | organic | 50 | 1.22% | 46.0% |
| train | plastic | 16789 | 2.17% | 35.8% |
| train | glass | 7324 | 10.2% | 15.7% |
| train | metal | 8990 | 6.41% | 16.2% |
| train | paper | 4750 | 22.3% | 9.2% |
| train | cardboard | 7379 | 13.19% | 6.3% |
| train | organic | 33871 | 1.14% | 47.0% |
| val | plastic | 1735 | 1.54% | 41.1% |
| val | glass | 2583 | 8.62% | 15.7% |
| val | metal | 1840 | 3.71% | 27.2% |
| val | paper | 186 | 8.12% | 18.3% |
| val | cardboard | 1640 | 14.57% | 7.5% |
| val | organic | 13824 | 1.05% | 48.8% |

| split | label files | single-class img | noise |
|---|---:|---:|---|
| test | 1363 | 93.5% | {"polygon_rows": 10, "zero_area": 1} |
| train | 21199 | 71.6% | {"polygon_rows": 6099, "zero_area": 10} |
| val | 3538 | 76.4% | {"polygon_rows": 624, "zero_area": 6} |

## Dataset: super

- Cross-split duplicates: **1 exact**, **3880 near (Hamming<=4)**
- Undecodable images: **0**
  - EXACT train<->val: `taco_yolo_batch_8__000005.jpg` == `taco_yolo_batch_7__000000.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_metal1253_jpg.rf.c93299f221d8b9614552891bf14f23be.jpg` ~ `rf_garbage_biodegradable1265_jpg.rf.b178945837b89b2a18caae9098f0cb1a.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_metal139_jpg.rf.907bf662a10abd30090f560695897f2a.jpg` ~ `rf_garbage_biodegradable1701_jpg.rf.35d4b22043be99db700539cdf382a060.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_metal292_jpg.rf.7b17daa581144a26f4f0bead419e5b6c.jpg` ~ `rf_garbage_biodegradable2004_jpeg.rf.7a706209be0dc1acb3f213e0017caebd.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_paper2320_jpg.rf.bb094a2ce325519ccb563cf0fb82668c.jpg` ~ `rf_garbage_biodegradable2004_jpeg.rf.7a706209be0dc1acb3f213e0017caebd.jpg`
  - NEAR ham=3 test<->train: `rf_garbage_plastic617_jpg.rf.60ab6209a57bf60156baf8278bd553f2.jpg` ~ `rf_garbage_biodegradable2004_jpeg.rf.7a706209be0dc1acb3f213e0017caebd.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_plastic266_jpg.rf.b6800ab17258d9ae974d28b1651c20f9.jpg` ~ `rf_garbage_biodegradable2004_jpeg.rf.7a706209be0dc1acb3f213e0017caebd.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_metal299_jpg.rf.4a37905038be6d58838f86e25d66c260.jpg` ~ `rf_garbage_biodegradable654_jpg.rf.1799277a1c9a6c29d98f06af1dfa1f68.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_paper2144_jpg.rf.0f0824d2fd920fd3f3c937de74e7915d.jpg` ~ `rf_garbage_biodegradable855_jpg.rf.2c4c13e201316a286b0207664e2bd430.jpg`
  - NEAR ham=3 test<->train: `rf_garbage_paper1388_jpg.rf.f2c63be1196ad10511f512160ddf2d79.jpg` ~ `rf_garbage_cardboard1283_jpg.rf.c9981c47022e2b7408cc03b002718c24.jpg`
  - NEAR ham=4 test<->train: `rf_garbage_paper1528_jpg.rf.f3dcccbc810ed4d2e50ac40f95ad6648.jpg` ~ `rf_garbage_cardboard1283_jpg.rf.c9981c47022e2b7408cc03b002718c24.jpg`

| split | class | boxes | median area | tiny <1% img |
|---|---|---:|---:|---:|
| test | plastic | 1813 | 3.79% | 25.2% |
| test | glass | 15 | 0.39% | 73.3% |
| test | metal | 579 | 7.89% | 18.8% |
| test | paper | 1396 | 17.28% | 15.8% |
| test | cardboard | 48 | 7.95% | 31.2% |
| test | organic | 50 | 1.22% | 46.0% |
| train | plastic | 15361 | 2.6% | 32.9% |
| train | glass | 7136 | 10.52% | 14.1% |
| train | metal | 8696 | 6.78% | 14.8% |
| train | paper | 4604 | 24.48% | 7.8% |
| train | cardboard | 7174 | 13.76% | 5.5% |
| train | organic | 33863 | 1.14% | 47.0% |
| val | plastic | 1252 | 1.91% | 35.2% |
| val | glass | 2579 | 8.61% | 15.7% |
| val | metal | 1797 | 3.77% | 26.6% |
| val | paper | 160 | 10.04% | 13.8% |
| val | cardboard | 1613 | 14.7% | 7.0% |
| val | organic | 13824 | 1.05% | 48.8% |

| split | label files | single-class img | noise |
|---|---:|---:|---|
| test | 1192 | 95.0% | {"zero_area": 1, "polygon_rows": 10} |
| train | 20165 | 71.4% | {"zero_area": 10, "polygon_rows": 6099} |
| val | 3354 | 76.2% | {"zero_area": 6, "polygon_rows": 624} |

## Dataset: merged

- Cross-split duplicates: **1723 exact**, **10289 near (Hamming<=4)**
- Undecodable images: **0**
  - EXACT test<->train: `c5_682_shoes1950.jpg` == `c5_1061_shoes1950.jpg`
  - EXACT test<->train: `c5_204_clothes154.jpg` == `c5_1111_clothes154.jpg`
  - EXACT test<->train: `c5_54_shoes57.jpg` == `c5_1225_shoes57.jpg`
  - EXACT test<->train: `c5_595_clothes5208.jpg` == `c5_1231_clothes5208.jpg`
  - EXACT test<->train: `c5_253_shoes1618.jpg` == `c5_129_shoes1618.jpg`
  - EXACT test<->train: `c5_546_shoes1150.jpg` == `c5_1309_shoes1150.jpg`
  - EXACT test<->train: `c5_185_battery77.jpg` == `c5_1364_battery77.jpg`
  - EXACT test<->train: `c5_544_clothes647.jpg` == `c5_1411_clothes647.jpg`
  - EXACT test<->train: `c5_243_clothes556.jpg` == `c5_1412_clothes556.jpg`
  - EXACT test<->train: `c5_447_battery654.jpg` == `c5_149_battery654.jpg`
  - NEAR ham=4 test<->train: `c5_283_green-glass300.jpg` ~ `c5_1047_battery669.jpg`
  - NEAR ham=4 test<->train: `c5_517_cardboard540.jpg` ~ `c5_1047_battery669.jpg`
  - NEAR ham=4 test<->train: `c5_676_cardboard_test_0221.jpg` ~ `c5_1047_battery669.jpg`
  - NEAR ham=4 test<->train: `c5_415_green-glass499.jpg` ~ `c5_1047_battery669.jpg`
  - NEAR ham=3 test<->train: `c5_773_battery437.jpg` ~ `c5_1079_battery362.jpg`
  - NEAR ham=4 test<->train: `c5_736_trash311.jpg` ~ `c5_108_shoes68.jpg`
  - NEAR ham=4 test<->train: `c5_54_shoes57.jpg` ~ `c5_108_shoes68.jpg`
  - NEAR ham=3 test<->train: `c5_773_battery437.jpg` ~ `c5_1126_battery83.jpg`
  - NEAR ham=3 test<->train: `c5_254_clothes3567.jpg` ~ `c5_1135_clothes1717.jpg`
  - NEAR ham=1 test<->train: `c5_325_clothes124.jpg` ~ `c5_1135_clothes1717.jpg`

| split | class | total | field | studio |
|---|---|---:|---:|---:|
| test | Background | 800 | 98 | 702 |
| test | cardboard | 800 | 462 | 338 |
| test | glass | 800 | 295 | 505 |
| test | metal | 800 | 496 | 304 |
| test | organic | 800 | 454 | 346 |
| test | paper | 800 | 431 | 369 |
| test | plastic | 800 | 475 | 325 |
| train | Background | 3500 | 437 | 3063 |
| train | cardboard | 3425 | 2000 | 1425 |
| train | glass | 3500 | 1359 | 2141 |
| train | metal | 3230 | 2000 | 1230 |
| train | organic | 3500 | 1958 | 1542 |
| train | paper | 3500 | 1905 | 1595 |
| train | plastic | 3384 | 2000 | 1384 |

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

## Cross-dataset eval contamination (hardcase eval vs other train sets)

- hardcase/val vs super/train: 1 exact + 795 near of 3538 images
- hardcase/test vs super/train: 0 exact + 366 near of 1363 images
- hardcase/val vs merged/train: 0 exact + 1340 near of 3538 images
- hardcase/test vs merged/train: 0 exact + 561 near of 1363 images

## Overfitting signals (yolo26n_hardcase_v2_long, may be mid-training)

- Epochs done: 100; best mAP50-95 0.5364 at epoch 100
- Last epoch: P 0.791 R 0.630 mAP50 0.707
- val-train cls-loss gap: 0.368 (mild gap normal; watch for val loss rising while train falls)
