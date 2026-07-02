# Model Risk Audit

Static provenance (verified by code reading, see plan/session notes):

- **Model inbreeding: CLEAR** - all labels trace to human annotations (TACO COCO via `export_taco_yolo_hardcase.py`, Roboflow-source annotations via `archive/build_super_yolo_dataset.py`); no script converts model predictions into labels. Upstream Roboflow community annotation quality is unverifiable.
- **Split handling: CLEAR** - manifest-driven / source-preserving; no random re-splitting by default.
- **Filename-marker cross-split contamination (merged_v5): CLEAR** - 0 in both directions.

## Dataset: hardcase

- Cross-split duplicates: **0 exact**, **4427 near (Hamming<=4)**
- Undecodable images: **0**
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
| test | plastic | 1897 | 3.86% | 24.8% |
| test | glass | 71 | 0.11% | 76.1% |
| test | metal | 566 | 8.11% | 15.7% |
| test | paper | 1375 | 18.2% | 13.2% |
| test | cardboard | 46 | 12.2% | 17.4% |
| test | organic | 46 | 1.29% | 41.3% |
| train | plastic | 15903 | 2.58% | 33.0% |
| train | glass | 7148 | 10.53% | 13.9% |
| train | metal | 8604 | 7.04% | 12.9% |
| train | paper | 4639 | 24.35% | 7.8% |
| train | cardboard | 7276 | 13.67% | 5.4% |
| train | organic | 31640 | 1.3% | 43.2% |
| val | plastic | 1632 | 1.63% | 39.6% |
| val | glass | 2478 | 9.39% | 12.3% |
| val | metal | 1742 | 4.29% | 23.2% |
| val | paper | 179 | 8.57% | 15.6% |
| val | cardboard | 1605 | 15.3% | 5.9% |
| val | organic | 12747 | 1.24% | 44.5% |

| split | label files | single-class img | noise |
|---|---:|---:|---|
| test | 1275 | 95.0% | {"polygon_rows": 10} |
| train | 20593 | 72.1% | {"polygon_rows": 5783} |
| val | 3450 | 76.9% | {"polygon_rows": 606} |

## Dataset: super

- Cross-split duplicates: **0 exact**, **3658 near (Hamming<=4)**
- Undecodable images: **0**
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
| test | plastic | 1658 | 4.57% | 21.7% |
| test | glass | 9 | 0.21% | 77.8% |
| test | metal | 542 | 8.53% | 15.1% |
| test | paper | 1347 | 19.24% | 13.1% |
| test | cardboard | 35 | 12.74% | 14.3% |
| test | organic | 46 | 1.29% | 41.3% |
| train | plastic | 14475 | 3.18% | 29.7% |
| train | glass | 6960 | 10.95% | 12.2% |
| train | metal | 8310 | 7.4% | 11.4% |
| train | paper | 4493 | 26.33% | 6.3% |
| train | cardboard | 7071 | 14.22% | 4.5% |
| train | organic | 31632 | 1.3% | 43.2% |
| val | plastic | 1149 | 2.17% | 32.6% |
| val | glass | 2474 | 9.38% | 12.2% |
| val | metal | 1699 | 4.35% | 22.5% |
| val | paper | 153 | 10.66% | 10.5% |
| val | cardboard | 1578 | 15.87% | 5.4% |
| val | organic | 12747 | 1.24% | 44.5% |

| split | label files | single-class img | noise |
|---|---:|---:|---|
| test | 1104 | 96.7% | {"polygon_rows": 10} |
| train | 19559 | 71.9% | {"polygon_rows": 5783} |
| val | 3266 | 76.7% | {"polygon_rows": 606} |

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

## Cross-dataset eval contamination (hardcase eval vs other train sets)

- hardcase/val vs super/train: 0 exact + 650 near of 3450 images
- hardcase/test vs super/train: 0 exact + 327 near of 1275 images
- hardcase/val vs merged/train: 0 exact + 1338 near of 3450 images
- hardcase/test vs merged/train: 0 exact + 561 near of 1275 images

## Overfitting signals (yolo26n_hardcase_v2_long, may be mid-training)

- Epochs done: 44; best mAP50-95 0.5228 at epoch 44
- Last epoch: P 0.753 R 0.617 mAP50 0.695
- val-train cls-loss gap: 0.031 (mild gap normal; watch for val loss rising while train falls)
