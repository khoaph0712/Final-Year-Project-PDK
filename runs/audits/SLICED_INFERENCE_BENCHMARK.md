# Sliced vs whole-image inference (clean val, deployed detector)

- 800 images, conf=0.3, imgsz=960, tiny = GT box < 1% image area

| mode | tiny recall | other recall | precision | avg ms/img |
|---|---:|---:|---:|---:|
| whole | 0.266 | 0.607 | 0.801 | 18 |
| sliced | 0.300 | 0.532 | 0.662 | 66 |