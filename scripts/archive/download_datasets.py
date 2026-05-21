from roboflow import Roboflow

# ── Paste your NEW API key here (only place) ────────────────
RF_API_KEY = "kPxvBKgvQPxtChunxphQ"
# ────────────────────────────────────────────────────────────

rf = Roboflow(api_key=RF_API_KEY)

print("\n[+] Downloading Trash Detection (64 classes, ~2,800 imgs)...")
rf.workspace("trash-dataset-for-oriented-bounded-box") \
  .project("trash-detection-1fjjc") \
  .version(1) \
  .download("yolov8", location=r"C:\FYP_v2\rf_trash_detection")

print("\n[+] Downloading TACO Trash Detections (~12,800 imgs)...")
rf.workspace("taco-ihjgk") \
  .project("yolov8-trash-detections-kgnug") \
  .version(7) \
  .download("yolov8", location=r"C:\FYP_v2\rf_taco_trash")

print("\n[+] Downloading Garbage Classification (~2,800 imgs)...")
rf.workspace("material-identification") \
  .project("garbage-classification-3") \
  .version(2) \
  .download("yolov8", location=r"C:\FYP_v2\rf_garbage_cls")

print("\n[OK] All downloads complete!")
print("Check your folders in C:\\FYP_v2\\")