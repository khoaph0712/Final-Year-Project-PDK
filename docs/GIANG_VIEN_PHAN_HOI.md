# Phản hồi giảng viên — dataset, lớp, cân bằng, hình loss/accuracy, features & models

Em có thể **copy đoạn trả lời ngắn** gửi cô hoặc ghép vào báo cáo / slide.

---

## 1. Em đang dùng dataset nào?

Em dùng **`merged_dataset_v3`** — bộ dữ liệu YOLO (thư mục `train/`, `valid/`, `test/` với `images/` và `labels/`), được ghép từ nhiều nguồn Roboflow và xử lý lại (organic, glass test, v.v.). File cấu hình: **`merged_dataset_v3/data.yaml`**.

*(Nếu em train YOLO/DL trực tiếp trên ảnh: cùng dataset v3; nhánh ML phân loại **crop từ bbox** vẫn đọc từ dataset đó.)*

---

## 2. Em train bao nhiêu class?


| Nhánh                                         | Số class | Ghi chú                                                                                                               |
| --------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| **YOLO / DL detector** (Ultralytics)          | **7**    | `plastic`, `glass`, `metal`, `paper`, `cardboard`, `organic`, `other` — đúng `nc: 7` trong `data.yaml`.               |
| **ML handcrafted features** (script hiện tại) | **6**    | Em **loại `other`** khỏi phần ML phân tích để tập trung các lớp rác có nhãn rõ hơn; số lớp trong báo cáo ML là **6**. |


Nếu cô hỏi “tại sao ML 6 mà YOLO 7”: em ghi ràng là **cùng dataset**, nhưng **ML run có chọn exclude một lớp** (`other`) theo hướng dẫn thí nghiệm.

---

## 3. Data training có imbalanced không?

- **Ở mức dataset gốc (folder train):** có thể **lệch class** vì nguồn ghép và số ảnh/bbox khác nhau giữa các loại rác.
- **Ở nhánh ML (crop + classical models):** em có **giới hạn tối đa số crop / class** (per-class cap, ví dụ 4000/class trên train nếu đủ dữ liệu) để **cố gắng cân bằng ở mức “object crop”**. Lớp ít dữ liệu có thể **không đủ** để đạt cap → vẫn lệch nhẹ; chi tiết xem **`class_support.json`** trong thư mục output ML (số mẫu train/test từng lớp).
- **Test split** sau khi shuffle có thể **không đều** giữa các lớp nếu trong tập test không đủ bbox — em **ghi rõ trong báo cáo** và/hoặc bổ sung bảng support.

**Câu trả lời một dòng cho cô:** *“Dataset gốc có thể imbalanced; nhánh ML em chủ động cap per-class để cân bằng crop; em báo cáo kèm `class_support.json` và nhận xét lớp nào thiếu mẫu.”*

---

## 4. Để trong report hình loss / accuracy cho cô xem

### Deep learning (YOLO — có loss & metric theo epoch)

- **`runs/dl/trash_yolov8n_v3/results.png`** — Ultralytics tổng hợp training (loss, precision/recall, mAP…).
- **`runs/dl/trash_yolov8n_v3/results.csv`** — số liệu để vẽ lại nếu cần.
- Chạy **`python scripts/plot_training.py`** (mặc định trỏ v3) → thêm **`training_curves.png`** trong `quality_check/` của run đó.

### Deep learning baseline (CNN nhỏ trên crop)

- **`runs/dl/dl_baseline/training_loss.png`** — loss theo epoch.

### Classical ML trên **vector đặc trưng** (không train kiểu neural epoch như YOLO)

- **`runs/ml/<run>/chart_model_comparison.png`** — **Accuracy và F1-macro** cho LogReg, SVM-RBF, RF (không có đường loss epoch vì không phải SGD end-to-end trên ảnh).
- **`runs/ml/<run>/confusion_*.png`** — ma trận nhầm lẫn từng model.

**Ghi chú cho báo cáo:** *Phần ML classical là “fit trên ma trận đặc trưng đã trích”, nên biểu đồ phù hợp là **accuracy/F1 + confusion matrix**, không phải loss vs epoch như YOLO.*

---

## 5. Ghi rõ đang train với **features** nào và **models** nào?

### Features (17 chiều / mỗi crop)

**Spatial (8):**  
`mean_intensity`, `std_intensity`, `p10_intensity`, `p50_intensity`, `p90_intensity`, `grad_mean`, `grad_std`, `edge_density`

**Frequency (9):**  
`fft_bin_1` … `fft_bin_8`, `high_freq_energy`  
*(FFT 2D, năng lượng theo vòng tròn bán kính + tóm tắt dải cao tần.)*

### Models (ML trên vector trên)


| Tên trong code | Mô hình                                                                         |
| -------------- | ------------------------------------------------------------------------------- |
| `logreg`       | Logistic Regression + Chuẩn hóa `StandardScaler`                                |
| `svm_rbf`      | SVM kernel RBF trên đặc trưng đã scale                                          |
| `rf`           | Random Forest (300 cây), có feature importance cho biểu đồ spatial vs frequency |


*(YOLO / tiny CNN là nhánh DL riêng — em liệt kê trong mục DL của báo cáo.)*

---

## 6. Sau khi em chạy lại `feature_ml_analysis.py`

File **`runs/ml/feature_ml_analysis/REPORT.md`** (hoặc `--out` em chọn) đã được bổ sung các mục:

- Dataset & classes  
- Danh sách đặc trưng đầy đủ  
- Bảng models  
- Gợi ý hình nào đưa vào báo cáo (ML + DL)

Em có thể đính kèm PDF/export từ đó cho cô.

---

*Tài liệu này song song với tiếng Việt để em trả lời trực tiếp feedback của cô.*
