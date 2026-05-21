# Hướng dẫn Huấn luyện YOLOv11 trên Google Colab với GPU miễn phí

Tài liệu này cung cấp quy trình từng bước để nén bộ dữ liệu `taco_yolo` đã được tạo trên máy cục bộ, tải lên Google Drive, và tiến hành huấn luyện mô hình YOLOv11 trên Google Colab sử dụng GPU miễn phí (NVIDIA T4). Cách tiếp cận này giúp bạn tối ưu hóa thời gian huấn luyện từ vài ngày (trên CPU cục bộ) xuống còn **8-12 phút**!

---

## Bước 1: Nén Bộ Dữ Liệu trên Máy Cục Bộ

Để tải lên Google Drive nhanh chóng và tránh thất lạc file, hãy nén thư mục `taco_yolo` thành một file `.zip`.

1. Mở PowerShell trong thư mục dự án `C:\FYP_v2\`.
2. Chạy lệnh sau để nén thư mục `external_datasets/taco_yolo` thành `taco_yolo.zip`:
   ```powershell
   Compress-Archive -Path external_datasets/taco_yolo -DestinationPath taco_yolo.zip -Force
   ```
3. Sau khi nén xong, bạn sẽ thấy file `taco_yolo.zip` nằm ngay tại thư mục gốc của dự án.

---

## Bước 2: Tải lên Google Drive

1. Truy cập vào [Google Drive](https://drive.google.com/) của bạn.
2. Tạo một thư mục mới tên là `FYP_YOLO`.
3. Tải file `taco_yolo.zip` vừa nén lên thư mục này.

---

## Bước 3: Tạo Google Colab Notebook và Huấn Luyện

1. Truy cập [Google Colab](https://colab.research.google.com/).
2. Tạo một Notebook mới (New Notebook).
3. **Kích hoạt GPU**:
   - Vào menu **Runtime** > **Change runtime type**.
   - Tại mục *Hardware accelerator*, chọn **T4 GPU** (hoặc GPU bất kỳ đang miễn phí).
   - Nhấn **Save**.

4. **Kết nối Colab với Google Drive**:
   Tạo một cell code mới và chạy đoạn mã sau để mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

5. **Giải nén dữ liệu trên môi trường Colab**:
   Chạy lệnh giải nén file zip từ Drive trực tiếp vào phân vùng đĩa tốc độ cao của Colab (`/content/`):
   ```bash
   !unzip -q /content/drive/MyDrive/FYP_YOLO/taco_yolo.zip -d /content/
   ```

6. **Cài đặt thư viện Ultralytics**:
   ```bash
   !pip install ultralytics
   ```

7. **Điều chỉnh file `data.yaml` cho môi trường Colab**:
   Đường dẫn trên Colab sẽ khác với máy cục bộ của bạn. Hãy ghi đè nội dung file `data.yaml` để trỏ đúng thư mục giải nén:
   ```python
   import yaml

   yaml_path = '/content/taco_yolo/data.yaml'
   with open(yaml_path, 'r') as f:
       data = yaml.safe_load(f)

   # Cập nhật đường dẫn gốc trên Colab
   data['path'] = '/content/taco_yolo'

   with open(yaml_path, 'w') as f:
       yaml.safe_dump(data, f, sort_keys=False)

   print("Đã cấu hình lại file data.yaml cho Google Colab thành công!")
   ```

8. **Tiến hành Huấn luyện YOLOv11-Nano**:
   Chạy lệnh CLI của Ultralytics để bắt đầu huấn luyện. Chúng ta sẽ chạy 50 epochs với kích thước ảnh 640.
   **Lưu ý**: Chúng ta sử dụng trọng số khởi tạo là `yolo11n.pt` (YOLOv11-Nano):
   ```bash
   !yolo task=detect mode=train model=yolo11n.pt data=/content/taco_yolo/data.yaml epochs=50 imgsz=640 device=0 project=/content/drive/MyDrive/FYP_YOLO/runs name=yolov11_taco_colab
   ```
   *Lưu ý: Bằng việc đặt `project` trỏ về Google Drive, toàn bộ các checkpoint tốt nhất (`best.pt`), đồ thị biểu diễn loss/accuracy sẽ tự động được lưu trực tiếp vào Drive của bạn.*

---

## Bước 4: Tải Mô Hình đã Huấn Luyện về Máy Cục Bộ

Khi quá trình huấn luyện hoàn tất:
1. Vào Google Drive thư mục `FYP_YOLO/runs/yolov11_taco_colab/weights/`.
2. Tải file `best.pt` về máy tính của bạn.
3. Di chuyển file `best.pt` vào thư mục dự án cục bộ tại đường dẫn:
   `C:\FYP_v2\runs\detect\yolov11_taco\weights\best.pt`

Khi đã có file `best.pt` cục bộ, bạn chỉ cần chạy script suy luận `scripts/yolo_inference.py` để trực tiếp kiểm chứng sức mạnh phân loại và định vị tuyệt hảo của YOLOv11!
