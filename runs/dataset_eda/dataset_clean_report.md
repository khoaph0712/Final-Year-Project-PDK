# Báo Cáo Bảo Dưỡng & Làm Sạch Dữ Liệu Học Máy (Dataset Audit & Clean Report)

Báo cáo này lập hồ sơ lưu trữ khoa học về việc loại bỏ nhiễu nhãn và rò rỉ dữ liệu để chuẩn bị cho hội đồng bảo vệ khóa luận.

## 1. Kết Quả Bảo Dưỡng Tổng Quan
- **Tổng số ảnh quét thành công:** 23960
- **Số lượng ảnh lỗi/hỏng bị loại bỏ:** 0
- **Số lượng tọa độ nhãn tràn viền được chuẩn hóa:** 42
- **Số lượng hộp giới hạn siêu nhỏ (Nhiễu <12x12px) bị cắt bỏ:** 4158
- **Số lượng ảnh trùng lặp gây rò rỉ dữ liệu (Data Leakage) giữa Train-Test bị xóa:** 628
- **Số lượng mẫu trống (không còn hộp giới hạn) bị loại bỏ:** 31

## 2. Thống Kê Sự Thay Đổi Số Lượng Nhãn
| Tên Lớp (Class) | Trước Khi Lọc | Sau Khi Lọc | Thay Đổi (Cắt Nhiễu) |
| :--- | :---: | :---: | :---: |
| **PLASTIC** | 21925 | 21654 | -271 |
| **GLASS** | 9830 | 9683 | -147 |
| **METAL** | 11405 | 11127 | -278 |
| **PAPER** | 6328 | 6266 | -62 |
| **CARDBOARD** | 9159 | 9082 | -77 |
| **ORGANIC** | 48288 | 44965 | -3323 |

## 3. Ý Nghĩa Khoa Học Của Việc Làm Sạch Dữ Liệu
> [!IMPORTANT]
> **Chống rò rỉ dữ liệu (Data Leakage):** Việc loại bỏ hoàn toàn các ảnh trùng lặp vật lý giữa tập huấn luyện (Train) và tập kiểm thử (Test) đảm bảo rằng các chỉ số F1-Score, mAP đạt được ở giai đoạn đánh giá là **hoàn toàn chính xác, trung thực và khoa học**, không bị thổi phồng ảo.
>
> [!TIP]
> **Cắt lọc nhiễu vi mô (Microscopic Noise Filtering):** Các hộp giới hạn siêu nhỏ (<12px) thường sinh ra do lỗi đánh nhãn bằng tay trên các điểm ảnh mờ ở hậu cảnh xa. Nếu giữ lại, chúng sẽ ép mô hình YOLO phải học các chi tiết không có thật, làm giảm khả năng hội tụ và gây ra hiện tượng Overfitting.
