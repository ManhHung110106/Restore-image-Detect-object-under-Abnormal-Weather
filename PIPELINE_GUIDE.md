# Hướng Dẫn Chi Tiết Pipeline Tối Ưu Thuật Toán Dehaze (DCP)

Tài liệu này ghi lại chi tiết toàn bộ phương pháp (method), quy trình (flow), cấu trúc thư mục, các file mã nguồn, cùng hướng dẫn cách chạy và triển khai hệ thống (bao gồm cả chạy trên Kaggle).

---

## 1. Phương Pháp (Method)
Phương pháp chính là **Dark Channel Prior (DCP)** - một thuật toán xử lý ảnh truyền thống không sử dụng Machine Learning để học trọng số, mà hoạt động dựa trên các tham số nội tại (hyperparameters).
Mục tiêu là **Tối ưu hóa siêu tham số (Hyperparameter Tuning)** cho DCP:
- `omega`: Tham số kiểm soát mức độ sương mù giữ lại (tạo độ sâu tự nhiên).
- `t0`: Ngưỡng truyền qua tối thiểu (ngăn vùng quá đặc sương bị đen).
- `patch_size`: Kích thước cửa sổ nội bộ để tìm Dark Channel.

**Thuật toán tìm kiếm**: Sử dụng **Optuna Multi-objective Optimization** để tìm ra tập hợp các bộ siêu tham số tốt nhất (Pareto Front) cân bằng giữa nhiều metrics đánh giá chất lượng ảnh.

---

## 2. Quy Trình (Flow)
Pipeline gồm 4 pha chính:
1. **Pha 1 (Tối ưu Ground Truth):** Tìm bộ tham số tối ưu (config_1) trên tập RESIDE (SOTS) với ảnh Ground Truth. Metrics: Tối đa hóa `PSNR`, `SSIM` và Tối thiểu hóa `LPIPS`.
2. **Pha 2 (Tối ưu No-Reference):** Tìm bộ tham số tối ưu (config_2) trên tập DAWN (ảnh không có Ground Truth). Metrics: Tối thiểu hóa `NIQE` và `BRISQUE` (sử dụng thư viện `pyiqa`).
3. **Pha 4 (Tối ưu Detection):** Tìm bộ tham số tối ưu (config_3) trên tập DAWN nhưng tối ưu trực tiếp cho AI. Metrics: Tối đa hóa `mAP@0.5:0.95`, `mAP@0.5` và `Recall` thông qua mô hình YOLOv8 in-memory.
4. **Pha 3 (Cross-Evaluation):** Lấy các config tìm được từ 3 pha trên để đánh giá chéo (Hold-out Test 20%) để rút ra phân tích toàn diện.

---

## 3. Quản Lý Dataset
### 3.1. RESIDE (SOTS)
- **Lưu trữ gốc:** Tải trực tiếp qua thư viện `kagglehub`.
- **Lấy ra & Xử lý:** Script `scripts/prepare_sots.py` tự động lấy đường dẫn ảnh từ cache của `kagglehub`, sau đó chia thành 80% Tuning (dùng thay Train/Val) và 20% Test.
- **Lưu trữ config:** Kết quả phân tách được lưu tại `configs/sots_split.json` (chỉ chứa đường dẫn, không copy file vật lý để tiết kiệm ổ cứng).

### 3.2. DAWN
- **Xử lý:** Dataset DAWN đã được chia theo chuẩn YOLO (vào các thư mục `train`, `val`, `test` tại `data/processed/dawn_yolo`).
- **Sử dụng:** Script sẽ quét thư mục `train` và `val` (chứa 80% dữ liệu) để chạy Optuna, và dùng thư mục `test` (chứa 20% dữ liệu) cho Cross-Evaluation. Ảnh nằm trong các sub-folder theo thời tiết (Fog, Rain, Snow, Sand) và script đã dùng cơ chế `glob recursive` để đọc toàn bộ.

---

## 4. Giải Thích Chi Tiết Từng File Code

- **`scripts/prepare_sots.py`**: 
  - *Chức năng*: Khám phá thư mục tải về của `kagglehub`, lấy danh sách ảnh hazy/clear tương ứng, trộn đều (shuffle với seed=42) và chia tỷ lệ 80:20. Sinh ra file cấu hình `configs/sots_split.json`.

- **`src/filters/optuna_tune_gt.py`**:
  - *Chức năng*: Chạy Pha 1 (Ground Truth).
  - *Logic*: Load ảnh từ JSON, với mỗi Trial của Optuna nó lấy 1 tập ngẫu nhiên (max 100 ảnh để tối ưu tốc độ), chạy dehaze, tính `PSNR`, `SSIM` (skimage) và `LPIPS` (thư viện lpips).
  
- **`src/filters/optuna_tune_nogt.py`**:
  - *Chức năng*: Chạy Pha 2 (No-Reference).
  - *Logic*: Load ảnh từ `data/processed/dawn_yolo/images/train` và `val`. Chạy dehaze và tính `NIQE`, `BRISQUE` qua module AI `pyiqa`. Cố gắng tìm ra tham số để ảnh trong tự nhiên nhất.
  
- **`src/detection/optuna_tune_yolo.py`**:
  - *Chức năng*: Chạy Pha 4 (Detection-driven).
  - *Logic*: Khởi tạo YOLOv8 1 lần trên RAM. Với mỗi Trial, chạy dehaze, cho ảnh dehazed chạy qua YOLO. Compare bbox dự đoán với bbox ground truth từ file `.txt`. Tính mAP qua `torchmetrics` mà không cần ghi file ảnh ra đĩa.

- **`src/evaluation/cross_eval.py`**:
  - *Chức năng*: Chạy Pha 3. Nhận đầu vào là tham số cứng (`--patch_size`, `--omega`, `--t0`). Dehaze trên toàn bộ 20% Test Set của SOTS và DAWN, trả về tất cả mọi metrics (PSNR, SSIM, LPIPS, NIQE, BRISQUE, YOLO mAP) in ra Terminal.

---

## 5. Output và Phiên Bản (Versioning)

Optuna tự động lưu toàn bộ dữ liệu vào **Database SQLite**. Do đó, không có kết quả nào bị mất đi hoặc ghi đè kể cả khi bạn dừng đột ngột và chạy lại (nó sẽ tiếp tục (resume) tại vị trí bị ngắt).

Tuy nhiên, để bạn có thể phân biệt các *lần chạy khác nhau* (ví dụ chạy hôm nay với chạy ngày mai), tôi đã thiết lập để bạn có thể truyền tham số tên database ở Terminal. Bạn nên đặt tên kèm timestamp.

**Cấu trúc Output:**
- `optuna_gt_<timestamp>.db`: File database chứa toàn bộ quá trình chạy Pha 1.
- `optuna_nogt_<timestamp>.db`: File database cho Pha 2.
- `optuna_yolo_<timestamp>.db`: File database cho Pha 4.

Bạn có thể dùng công cụ `optuna-dashboard sqlite:///tên_file.db` để mở giao diện Web xem các đồ thị Pareto và bảng số liệu rất đẹp trực quan.

---

## 6. Hướng Dẫn Chạy (Cách sử dụng)

### Cài đặt môi trường
Đảm bảo bạn cài các thư viện sau:
```bash
pip install optuna lpips pyiqa torchmetrics ultralytics scikit-image opencv-python
```

### Chế độ Test (Kiểm tra xem code có lỗi không)
Chạy với 2-3 trials để thấy code kết thúc thành công:
```bash
python src/filters/optuna_tune_gt.py --trials 2 --out_db sqlite:///test_gt.db
python src/filters/optuna_tune_nogt.py --trials 2 --out_db sqlite:///test_nogt.db
python src/detection/optuna_tune_yolo.py --trials 2 --out_db sqlite:///test_yolo.db
```

### Chế độ Chạy Thật (Tối ưu sâu)
Khuyến nghị chạy từ 50-100 trials qua đêm. Sử dụng timestamp ở biến out_db để lưu lại:
```bash
# Trên Windows PowerShell bạn có thể gõ:
$time = Get-Date -Format "yyyyMMdd_HHmm"
python src/filters/optuna_tune_gt.py --trials 50 --out_db "sqlite:///optuna_gt_$time.db"
```

### Đánh giá chéo (Sau khi có kết quả từ Optuna)
Ví dụ Optuna báo config tốt nhất là `patch_size=15`, `omega=0.85`, `t0=0.1`.
```bash
python src/evaluation/cross_eval.py --patch_size 15 --omega 0.85 --t0 0.1 --save_output
```
**Lưu ý**: Khi thêm cờ `--save_output`, script sẽ tự động tạo một thư mục `runs/cross_eval/run_<timestamp>` và lưu lại toàn bộ ảnh đã dehaze của SOTS, ảnh vẽ bounding box của DAWN, file `config.json` chứa các cấu hình đã chọn, và `metrics.json` lưu log thông số.

---

## 7. Làm sao để tải và chạy trên Kaggle

Kaggle là môi trường hoàn hảo vì cung cấp sẵn GPU T4x2 hoặc P100 (giúp load mô hình LPIPS, pyiqa và YOLO cực nhanh).

**Bước 1: Khởi tạo Notebook Kaggle**
Tạo một Notebook mới, gắn bộ dữ liệu DAWN (nếu có trên Kaggle) hoặc zip thư mục `ProjectCV` của bạn lên. Bật tính năng **Internet** và **GPU** trong cài đặt Notebook.

**Bước 2: Cài thư viện**
Tạo 1 block code đầu tiên:
```python
!pip install optuna lpips pyiqa torchmetrics ultralytics
```

**Bước 3: Chạy Scripts**
Sử dụng cú pháp lệnh command line `!` của Jupyter:
```python
import time
timestamp = time.strftime("%Y%m%d_%H%M")

# Chạy tối ưu Pha 1
!python src/filters/optuna_tune_gt.py --trials 50 --out_db sqlite:///optuna_gt_{timestamp}.db

# Chạy tối ưu Pha 2
!python src/filters/optuna_tune_nogt.py --trials 50 --out_db sqlite:///optuna_nogt_{timestamp}.db

# Chạy tối ưu Pha 4
!python src/detection/optuna_tune_yolo.py --trials 50 --out_db sqlite:///optuna_yolo_{timestamp}.db
```

**Bước 4: Lưu kết quả**
Các file `.db` sẽ xuất hiện ở thư mục `/kaggle/working`. Khi notebook chạy xong, bạn tải các file `.db` này về máy cá nhân và dùng lệnh `optuna-dashboard` để xem kết quả.
