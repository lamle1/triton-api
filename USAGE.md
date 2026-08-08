# Hướng Dẫn Sử Dụng Và Tích Hợp API System Triton API

Tài liệu này hướng dẫn chi tiết cách khai thác các tính năng trên Giao diện Web Admin Portal, Vision NVR Client và hướng dẫn lập trình viên gọi REST API / WebSocket để tích hợp hệ thống **Triton API**.

---

## 1. Hướng Dẫn Triển Khai & Cấu Hình

### 1.1. Tải Mã Nguồn & Cấu Hình Môi Trường
```bash
# 1. Clone repository
git clone https://github.com/lamle1/triton-api.git
cd triton-api

# 2. Cấu hình biến môi trường và Docker Compose
cp .env.example .env
cp docker-compose.example.yaml docker-compose.yaml

# 3. Khởi chạy toàn bộ hệ thống
docker compose up -d
```

---

## 2. Hướng Dẫn Bảng Điều Khiển Quản Trị (Admin Web Portal)

Địa chỉ truy cập: `http://<API_SERVER_IP>:8003/admin/`

### 2.1. Đăng Nhập Hệ Thống
1. Nhập Mật khẩu Quản trị (đã thiết lập qua biến môi trường `ADMIN_PASSWORD` trong file `.env`).
2. Nhấp **Login** để nhận Session Token (lưu trữ an toàn trong Cookie trình duyệt `admin_session`).

### 2.2. Quản Lý Tài Khoản Quản Trị (Accounts Management)
- Tạo tài khoản admin phụ, đổi mật khẩu và xóa tài khoản quản trị trực tiếp trên tab **Accounts**.

### 2.3. Quản Lý API Key Doanh Nghiệp (API Key Management)
Giao diện quản lý API Key cung cấp các chức năng chuẩn bảo mật doanh nghiệp:
- **Tạo API Key Mới**: Nhấp button **Create Secret Key**, nhập tên mô tả (ví dụ `Client_NVR_Camera_Floor1`) và chọn thời hạn hết hạn (7 ngày, 30 ngày, Không hết hạn). Hệ thống sinh ngẫu nhiên 32-byte key kèm tiền tố `tr_live_...` và hiển thị 1 lần duy nhất.
- **Chỉnh Sửa Thông Tin Key**: Nhấp biểu tượng **Edit** trên dòng tương ứng để cập nhật Tên khóa hoặc Gia hạn ngày hết hạn (`PUT /api/v1/admin/keys/{id}`).
- **Thu Hồi / Xóa Cứng Key**: Nhấp biểu tượng **Revoke / Delete** để xóa vĩnh viễn khóa khỏi CSDL SQLite (`DELETE /api/v1/admin/keys/{id}`).
- **Theo Dõi Lượt Gọi (Usage Stats)**: Hiển thị tổng số request đã xử lý và thời gian sử dụng gần nhất của từng khóa.

### 2.4. Giám Sát Tài Nguyên Container (System Resource Monitoring)
- Tab **Status**: Hiển thị chỉ số phần trăm CPU (`% CPU Usage`), dung lượng RAM đang sử dụng (`Memory Usage / Limit`) và băng thông mạng I/O của Triton Container real-time thông qua Docker Engine API socket.

---

## 3. Hướng Dẫn Sử Dụng NVR Stream & AI Detection UI

Tất cả các tính năng xem video thời gian thực và quản lý mô hình AI được tích hợp trực tiếp trên Web Admin Portal (`/admin/`).

### 3.1. Kết Nối Luồng Camera RTSP Real-Time (Stream Tab)
1. Mở tab **Stream** trên Web Admin Portal (`/admin/`).
2. Nhấp **+ Add Stream**, chọn loại nguồn **RTSP Stream**.
3. Nhập URL Camera RTSP (Ví dụ: `rtsp://admin:password@192.168.1.100:554/stream1`).
4. Chọn Mô hình AI nhận diện (Ví dụ: `yolov8s` hoặc `yoloe-s`).
5. Nhập danh sách từ khóa phân lớp tùy chỉnh (Prompts/Classes) nếu dùng YOLOE (Ví dụ: `fire, smoke, helmet`).
6. Bật/Tắt cờ **Enable Tracking** để kích hoạt bộ theo vết ByteTrack và lưu trữ đặc trưng Re-ID.
7. Nhấp **CONNECT** để bắt đầu nhận luồng WebRTC/JPEG video hiển thị bounding boxes.

### 3.2. Cơ Chế Multi-Session Stream Isolation (Tách Luồng Tự Động)
- Khi User A kết nối camera với mô hình `yolov8s` và User B kết nối tới cùng URL camera đó nhưng chọn mô hình `yoloe-s`: Hệ thống tự động phân tách 2 luồng worker riêng biệt (`stream_id` độc lập).
- Mỗi người dùng chỉ nhận đúng các bounding box và kết quả nhận diện của mô hình mình đăng ký, không bị chồng chéo khung vuông (Overlapping Boxes).

---

## 4. Hướng Dẫn Tìm Kiếm Đối Tượng Chéo Camera (Cross-Camera Re-ID)

### 4.1. Truy Vấn Theo Ảnh Upload (Search By Image)
1. Mở tab **Tracking** trên Admin Portal (`/admin/`).
2. Tải lên 1 bức ảnh crop đối tượng cần tìm (người, phương tiện).
3. Đặt Ngưỡng Tương Đồng Cosine (Khuyên dùng `0.60` - `0.70`).
4. Nhấp **Search Vector DB**.
5. Hệ thống gọi mô hình **TransReID ViT-S/16 / OSNet** trích xuất Vector 512-D và đối sánh trên Qdrant Vector DB trong <10ms.

### 4.2. Hiển Thị Kết Quả & Sơ Đồ Lộ Trình (Space-Time Trajectory Map)
- Danh sách kết quả được nhóm theo `global_id` duy nhất của từng đối tượng.
- Hiển thị danh sách các mốc thời gian, tên camera và ảnh crop tương đồng.
- Dựng sơ đồ vị trí lộ trình xuất hiện theo mốc thời gian và cho phép xem lại đoạn video sự kiện HLS/MP4 phát lại.

---

## 5. Tài Liệu REST API Cho Lập Trình Viên (API Reference)

Trình duyệt Swagger & ReDoc trực quan:
- **Swagger UI Interactive API**: `http://<API_SERVER_IP>:8003/docs`
- **ReDoc API Reference**: `http://<API_SERVER_IP>:8003/redoc`

Tất cả các REST API request suy luận cần truyền Header xác thực khi `REQUIRE_API_KEY=true`:
```http
X-API-Key: tr_live_your_secret_api_key_here
```

### 5.1. Khởi Tạo Luồng Camera RTSP (`POST /streams`)
**Endpoint**: `POST /streams`

**Request Body (JSON)**:
```json
{
  "name": "Front Gate Camera",
  "url": "rtsp://admin:pass@192.168.1.100:554/live",
  "models": ["yolov8s"],
  "classes": "person,car",
  "conf": 0.5,
  "fps": 30,
  "preview_fps": 10,
  "enable_tracking": true,
  "enable_recording": false
}
```

**Response (JSON)**:
```json
{
  "id": "stream-1",
  "name": "Front Gate Camera",
  "url": "rtsp://admin:pass@192.168.1.100:554/live",
  "status": "running",
  "requested_models": ["yolov8s"],
  "live_transport": "go2rtc",
  "go2rtc_name": "triton_stream_1_raw",
  "go2rtc_public_url": "http://localhost:1984",
  "tracking_enabled": true,
  "recording_enabled": false
}
```

---

### 5.2. Truy Vấn Re-ID Theo Ảnh Upload (`POST /api/v1/tracking/search-by-image`)
**Endpoint**: `POST /api/v1/tracking/search-by-image`

**Form Data**:
- `file`: File ảnh crop BGR (`image/jpeg` hoặc `image/png`).
- `score_threshold`: float (Mặc định: `0.60`).
- `limit`: int (Mặc định: `20`).

**Ví dụ lệnh cURL**:
```bash
curl -X POST "http://localhost:8003/api/v1/tracking/search-by-image" \
  -H "X-API-Key: tr_live_your_secret_api_key" \
  -F "file=@/path/to/target_person.jpg" \
  -F "score_threshold=0.65" \
  -F "limit=10"
```

**Response (JSON)**:
```json
{
  "status": "success",
  "query_count": 10,
  "results": [
    {
      "global_id": "PER-5C07E8",
      "similarity_score": 0.8942,
      "class_name": "person",
      "camera_id": "cam_entrance_gate",
      "camera_name": "Front Gate Camera",
      "timestamp": "2026-08-05T22:20:27.123456",
      "bbox": [0.35, 0.22, 0.55, 0.78],
      "image_url": "http://localhost:8003/events_images/crop_PER-5C07E8_1721832000.jpg"
    }
  ]
}
```

---

### 5.3. Quản Lý API Key via Admin REST API (`PUT` & `DELETE`)

#### Cập Nhật Tên / Hạn Dùng Key (`PUT /api/v1/admin/keys/{id}`)
```bash
curl -X PUT "http://localhost:8003/api/v1/admin/keys/1" \
  -H "Content-Type: application/json" \
  -b "admin_session=your_session_cookie" \
  -d '{
    "key_name": "Updated_Floor2_Camera",
    "expires_days": 30
  }'
```

#### Thu Hồi & Xóa Cứng Key (`DELETE /api/v1/admin/keys/{id}`)
```bash
curl -X DELETE "http://localhost:8003/api/v1/admin/keys/1" \
  -b "admin_session=your_session_cookie"
```
