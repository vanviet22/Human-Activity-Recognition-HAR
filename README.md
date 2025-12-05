📖 1. Giới thiệu

Dự án xây dựng hệ thống nhận dạng hành động của con người trong ảnh tĩnh, kết hợp giữa mô hình phát hiện đối tượng (object detection) và mô hình phân loại hành động (action classification).

Pipeline chính:

YOLOv8n – phát hiện người và các đồ vật liên quan trong ảnh

Cluster Person + Object Boxes – gom nhóm người và đồ vật theo ngữ cảnh

EfficientNetB1 (Fine-tuned) – phân loại hành động dựa trên vùng ảnh đã gom/crop

FastAPI Backend – xử lý ảnh/camera

Frontend Web – giao diện chạy trực tiếp trên trình duyệt

Hệ thống hoạt động tốt trên ảnh tải lên và camera realtime.

📊 2. Bộ dữ liệu

Nguồn: Kaggle – Human Action Recognition (HAR). ULR : https://www.kaggle.com/code/kirollosashraf/human-action-recognition-har

💠 Thông tin dữ liệu:

12.600 ảnh train (15 lớp, mỗi lớp 840 ảnh → cân bằng tuyệt đối)

5.410 ảnh test

Kích thước ảnh: 240×240 RGB

Một ảnh chỉ chứa một hành động

💠 15 lớp hành động:
Calling, Clapping, Cycling, Dancing, Drinking, Eating,
Fighting, Hugging, Laughing, Listening to Music,
Running, Sitting, Sleeping, Texting, Using Laptop

💠 Đặc điểm quan trọng ảnh hưởng pipeline:

Một số hành động cần nhiều người tương tác: hugging, fighting, kissing

Một số hành động cần đồ vật: laptop, bicycle, cell phone, bottle

YOLO trả về box riêng lẻ → phải tự gom nhóm để mô hình hiểu đúng hành động

🧠 3. Phương pháp & Mô hình
3.1 YOLOv8n – Phát hiện người + đồ vật liên quan

YOLO được dùng để lấy:

Box người (person)

Box đồ vật liên quan đến hành động:

laptop, cell phone, bottle, cup, book, bicycle, …

Mỗi box bao gồm:

(x1, y1, x2, y2)

3.2 Vấn đề: Nhiều người + Nhiều đồ vật

YOLO phát hiện từng đối tượng riêng lẻ, dẫn đến:

Các hành động nhóm (hugging, fighting) → cần gom người lại

Các hành động gắn với đồ vật (using laptop, cycling) → cần gắn object vào đúng người

Nếu không xử lý → action model sẽ nhận một người riêng lẻ → phân loại sai.

3.3 Cluster Person Boxes – Giải quyết ngữ cảnh hành động

Nhóm em xây dựng hàm cluster_person_boxes với hai nhiệm vụ:

✔ 1) Gom các box người lại thành một cụm

Tính khoảng cách giữa tâm box

Gom vào cùng một cluster nếu khoảng cách < dist_thresh

✔ 2) Gắn các object-box liên quan vào đúng cụm người

Dựa trên khoảng cách person–object

Mỗi cluster tạo thành một vùng ảnh mang đầy đủ ngữ cảnh

Ví dụ:

2 người ôm nhau → 1 cluster

Người + laptop gần nhau → crop chung

Người + bicycle → gom lại thành vùng hành động đầy đủ

3.4 EfficientNetB1 – Phân loại hành động

Train theo phương pháp transfer learning:

Freeze backbone EfficientNetB1

Thêm:

BatchNorm

Dropout(0.6)

Dense(15, softmax)

Training:

Optimizer: Adam (1e-4)

Label smoothing: 0.1

EarlyStopping + ReduceLR + ModelCheckpoint

Input: 160×160 crop

Output: 15 hành động

🏗 4. Kiến trúc tổng thể
Ảnh/Camera → YOLOv8 → Lấy người + đồ vật → Cluster → Crop cluster → EfficientNetB1 → Hành động


Thành phần chính:

backend/recognize.py – YOLO + clustering + crop + phân loại

backend/main.py – FastAPI endpoints

frontend/ – HTML/CSS/JS chạy trực tiếp trong trình duyệt

📂 5. Cấu trúc thư mục
backend/
│── main.py                 # FastAPI server
│── recognize.py            # YOLO detection, clustering, action recognition
│── output_model/           # EfficientNetB1 đã train
│── training_process/       # Notebook huấn luyện Kaggle
frontend/
│── index.html
│── style.css
│── script.js
requirements.txt
README.md

🚀 6. Cài đặt & Chạy dự án
1️⃣ Cài thư viện
pip install -r requirements.txt

2️⃣ Chạy backend FastAPI
uvicorn backend.main:app --reload

3️⃣ Mở giao diện web
frontend/index.html

🖼 7. Kết quả

Nhận dạng ảnh và camera realtime

Phân loại tốt 15 hành động

Hoạt động ổn định với người + đồ vật + nhiều người tương tác

Clustering cải thiện rõ độ chính xác với các hành động nhóm
