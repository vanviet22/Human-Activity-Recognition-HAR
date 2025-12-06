# 📌 Human Activity Recognition (HAR)

      - YOLOv8n Detection + EfficientNetB1 Classification + FastAPI + Web UI

### 📖 1. Giới thiệu

     - Đề tài xây dựng hệ thống nhận dạng hành động của con người trong ảnh tĩnh, kết hợp mô hình phát hiện đối tượng (object detection) và mô hình phân loại hành động (action classification).

      - Pipeline tổng quan:

            + YOLOv8n – phát hiện người và các đồ vật liên quan trong ảnh

            + Cluster Person + Object Boxes – gom nhóm người và đồ vật trong cùng ngữ cảnh

            + EfficientNetB1 (Fine-tuned) – phân loại hành động trên vùng ảnh đã gom/crop

            + FastAPI Backend – xử lý ảnh và camera

            + Frontend Web – giao diện chạy trực tiếp trong trình duyệt

      - Hệ thống hoạt động tốt trên ảnh tải lên và camera realtime.

### 📊 2. Bộ dữ liệu

      - Nguồn: Kaggle – Human Action Recognition (HAR)
      
      - URL: https://www.kaggle.com/code/kirollosashraf/human-action-recognition-har
      
      - Thông tin dữ liệu

            + 12.600 ảnh train (15 lớp, mỗi lớp 840 ảnh – cân bằng tuyệt đối)

            + 5.410 ảnh test

            + Kích thước ảnh: 240×240 RGB

            + Mỗi ảnh chứa một hành động

      - 15 lớp hành động: Calling, Clapping, Cycling, Dancing, Drinking, Eating, Fighting, Hugging, Laughing, Listening to Music, Running, Sitting, Sleeping, Texting, Using Laptop

      - Đặc điểm quan trọng của dữ liệu

            + Một số hành động cần nhiều người tương tác: hugging, fighting, kissing

            + Một số hành động cần đồ vật: laptop, bicycle, cell phone, bottle,...

### 🧠 3. Phương pháp & Mô hình

3.1 YOLOv8n – Phát hiện người + đồ vật liên quan

      - YOLO được sử dụng để phát hiện:

            + Box người (person)

            + Box đồ vật liên quan đến hành động: laptop, cell phone, bottle, cup, book, bicycle, …

            + Mỗi bounding box gồm: (x1, y1, x2, y2)

3.2 Vấn đề: Nhiều người + nhiều đồ vật

      -  YOLO phát hiện từng đối tượng riêng lẻ, dẫn đến khó khăn:

      - Với các hành động nhóm (hugging, fighting, kissing) → cần xem nhiều người cùng lúc

      - Với các hành động có đồ vật → phải gắn đồ vật vào đúng người

      - Nếu không xử lý, mô hình phân loại sẽ nhận một vùng người rời rạc → dự đoán sai hành động.

3.3 Cluster Person Boxes – Giải quyết ngữ cảnh hành động

      - Đề tài sử dụng hàm cluster_person_boxes với hai mục tiêu:

            + Gom các box người thành một cụm

            + Tính khoảng cách giữa tâm box

            + Gom vào cùng cluster nếu khoảng cách < dist_thresh
            
            + Ví dụ: 2 người ôm nhau → 1 cluster

      - Gắn các object-box vào đúng cụm người

            + Dựa trên khoảng cách person–object

            + Tạo ra vùng ảnh chứa đầy đủ ngữ cảnh hành động

            +  Ví dụ: Người + laptop → crop chung

3.4 EfficientNetB1 – Mô hình Phân loại hành động

      - Mô hình được fine-tune theo phương pháp transfer learning với 2 giai đoạn như sau:

            + Giai đoạn 1: đóng băng toàn bộ layer của NetB1 và thêm một lớp cuối vào mô hình để phù hợp với dữ liệu.

            + Giai đoạn 2: Mở 20% layer cuối của mô hình => sử dụng tham số của mô hình pretrained NetB1 để đạt kết quả tốt hơn.

      - Ý nghĩa 2 giai đoạn:
            + Kết quả của Giai đoạn 1 cung cấp một bộ trọng số khởi tạo tốt (well-initialized weights) cho lớp phân loại.
            
		+ Nhờ đó, khi bước sang Giai đoạn 2, mô hình tránh được hiện tượng "gradient exploding" (bùng nổ gradient) có thể xảy ra nếu huấn luyện toàn bộ mạng ngay từ đầu với trọng số ngẫu nhiên. Quá trình Fine-tuning nhờ vậy diễn ra nhanh hơn, ổn định hơn và đạt điểm hội tụ tốt hơn.

      - Kết quả: 

<img width="727" height="303" alt="image" src="https://github.com/user-attachments/assets/df68c6ff-7c3d-4999-90d4-40a058fb21a4" />


### 🏗 4. Kiến trúc tổng thể

Ảnh / Camera

      ↓
	  
YOLOv8n (Phát hiện người + đồ vật)

      ↓
	  
Cluster person/object boxes

      ↓
	  
Crop vùng cụm người

      ↓
	  
EfficientNetB1 (Phân loại hành động)

      ↓
	  
Trả kết quả


	- Thành phần chính:

		+ backend/recognize.py – YOLO + cluster + crop + phân loại

		+ backend/main.py – FastAPI server

		+ frontend/ – HTML/CSS/JS

### 📂 5. Cấu trúc thư mục

backend/

│── main.py               # FastAPI server

│── recognize.py          # YOLO detection, clustering, action recognition

│── output_model/         # EfficientNetB1 đã train

│── training_process/     # Notebook huấn luyện

frontend/

│── index.html

│── style.css

│── script.js

requirements.txt

README.md

### 🚀 6. Cài đặt & Chạy dự án

1️⃣ Cài thư viện: pip install -r requirements.txt

2️⃣ Chạy backend FastAPI: uvicorn backend.main:app --reload

3️⃣ Mở giao diện web: frontend/index.html

### 🖼 7. Kết quả

	- Nhận dạng ảnh và camera realtime

	- Phân loại tốt 15 hành động

	- Hoạt động ổn định với trường hợp nhiều người + nhiều đồ vật

Clustering cải thiện độ chính xác với các hành động nhóm
