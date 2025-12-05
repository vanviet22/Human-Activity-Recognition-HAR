# Dự đoán hành động từ ảnh qua 2 phương thức là qua tải ảnh hoặc mở cam.
# Sử dụng YOLOv8 để phát hiện người trong ảnh => crop vùng ảnh và predict.
import tensorflow as tf
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.layers import BatchNormalization, Dense
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array
import base64
import os
import io
import joblib
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load YOLO
yolo_model = YOLO("yolov8n.pt")

# Load class names  của hành động
class_names = joblib.load(os.path.join(base_dir, "output_model", "class_names.pkl"))

# Load model hành động
num_classes = class_names.__len__()
model_path = os.path.join(base_dir, "output_model", "best_model_of_EffB1.keras")
model = tf.keras.models.load_model(model_path, compile=False)
print("✅ Model fine-tuned EfficientNetB1 đã load xong!")
print(model.layers[-1].activation)

# Tiền xử lý ảnh
def base_transform(img_path_or_array):
    """
    Pipeline inference cho EfficientNetB1:
    - Resize 160x160
    - preprocess_input
    """
    # mở ảnh
    if isinstance(img_path_or_array, str):
        img = Image.open(img_path_or_array).convert('RGB')
    else:
        img = Image.fromarray(img_path_or_array).convert('RGB')
    # resize
    img = img.resize((160,160))
    # to array
    img_array = img_to_array(img)
    # add batch dim
    img_array = np.expand_dims(img_array, axis=0)
    # preprocess
    img_array = preprocess_input(img_array)
    
    return img_array

def cluster_person_boxes(persons_locat, dist_thresh=100):
    """
    Nhóm các box người gần nhau thành 1 cluster
    dist_thresh: khoảng cách tối đa giữa center để gộp (pixels)
    """
    centers = [((int(box.xyxy[0][0]+box.xyxy[0][2])/2),
                (int(box.xyxy[0][1]+box.xyxy[0][3])/2))
               for box in persons_locat]

    clusters = []

    for i, c in enumerate(centers):
        added = False
        for cluster in clusters:
            if any(np.linalg.norm(np.array(c)-np.array(cc)) < dist_thresh for cc in cluster):
                cluster.append(c)
                added = True
                break
        if not added:
            clusters.append([c])

    # Trả về index các box trong mỗi cluster
    clustered_boxes = []
    for cluster in clusters:
        cluster_indices = [i for i, c in enumerate(centers) if c in cluster]
        clustered_boxes.append(cluster_indices)

    return clustered_boxes


def recognize_action(image):
    results = yolo_model(image)
    persons_locat = [box for box in results[0].boxes if yolo_model.names[int(box.cls[0])] == "person"]
    if not persons_locat:
        return {"img_base64": None}

    OBJECTS = ["laptop", "cell phone", "bottle", "cup", "book", "bicycle","person"]

    box_locat = [
        box for box in results[0].boxes
        if yolo_model.names[int(box.cls[0])] in OBJECTS
    ]

    # Cluster các người gần nhau
    clusters = cluster_person_boxes(box_locat, dist_thresh=100)
    preds_info = []

    for cluster in clusters:
        # Bounding box bao quanh toàn bộ cluster
        x1 = min([int(box_locat[i].xyxy[0][0]) for i in cluster])
        y1 = min([int(box_locat[i].xyxy[0][1]) for i in cluster])
        x2 = max([int(box_locat[i].xyxy[0][2]) for i in cluster])
        y2 = max([int(box_locat[i].xyxy[0][3]) for i in cluster])

        crop_region = image[y1:y2, x1:x2]
        input_tensor = base_transform(crop_region)

        # Dự đoán
        pred = model.predict(input_tensor)
        probs = pred[0]
        pred_class = np.argmax(probs)
        conf = probs[pred_class] * 100

        if conf < 10:
            pred_label = "Action not trained"
        else:
            pred_label = f"{class_names[pred_class]} "

        preds_info.append((x1, y1, x2, y2, pred_label))

    # Vẽ kết quả lên ảnh
    for x1, y1, x2, y2, pred_label in preds_info:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, pred_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(f"Hành động: {pred_label} ({conf:.1f}%)")

    # Encode ảnh trả về base64
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {"img_base64": img_base64}


# Nhận diện qua ảnh tải lên
def recognize_from_image(file_bytes: bytes = None, image_path: str = None):
    """
    Nếu upload từ frontend, truyền file_bytes
    Nếu dùng file path trên server, truyền image_path
    """
    if file_bytes is not None:
        np_img = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    elif image_path is not None:
        image = cv2.imread(image_path)
    else:
        raise ValueError("Cần truyền file_bytes hoặc image_path")

    if image is None:
        return {"results": [], "message": "Không đọc được ảnh"}

    return recognize_action(image)



# Nhận diện qua camera (frame base64 từ frontend)
def recognize_from_camera(frame_data):
    """
    Xử lý một frame ảnh được gửi từ frontend qua camera (base64)
    và trả về kết quả nhận diện hành động.
    """
    # Giải mã base64 thành ảnh
    if isinstance(frame_data, str):
        # Loại bỏ phần prefix như "data:image/jpeg;base64,"
        frame_data = frame_data.split(",")[-1]
        frame_data = base64.b64decode(frame_data)

    np_img = np.frombuffer(frame_data, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    return recognize_action(image)

# if __name__ == "__main__":
#     # Test nhận diện từ ảnh
#     image_path = r"C:\Users\ADMIN\Downloads\2-15814112319141424056295-1581419177707-1581419177707591955539.jpg"
#     result = recognize_from_image(image_path=image_path)
    # print("Kết quả nhận diện từ ảnh:", result)

#     # # Test nhận diện từ camera (giả lập với một ảnh)
#     with open(image_path, "rb") as img_file:
#         img_bytes = img_file.read()
#         img_base64 = base64.b64encode(img_bytes).decode('utf-8')
#         frame_data = f"data:image/jpeg;base64,{img_base64}"

#     result_cam = recognize_from_camera(frame_data)
#     print("Kết quả nhận diện từ camera:", result_cam)