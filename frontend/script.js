const uploadBtn = document.getElementById('uploadMode');
const cameraBtn = document.getElementById('cameraMode');
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const camera = document.getElementById('camera');
const predictBtn = document.getElementById('predictBtn');
const resultImage = document.getElementById('resultImage');
const statusText = document.getElementById('statusText');

let cameraStream = null;
let isPredicting = false;

// ------------------ Reset khung kết quả ------------------
function resetResultBox() {
  const resultBox = document.querySelector('.two-column .image-column:nth-child(2) .image-box');
  
  resultImage.src = '';
  resultImage.style.display = 'none';
  statusText.textContent = '';
  resultBox.querySelectorAll('canvas, .overlay').forEach(el => el.remove());
  resultBox.style.background = '#fafafa';
}

// ------------------ Chuyển chế độ Upload ------------------
uploadBtn.onclick = () => {
  uploadBtn.classList.add('active');
  cameraBtn.classList.remove('active');

  imageInput.style.display = 'block';
  predictBtn.style.display = 'inline-block';
  camera.style.display = 'none';

  previewImage.src = '';
  previewImage.style.display = 'none';
  resetResultBox();

  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
};

// ------------------ Chuyển chế độ Camera ------------------
cameraBtn.onclick = async () => {
  cameraBtn.classList.add('active');
  uploadBtn.classList.remove('active');

  imageInput.style.display = 'none';
  predictBtn.style.display = 'none';
  previewImage.style.display = 'none';
  resultImage.style.display = 'none';
  statusText.textContent = '';
  resetResultBox();

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    camera.srcObject = cameraStream;
    camera.style.display = 'block';
    startRealtimePrediction();
  } catch (err) {
    alert("Không thể mở camera: " + err.message);
  }
};

// ------------------ Upload ảnh ------------------
imageInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImage.src = ev.target.result;
    previewImage.style.display = 'block';
  };
  reader.readAsDataURL(file);
});

// ------------------ Nhận dạng ảnh tĩnh ------------------
predictBtn.onclick = async () => {
  const file = imageInput.files[0];
  if (!file) return alert("Vui lòng chọn ảnh!");

  const formData = new FormData();
  formData.append("file", file);
  statusText.textContent = " Đang nhận dạng...";

  try {
    const res = await fetch("http://localhost:8000/predict-image", { method: "POST", body: formData });
    const data = await res.json();

    if (!data.img_base64 || data.img_base64 === "None") {
      statusText.textContent = data.message || " Không có người trong ảnh tải lên!";
      resultImage.style.display = 'none';
      return;
    }

    resultImage.src = `data:image/jpeg;base64,${data.img_base64}`;
    resultImage.style.display = 'block';
    statusText.textContent = data.message || " Nhận dạng thành công!";
  } catch (err) {
    console.error(err);
    statusText.textContent = " Lỗi khi gửi yêu cầu.";
  }
};

// ------------------ Nhận dạng camera realtime ------------------
function startRealtimePrediction() {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  async function sendFrame() {
    setTimeout(sendFrame, 200); // gửi frame tiếp theo sau 200ms

    if (isPredicting || !cameraStream || camera.videoWidth === 0 || camera.videoHeight === 0) return;
    isPredicting = true;
    canvas.width = camera.videoWidth;
    canvas.height = camera.videoHeight;
    ctx.drawImage(camera, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async blob => {
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");
      try {
        const res = await fetch("http://localhost:8000/predict-camera", { method: "POST", body: formData });
        const data = await res.json();
        if (data.img_base64 && data.img_base64 !== "None") {
          resultImage.src = `data:image/jpeg;base64,${data.img_base64}`;
          resultImage.style.display = 'block';
        } else {
          statusText.textContent = data.message || " Không có người trong khung hình!";
          resultImage.style.display = 'none';
        }
      } catch (err) {
        console.error(err);
      } finally {
        isPredicting = false;
      }
    }, "image/jpeg");
  }

  sendFrame();
}




