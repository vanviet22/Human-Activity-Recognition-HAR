from fastapi import FastAPI, UploadFile, File
from recognize import recognize_from_image, recognize_from_camera
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)
# Nhận diện từ ảnh upload
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = recognize_from_image(file_bytes=contents)
    return JSONResponse(content=result)

# Nhận diện từ camera (frame blob)
@app.post("/predict-camera")
async def predict_camera(file: UploadFile = File(...)):
    contents = await file.read()
    result = recognize_from_camera(contents)
    return JSONResponse(content=result)

" lệnh chạy server trên terminal: uvicorn main:app --host 0.0.0.0 --port 8000 --reload"