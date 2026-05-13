import sys
import io
sys.path.append(".")

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference import predict

app = FastAPI(title="Zar3y - Crop Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISEASE_INFO = {
    "Corn Common Rust": {
        "description": "Orange to brown pustules appear on both leaf surfaces.",
        "action": "Apply a suitable fungicide for rust diseases as soon as possible.",
    },
    "Pepper Bell Bacterial Spot": {
        "description": "Small water-soaked spots that turn dark brown on leaves and fruit.",
        "action": "Spray a copper-based bactericide and avoid overhead irrigation.",
    },
    "Pepper Bell Healthy": {
        "description": "The plant appears healthy with no signs of disease.",
        "action": "Continue regular watering and fertilization.",
    },
    "Potato Early Blight": {
        "description": "Dark brown spots with concentric rings on older leaves.",
        "action": "Apply a fungicide and remove infected lower leaves.",
    },
    "Potato Late Blight": {
        "description": "A serious disease causing dark water-soaked lesions that spread rapidly.",
        "action": "Spray a fungicide immediately and avoid overhead irrigation.",
    },
    "Potato Healthy": {
        "description": "The plant appears healthy with no signs of disease.",
        "action": "Continue regular care and monitoring.",
    },
    "Tomato Early Blight": {
        "description": "Dark brown spots with concentric rings, usually on older leaves.",
        "action": "Apply a fungicide and ensure good airflow between plants.",
    },
    "Tomato Healthy": {
        "description": "The plant appears healthy with no signs of disease.",
        "action": "Continue regular watering and fertilization.",
    },
    "Tomato Late Blight": {
        "description": "Dark water-soaked lesions on leaves and stems that spread quickly.",
        "action": "Apply a copper-based fungicide immediately and remove infected parts.",
    },
    "Tomato Leaf Mold": {
        "description": "Yellow patches on upper leaf surface with olive-green mold beneath.",
        "action": "Reduce humidity, improve ventilation, and apply a fungicide.",
    },
}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    result = predict(image)

    class_name = result["class"]
    confidence = round(result["confidence"] * 100, 1)

    info = DISEASE_INFO.get(class_name, {
        "description": "No information available.",
        "action": "Please consult an agricultural expert.",
    })

    return {
        "class": class_name,
        "confidence": confidence,
        "description": info["description"],
        "action": info["action"],
    }