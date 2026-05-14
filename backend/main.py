import io
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from inference import predict

app = FastAPI(title="Zar3y - Crop Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    started = time.perf_counter()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = predict(image)

    return {
        "label": result["label"],
        "class": result["class"],
        "confidence": round(result["confidence"] * 100, 1),
        "description": result["description"],
        "action": result["action"],
        "gradcam_overlay": result["gradcam_overlay"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "model_path": result["model_path"],
    }
