# Zar3y - Crop Disease Detection

Zar3y is a Streamlit + FastAPI demo for phone-photo crop disease detection across the 10 locked PlantVillage classes from the project brief. The app returns the predicted class, confidence score, plain-language symptoms, next-step guidance, and a Grad-CAM overlay for explainability.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run The Demo

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`.

## Training And Evaluation

```powershell
.\.venv\Scripts\python.exe src\data_prep.py
.\.venv\Scripts\python.exe src\train.py
.\.venv\Scripts\python.exe src\evaluate.py
.\.venv\Scripts\python.exe src\grad_cam.py
```

Dataset version: PlantVillage subset from Kaggle, filtered to the 10 locked classes.

Base model: Keras `MobileNetV3Small`, ImageNet pretrained, frozen first phase then last 30 layers fine-tuned.

Split seed: `42`, stratified `70/15/15`.

## Class Counts

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Corn_(maize)___Common_rust_ | 834 | 179 | 179 | 1192 |
| Pepper,_bell___Bacterial_spot | 698 | 149 | 150 | 997 |
| Pepper,_bell___healthy | 1035 | 222 | 221 | 1478 |
| Potato___Early_blight | 700 | 150 | 150 | 1000 |
| Potato___healthy | 106 | 23 | 23 | 152 |
| Potato___Late_blight | 700 | 150 | 150 | 1000 |
| Tomato___Early_blight | 700 | 150 | 150 | 1000 |
| Tomato___healthy | 1114 | 239 | 238 | 1591 |
| Tomato___Late_blight | 1336 | 286 | 287 | 1909 |
| Tomato___Leaf_Mold | 666 | 143 | 143 | 952 |

`Potato___healthy` is under 500 training images, so class-weighted loss is used.

## Results

Held-out test metrics from `outputs/eval_report.json`:

| Metric | Value |
|---|---:|
| Accuracy | 95.92% |
| Macro F1 | 95.00% |
| Macro Precision | 94.52% |
| Macro Recall | 96.03% |

OOD field-photo result from `outputs/ood_report.json`: `21/40` correct, top-1 accuracy `52.5%`. This is intentionally reported honestly because field photos are much harder than PlantVillage images.

## Quantization Note

The repository includes `models/model_dynamic.tflite`, but that file was created with a different TensorFlow/TFLite runtime and fails on this Windows Python 3.12 environment with a `FULLY_CONNECTED version 12` opcode error. The backend therefore falls back to `models/best_model.keras` for local demo inference and Grad-CAM.

For the INT8 deliverable, run `src/quantize_tflite.py` in Python 3.10 or 3.11, matching the project brief environment. Python 3.12 + TensorFlow 2.16 hit Keras export issues on this machine.

## Artifacts

- `models/best_model.keras`
- `models/best_model.h5`
- `models/best_model.weights.h5`
- `models/model_dynamic.tflite`
- `outputs/augmentation_samples.png`
- `outputs/confusion_matrix.png`
- `outputs/training_curves.png`
- `outputs/eval_report.json`
- `outputs/ood_report.json`
- `outputs/grad_cam_examples/`
- `outputs/failure_cases/`
