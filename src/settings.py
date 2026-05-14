from pathlib import Path

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
PLANT_VILLAGE_DIR = DATA_DIR / "plant_village"
FIELD_PHOTOS_DIR = DATA_DIR / "field_photos"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

KERAS_MODEL_PATH = MODELS_DIR / "best_model.keras"
H5_MODEL_PATH = MODELS_DIR / "best_model.h5"
WEIGHTS_PATH = MODELS_DIR / "best_model.weights.h5"
QUANTIZED_MODEL_PATH = MODELS_DIR / "model_quantized.tflite"
DYNAMIC_MODEL_PATH = MODELS_DIR / "model_dynamic.tflite"
LABELS_PATH = MODELS_DIR / "label2id.json"

LOCKED_CLASSES = [
    "Corn_(maize)___Common_rust_",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

CLASS_DISPLAY_NAMES = {
    "Corn_(maize)___Common_rust_": "Corn Common Rust",
    "Pepper,_bell___Bacterial_spot": "Pepper Bell Bacterial Spot",
    "Pepper,_bell___healthy": "Pepper Bell Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Potato Healthy",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___healthy": "Tomato Healthy",
}

DISEASE_INFO = {
    "Corn_(maize)___Common_rust_": {
        "description": "Small, powdery, reddish-brown pustules appear on both leaf surfaces.",
        "action": "Apply a rust fungicide early and remove severely infected leaves.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Water-soaked spots turn dark brown on leaves and fruit.",
        "action": "Use a copper-based bactericide and avoid overhead irrigation.",
    },
    "Pepper,_bell___healthy": {
        "description": "The leaf appears healthy with no clear disease symptoms.",
        "action": "Continue regular watering, nutrition, and field monitoring.",
    },
    "Potato___Early_blight": {
        "description": "Dark target-like spots with concentric rings appear on older leaves.",
        "action": "Remove infected foliage and apply a suitable fungicide.",
    },
    "Potato___Late_blight": {
        "description": "Large, irregular, water-soaked lesions can spread rapidly.",
        "action": "Spray a late-blight fungicide immediately and destroy badly infected plants.",
    },
    "Potato___healthy": {
        "description": "The leaf appears healthy with no clear disease symptoms.",
        "action": "Continue regular crop care and monitor for new symptoms.",
    },
    "Tomato___Early_blight": {
        "description": "Dark concentric rings usually begin on lower, older leaves.",
        "action": "Improve airflow, remove infected leaves, and apply fungicide.",
    },
    "Tomato___Late_blight": {
        "description": "Dark, water-soaked lesions appear on leaves and stems and spread quickly.",
        "action": "Apply a copper-based or late-blight fungicide and remove infected parts.",
    },
    "Tomato___Leaf_Mold": {
        "description": "Yellow patches appear on the upper leaf surface with olive mold underneath.",
        "action": "Reduce humidity, improve ventilation, and apply a fungicide.",
    },
    "Tomato___healthy": {
        "description": "The leaf appears healthy with no clear disease symptoms.",
        "action": "Continue regular watering, fertilization, and field monitoring.",
    },
}
