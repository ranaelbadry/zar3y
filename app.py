import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Zar3y - Crop Disease Detection",
    page_icon="🌿",
    layout="centered",
)

BACKEND_URL = "http://localhost:8000/predict"

st.title("🌿 Zar3y — Crop Disease Detection")
st.markdown("Upload or take a photo of a plant leaf and we'll identify any disease instantly.")

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

uploaded_file = None

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        key="upload",
    )

with tab2:
    camera_file = st.camera_input("Take a photo of the leaf", key="camera")
    if camera_file:
        uploaded_file = camera_file

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        try:
            response = requests.post(
                BACKEND_URL,
                files={"file": ("leaf.jpg", buf, "image/jpeg")},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            confidence = data["confidence"]

            if confidence >= 80:
                badge = "🟢 High Confidence"
            elif confidence >= 60:
                badge = "🟡 Medium Confidence"
            else:
                badge = "🔴 Low Confidence"

            st.divider()
            st.subheader(f"Result: {data['class']}")
            st.metric("Confidence", f"{confidence}%", delta=badge)

            st.info(f"**Symptoms:** {data['description']}")
            st.success(f"**Recommended Action:** {data['action']}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend. Make sure it is running on localhost:8000")
            st.code("uvicorn backend.main:app --reload --port 8000")

        except Exception as e:
            st.error(f"Something went wrong: {e}")