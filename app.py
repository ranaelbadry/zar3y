import base64
import io
import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Zar3y - Crop Disease Detection",
    page_icon="🌿",
    layout="centered",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

st.title("🌿 Zar3y")
st.caption("Crop disease detection from phone photos")

with st.container():
    st.markdown(
        "Hassan sees suspicious yellow spots on a tomato leaf and needs a quick, explainable decision before spraying."
    )

tab_upload, tab_camera = st.tabs(["Upload Image", "Use Camera"])

uploaded_file = None
with tab_upload:
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

with tab_camera:
    camera_file = st.camera_input("Take a photo of the leaf")
    if camera_file is not None:
        uploaded_file = camera_file

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input photo", use_container_width=True)

    with st.spinner("Analyzing leaf photo..."):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        buf.seek(0)

        try:
            response = requests.post(
                BACKEND_URL,
                files={"file": ("leaf.jpg", buf, "image/jpeg")},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend on localhost:8000.")
            st.code("uvicorn backend.main:app --reload --port 8000")
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    confidence = float(data["confidence"])
    if confidence >= 80:
        confidence_label = "High confidence"
    elif confidence >= 60:
        confidence_label = "Medium confidence"
    else:
        confidence_label = "Low confidence"

    st.divider()
    st.subheader(data["class"])
    st.metric("Confidence", f"{confidence:.1f}%", confidence_label)

    overlay = data.get("gradcam_overlay")
    if overlay:
        st.image(
            Image.open(io.BytesIO(base64.b64decode(overlay))),
            caption="Grad-CAM overlay",
            use_container_width=True,
        )
    else:
        st.warning("Grad-CAM overlay was not available for this run.")

    st.info(f"Symptoms: {data['description']}")
    st.success(f"Next step: {data['action']}")
    st.caption(f"Backend latency: {data.get('latency_ms', 'n/a')} ms")
