import streamlit as st
import numpy as np
import base64
import os
from PIL import Image
import models


# --- Model Data ---
data = {
    "Oral Infections V1 (Canker Sores, Cold Sores, Mouth Cancer, Normal, Oral Thrush)": {
        "class_names": ['Canker Sores', 'Cold Sores', 'Mouth Cancer', 'Normal', 'Oral Thrush'],
        "weights_name": "exp-3.pt",
    },
    "Oral Infections V2 (Canker Sores, Herpes, Normal, Oral Cancer, Oral Thrush)": {
        "class_names": ['Canker Sores', 'Herpes', 'Normal', 'Oral Cancer', 'Oral Thrush'],
        "weights_name": "oral.pt",
    }
}

# --- Page Config ---
st.set_page_config(page_title="Oral Image Classifier", layout="wide")


# --- Session State ---
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# --- Background Image ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image_path = "img.jpg"

if os.path.exists(bg_image_path):
    bg_image_encoded = get_base64_image(bg_image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg_image_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# --- Title ---
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #f0f0f0;
        font-weight: bold;
        margin-top: -60px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Oral Image Classification</div>', unsafe_allow_html=True)


# --- Model Selection ---
model_name = st.selectbox("🔍 Choose a Classification Model", list(data.keys()))
model_data = data[model_name]


# --- Upload ---
uploaded_file = st.file_uploader("📤 Upload/Drop an Image", type=["jpg", "jpeg", "png", "webp"])


if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1.2, 1, 1.2], gap="large")

    # --- Show Image ---
    with col1:
        st.image(img, use_container_width=True, caption="📥 Uploaded Image")

    # --- Button ---
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("""
            <style>
            div.stButton > button {
                margin-left: 30%;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("🚀 Run Classification"):
            with st.spinner("Classifying..."):
                label, score = models.classifier(
                    img,
                    model_data["weights_name"],
                    model_data["class_names"]
                )
                st.session_state.prediction = (label, score)

    # --- Result ---
    with col3:
        if st.session_state.prediction:
            label, score = st.session_state.prediction

            st.markdown(f"""
                <br><br>
                <h3>📷 Prediction: <code>{label}</code></h3>
                <h4>🔢 Confidence: <code>{score*100:.1f}%</code></h4>
                <h5>🧠 Model Used: <code>{model_data['weights_name']}</code></h5>
            """, unsafe_allow_html=True)
