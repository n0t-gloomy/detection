import streamlit as st
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def find_model_files(directory="weights"):
    """Find all .pt model files in directory"""
    models = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith(".pt"):
                models.append(file)
    return sorted(models)


@st.cache_resource
def load_yolo_model(path):
    """Load YOLO model"""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="YOLO Classification with Grad-CAM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 YOLO Classification with Grad-CAM")
st.write("Upload an image to classify and visualize model decisions")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Settings")

# Model selection dropdown
available_models = find_model_files("weights")

if available_models:
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        help="Choose a YOLO .pt model to use"
    )
    model_path = f"weights/{selected_model}"
else:
    st.sidebar.warning("⚠️ No .pt models found in weights/ directory")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="weights/oral.pt",
        help="Path to your YOLO .pt model file"
    )

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

st.sidebar.divider()
st.sidebar.caption("Model will be cached after first load for faster inference")

# ============================================================================
# MAIN APP
# ============================================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)

    # Ensure RGB (drop alpha channel if present)
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_np = np.array(image)

    # Create columns
    col1, col2 = st.columns(2)

    # Display original image
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        st.caption(f"Size: {image.size[0]} × {image.size[1]}px")

    # Check model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.info(f"Looking for model at: `{model_path}`")
        st.stop()

    # Load model
    model = load_yolo_model(model_path)
    if model is None:
        st.stop()

    # Inference
    with st.spinner("🔄 Running inference..."):
        results = model(image)
        result = results[0]

    # Extract predictions
    if hasattr(result, 'probs') and result.probs is not None:
        probs = result.probs.data
        top1_idx = probs.argmax().item()
        top1_conf = probs[top1_idx].item()
        top1_name = result.names[top1_idx]

        # Display predictions
        with col2:
            st.subheader("🎯 Prediction")
            st.metric("Class", top1_name)
            st.metric("Confidence", f"{top1_conf:.1%}")

            # Top-K predictions
            st.subheader("Top Predictions")
            top_k = min(5, len(result.names))
            top_confs, top_indices = torch.topk(probs.squeeze(), k=top_k)

            for conf, idx in zip(top_confs, top_indices):
                class_name = result.names[idx.item()]
                conf_val = conf.item()
                if conf_val >= conf_threshold:
                    st.progress(
                        conf_val,
                        text=f"{class_name}: {conf_val:.1%}"
                    )


    else:
        with col2:
            st.warning("⚠️ No classification probabilities found.")
            # Try to show detection boxes if this is a detection model
            if hasattr(result, 'boxes' ) and result.boxes is not None and len(result.boxes):
                st.info("This looks like a **detection** model (not classification). Detections found:")
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = result.names[cls_id]
                    conf = box.conf.item()
                    st.write(f"• `{cls_name}` — {conf:.1%}")
            else:
                st.info("Make sure your model was trained for **classification** (not detection/segmentation).")

else:
    st.info("👆 Upload an image to get started")

# ============================================================================
# ABOUT EXPANDER
# ============================================================================

with st.expander("ℹ️ About this app"):
    st.markdown("""
### Features
- **YOLO Classification**: Fast, accurate predictions
- **Model Selection**: Choose from available models via dropdown
- **Top-K Predictions**: See alternative classifications with confidence scores
- **Confidence Threshold**: Filter out low-confidence predictions

### How it works
1. Select a model from the dropdown (or specify path)
2. Upload an image (JPG, JPEG, PNG, or WEBP)
3. Model classifies the image and returns probabilities
4. Top predictions are displayed with a progress bar

### Tips
- **Confidence Threshold**: Raise it to hide low-confidence classes
- **Different Models**: Different .pt files may be trained on different classes
- **Image formats**: RGBA/palette PNGs are auto-converted to RGB

### Supported Models
- YOLO Classification models (.pt format)
- Standard YOLO sizes: nano (n), small (s), medium (m), large (l), xlarge (x)
""")

# ============================================================================
# AVAILABLE MODELS SECTION
# ============================================================================

st.divider()
st.subheader("📦 Available Models")

if available_models:
    st.success(f"Found {len(available_models)} model(s):")
    for model_file in available_models:
        st.write(f"• `{model_file}`")
else:
    st.warning("No models found in `weights/` directory")
    st.info("Place your YOLO .pt files in the `weights/` folder")