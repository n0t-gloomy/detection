import os

import streamlit as st
import torch
from PIL import Image

from gradcam_cls import gradcam_overlay, list_debug_conv_layer_names


@st.cache_resource
def find_model_files(directory="weights"):
    """Find all .pt model files in directory."""
    models = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith(".pt"):
                models.append(file)
    return sorted(models)


@st.cache_resource
def load_yolo_model(path):
    """Load YOLO model (cached)."""
    try:
        from ultralytics import YOLO

        return YOLO(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


st.set_page_config(
    page_title="YOLO Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 YOLO Classification")
st.write("Upload an image to classify and optionally visualize **Grad-CAM** for the top prediction.")

st.sidebar.header("⚙️ Settings")

available_models = find_model_files("weights")

if available_models:
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        help="Choose a YOLO .pt model to use",
    )
    model_path = f"weights/{selected_model}"
else:
    st.sidebar.warning("⚠️ No .pt models found in weights/ directory")
    model_path = st.sidebar.text_input(
        "Model Path",
        value="weights/oral.pt",
        help="Path to your YOLO .pt model file",
    )

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

show_gradcam = st.sidebar.checkbox(
    "Show Grad-CAM (top-1)",
    value=False,
    help="Explains the predicted class only. Uses the same resize/scale as typical YOLO-cls (RGB, imgsz from checkpoint, ÷255).",
)

cam_layer_name: str | None = None
cam_alpha = 0.45
if os.path.exists(model_path):
    _warm = load_yolo_model(model_path)
    if _warm is not None:
        with st.sidebar.expander("Advanced: Grad-CAM layer (debug)", expanded=False):
            _names = list_debug_conv_layer_names(_warm)
            _choice = st.selectbox(
                "Convolution layer",
                options=["Auto (recommended)"] + _names,
                index=0,
                help="Override only if you are debugging heatmaps. Default picks the last spatial conv map.",
            )
            cam_layer_name = None if _choice.startswith("Auto") else _choice
            cam_alpha = st.slider("Overlay strength", 0.2, 0.8, 0.45, 0.05)

st.sidebar.divider()
st.sidebar.caption("Model is cached after first load for faster inference.")

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: `{model_path}`")
        st.stop()

    model = load_yolo_model(model_path)
    if model is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.size[0]} × {image.size[1]} px")

    with st.spinner("🔄 Running inference..."):
        results = model(image)
        result = results[0]

    if hasattr(result, "probs") and result.probs is not None:
        probs = result.probs.data
        top1_idx = int(probs.argmax().item())
        top1_conf = float(probs[top1_idx].item())
        top1_name = result.names[top1_idx]

        with col2:
            st.subheader("🎯 Prediction")
            st.metric("Class", top1_name)
            st.metric("Confidence", f"{top1_conf:.1%}")

            st.subheader("Top Predictions")
            top_k = min(5, len(result.names))
            top_confs, top_indices = torch.topk(probs.squeeze(), k=top_k)

            for conf, idx in zip(top_confs, top_indices):
                class_name = result.names[idx.item()]
                conf_val = float(conf.item())
                if conf_val >= conf_threshold:
                    st.progress(conf_val, text=f"{class_name}: {conf_val:.1%}")

        if show_gradcam:
            st.subheader("Grad-CAM (top-1)")
            st.caption(
                f"Explaining **{top1_name}** (class index {top1_idx}). "
                "Input is resized to the model `imgsz` (from your `.pt`, else 640), then overlaid on the original upload."
            )
            with st.spinner("Generating Grad-CAM…"):
                try:
                    overlay_rgb = gradcam_overlay(
                        model,
                        image,
                        top1_class_idx=top1_idx,
                        layer_name=cam_layer_name,
                        alpha=float(cam_alpha),
                    )
                    st.image(overlay_rgb, use_container_width=True, caption="Grad-CAM overlay (jet colormap)")
                except Exception as e:
                    st.error("Grad-CAM failed — try another layer under Advanced, or run on GPU if you see half-precision/CPU errors.")
                    st.exception(e)
    else:
        with col2:
            st.warning("⚠️ No classification probabilities found.")
            if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes):
                st.info("This looks like a **detection** model (not classification). Detections found:")
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = result.names[cls_id]
                    conf = float(box.conf.item())
                    st.write(f"• `{cls_name}` — {conf:.1%}")
            else:
                st.info("Make sure your model was trained for **classification** (not detection/segmentation).")

else:
    st.info("👆 Upload an image to get started")

with st.expander("ℹ️ About this app"):
    st.markdown(
        """
### Features
- **YOLO Classification** inference via Ultralytics
- **Grad-CAM** for the **top-1** class (optional)
- **Advanced layer override** (hidden in sidebar) for debugging only

### Grad-CAM notes
- Uses logits for the predicted class index and a convolutional feature map (auto layer unless you override).
- Preprocessing matches common YOLO-cls defaults: **RGB**, **square resize to `imgsz`**, **÷255**. `imgsz` is read from the checkpoint when available (your cloud training used **640**).

### Notices
- This program is a non-commercial product. Do not use for commercial uses. All rights reserved.
"""
    )

st.divider()
st.subheader("📦 Available Models")

if available_models:
    st.success(f"Found {len(available_models)} model(s):")
    for model_file in available_models:
        st.write(f"• `{model_file}`")
else:
    st.warning("No models found in `weights/` directory")
    st.info("Place your YOLO `.pt` files in the `weights/` folder.")