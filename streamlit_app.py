import streamlit as st

import torch

import torch.nn.functional as F

import cv2

import numpy as np

from PIL import Image

from ultralytics import YOLO

import os

from pathlib import Path





# ============================================================================

# GRAD-CAM IMPLEMENTATION (Embedded)

# ============================================================================



class GradCAM:
    def __init__(self, model, target_layer = -1):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        # FIX: Handle list output from model
        if isinstance(output, list):
            output = output[0]  # Take the first element (usually the classification logits)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam, class_idx





def overlay_heatmap(image, heatmap, alpha=0.4):

    """Overlay heatmap on image"""

    h, w = image.shape[:2]

    heatmap = cv2.resize(heatmap, (w, h))

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

   

    result = cv2.addWeighted(

        image.astype(np.float32), 1 - alpha,

        heatmap_color.astype(np.float32), alpha,

        0

    )

    return result.astype(np.uint8)





def resize_image_for_yolo(image_array, target_size=640):

    """

    Resize image to YOLO-compatible size (divisible by 32)

   

    Args:

        image_array: numpy array (H, W, 3)

        target_size: target size (default 640)

   

    Returns:

        resized image array with dimensions divisible by 32

    """

    # Make sure dimensions are divisible by 32

    if target_size % 32 != 0:

        target_size = (target_size // 32) * 32

   

    # Resize using PIL for better quality

    image_pil = Image.fromarray(image_array)

    image_resized = image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)

   

    return np.array(image_resized)





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



show_gradcam = st.sidebar.checkbox(

    "Show Grad-CAM Visualization",

    value=True,

    help="Enable visual explanation of predictions"

)



heatmap_alpha = st.sidebar.slider(

    "Heatmap Opacity",

    min_value=0.0,

    max_value=1.0,

    value=0.5,

    step=0.05

)



conf_threshold = st.sidebar.slider(

    "Confidence Threshold",

    min_value=0.0,

    max_value=1.0,

    value=0.0,

    step=0.05

)



# Image size for Grad-CAM (must be divisible by 32)

grad_cam_size = st.sidebar.selectbox(

    "Grad-CAM Image Size",

    options=[320, 416, 512, 640, 704, 768],

    index=3,  # Default 640

    help="Larger = better quality but slower. Must be divisible by 32"

)



st.sidebar.divider()

st.sidebar.caption("Model will be cached after first load for faster inference")



# ============================================================================

# MAIN APP

# ============================================================================



# File uploader

uploaded_file = st.file_uploader(

    "📤 Upload an image",

    type=["jpg", "jpeg", "png", "webp"],

    help="Supported formats: JPG, JPEG, PNG, WEBP"

)



if uploaded_file is not None:

    # Load image

    image = Image.open(uploaded_file)

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

                st.metric("Class", top1_name, delta=None)

                st.metric("Confidence", f"{top1_conf:.1%}", delta=None)

               

                # Top 5 predictions

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

       

        # Grad-CAM visualization

        if show_gradcam:

            st.divider()

            st.subheader("🔍 Grad-CAM Visualization")

            st.write(

                "**Red regions** = high importance | **Blue regions** = low importance"

            )

           

            try:

                with st.spinner(f"Generating Grad-CAM (resizing to {grad_cam_size}×{grad_cam_size})..."):

                    # Resize image to YOLO-compatible size

                    image_resized = resize_image_for_yolo(image_np, target_size=grad_cam_size)

                   

                    # Prepare tensor

                    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                   

                    # Validate dimensions

                    if image_tensor.dim() != 4:

                        raise ValueError(f"Expected 4D tensor, got {image_tensor.dim()}D")

                   

                    b, c, h, w = image_tensor.shape

                    if h % 32 != 0 or w % 32 != 0:

                        raise ValueError(f"Image dimensions ({h}, {w}) not divisible by 32")

                   

                    device = next(model.parameters()).device

                    image_tensor = image_tensor.to(device)

                   

                    # Generate Grad-CAM with target class

                    grad_cam = GradCAM(model)

                    heatmap = grad_cam.generate(image_tensor, target_class=top1_idx)

                   

                    # Overlay on resized image

                    visualization = overlay_heatmap(image_resized, heatmap, alpha=heatmap_alpha)

                   

                    # Display

                    st.image(

                        Image.fromarray(visualization),

                        caption=f"Grad-CAM Heatmap ({grad_cam_size}×{grad_cam_size}) for '{top1_name}'",

                        use_column_width=True

                    )

                   

                    st.success("✅ Grad-CAM visualization generated successfully!")

                   

            except Exception as e:

                st.error(f"❌ Grad-CAM error: {str(e)}")

                st.info("**Troubleshooting:**")

                st.info("• Verify your model is a YOLO classification model (not detection)")

                st.info("• Try disabling Grad-CAM if the model architecture isn't supported")

                st.info("• Check that the model file isn't corrupted")



else:

    st.info("👆 Upload an image to get started")

   

    with st.expander("ℹ️ About this app"):

        st.markdown("""

        ### Features

        - **YOLO Classification**: Fast, accurate predictions

        - **Model Selection**: Choose from available models via dropdown

        - **Grad-CAM Visualization**: Understand model decisions

        - **Configurable Image Size**: Trade quality for speed

        - **Top-K Predictions**: See alternative classifications

        - **Confidence Display**: Know how certain the model is

       

        ### How it works

        1. Select a model from the dropdown (or specify path)

        2. Upload an image (JPG, JPEG, PNG, or WEBP)

        3. Model classifies the image

        4. Image is resized to a YOLO-compatible size (divisible by 32)

        5. Grad-CAM shows which regions influenced the decision

        6. Red = high importance, Blue = low importance

       

        ### Tips

        - **Image Size**: Larger size = better visualization but slower processing

        - **Heatmap Opacity**: Adjust for better visibility (0.3-0.7 recommended)

        - **Confidence Threshold**: Filter low-confidence predictions

        - **Different Models**: Try different models for different results

       

        ### Image Resizing

        Images are automatically resized to YOLO-compatible dimensions:

        - Must be divisible by 32 (e.g., 320, 416, 512, 640, 704, 768)

        - Maintains aspect ratio

        - Ensures Grad-CAM works properly

       

        ### Supported Models

        - YOLO Classification models (.pt format)

        - Standard YOLO sizes: nano (n), small (s), medium (m), large (l), xlarge (x)

        """)

   

    # Show available models

    st.divider()

    st.subheader("📦 Available Models")

    if available_models:

        st.success(f"Found {len(available_models)} model(s):")

        for model_file in available_models:

            st.write(f"• `{model_file}`")

    else:

        st.warning("No models found in `weights/` directory")

        st.info("Place your YOLO .pt files in the `weights/` folder")