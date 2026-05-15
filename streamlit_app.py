import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from pathlib import Path

# ============================================================================ #
# GRAD-CAM IMPLEMENTATION (Embedded)
# ============================================================================ #
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        # Using register_full_backward_hook for stability in modern PyTorch versions
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        # FIX: Explicitly extract the tensor from the output tuple/list
        if isinstance(output, (tuple, list)):
            output = output[0]  # The first element contains the main classification logits tensor
        elif hasattr(output, 'logits'):
            output = output.logits
            
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        
        # Create one-hot encoding for the target class safely using the isolated tensor
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to 0-1 range
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Convert tensor matrix back to a 2D numpy array layout
        cam_np = cam.squeeze().cpu().numpy()
        return cam_np, class_idx

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    h, w = image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(
        image.astype(np.float32), 1 - alpha,
        heatmap_color.astype(np.float32), alpha, 0
    )
    return result.astype(np.uint8)

def resize_image_for_yolo(image_array, target_size=640):
    """ Resize image to YOLO-compatible size (divisible by 32) """
    if target_size % 32 != 0:
        target_size = (target_size // 32) * 32
        
    image_pil = Image.fromarray(image_array)
    image_resized = image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return np.array(image_resized)

# ============================================================================ #
# UTILITY FUNCTIONS
# ============================================================================ #
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

# ============================================================================ #
# STREAMLIT APP
# ============================================================================ #
st.set_page_config(
    page_title="YOLO Classification with Grad-CAM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🧠 YOLO Classification with Grad-CAM")
st.write("Upload an image to classify and visualize model decisions")

# ============================================================================ #
# SIDEBAR CONFIGURATION
# ============================================================================ #
st.sidebar.header("⚙️ Settings")

# Ensure weights directory exists safely
if not os.path.exists("weights"):
    os.makedirs("weights")

available_models = find_model_files("weights")
if available_models:
    selected_model = st.sidebar.selectbox(
        "Select Model", options=available_models, help="Choose a YOLO .pt model to use"
    )
    model_path = f"weights/{selected_model}"
else:
    st.sidebar.warning("⚠️ No .pt models found in weights/ directory")
    model_path = st.sidebar.text_input(
        "Model Path", value="weights/oral.pt", help="Path to your YOLO .pt model file"
    )

show_gradcam = st.sidebar.checkbox(
    "Show Grad-CAM Visualization", value=True, help="Enable visual explanation of predictions"
)
heatmap_alpha = st.sidebar.slider(
    "Heatmap Opacity", min_value=0.0, max_value=1.0, value=0.5, step=0.05
)
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05
)

grad_cam_size = st.sidebar.selectbox(
    "Grad-CAM Image Size", options=[320, 416, 512, 640, 704, 768], index=3,
    help="Larger = better quality but slower. Must be divisible by 32"
)
st.sidebar.divider()
st.sidebar.caption("Model will be cached after first load for faster inference")

# ============================================================================ #
# MAIN APP
# ============================================================================ #
uploaded_file = st.file_uploader(
    "📤 Upload an image", type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB")) # Force RGB space
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.size[0]} × {image.size[1]}px")
        
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()
        
    model = load_yolo_model(model_path)
    if model is None:
        st.stop()
        
    with st.spinner("🔄 Running inference..."):
        results = model(image_np)
        result = results[0]
        
    if hasattr(result, 'probs') and result.probs is not None:
        probs = result.probs.data
        top1_idx = probs.argmax().item()
        top1_conf = probs[top1_idx].item()
        top1_name = result.names[top1_idx]
        
        with col2:
            st.subheader("🎯 Prediction")
            st.metric("Class", top1_name, delta=None)
            st.metric("Confidence", f"{top1_conf:.1%}", delta=None)
            
            st.subheader("Top Predictions")
            top_k = min(5, len(result.names))
            top_confs, top_indices = torch.topk(probs.squeeze(), k=top_k)
            for conf, idx in zip(top_confs, top_indices):
                class_name = result.names[idx.item()]
                conf_val = conf.item()
                if conf_val >= conf_threshold:
                    st.progress(conf_val, text=f"{class_name}: {conf_val:.1%}")
                    
        if show_gradcam:
            st.divider()
            st.subheader("🔍 Grad-CAM Visualization")
            st.write("**Red regions** = high importance | **Blue regions** = low importance")
            
            try:
                with st.spinner(f"Generating Grad-CAM..."):
                    image_resized = resize_image_for_yolo(image_np, target_size=grad_cam_size)
                    
                    # Layout structural conversion (H,W,C) to (1,C,H,W) PyTorch tensor formatting
                    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    
                    device = next(model.model.parameters()).device
                    image_tensor = image_tensor.to(device)
                    
                    # EXTRACT THE CORE PYTORCH MODEL
                    pytorch_model = model.model
                    
                    # TARGET LAYER ASSIGNMENT: Dynamic check fallback for YOLO architectures
                    if hasattr(pytorch_model, 'model') and len(pytorch_model.model) > 1:
                        target_layer = pytorch_model.model[-2] # Final neck integration features layer [1]
                    else:
                        # Direct fallback to access sub-modules if sequential container isn't standard
                        target_layer = list(pytorch_model.modules())[-4]
                        
                    # Initialize GradCAM using the extracted target layer object
                    grad_cam = GradCAM(model=pytorch_model, target_layer=target_layer)
                    
                    # Execute visualization tracking (Calls the __call__ function block)
                    heatmap, _ = grad_cam(image_tensor, class_idx=top1_idx)
                    
                    # Overlay and paint mapping on image 
                    visualization = overlay_heatmap(image_resized, heatmap, alpha=heatmap_alpha)
                    
                    st.image(
                        Image.fromarray(visualization),
                        caption=f"Grad-CAM Heatmap for '{top1_name}'",
                        use_container_width=True
                    )
                    st.success("✅ Grad-CAM visualization generated successfully!")
            except Exception as e:
                st.error(f"❌ Grad-CAM error: {str(e)}")
    else:
        st.error("❌ The selected model does not output classification probabilities. Ensure it is a classification-trained task model.")
else:
    st.info("👆 Upload an image to get started")
