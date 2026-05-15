import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os


# ============================================================================
# GRAD-CAM IMPLEMENTATION (Embedded)
# ============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Hook into last conv layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        
        if last_conv:
            last_conv.register_forward_hook(forward_hook)
            last_conv.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            if hasattr(output, 'probs'):
                score = output.probs.data.max()
            else:
                score = output.max()
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients")
        
        # Compute weighted activations
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam[0]).cpu().detach().numpy()
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam


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

# ============================================================================
# MODEL LOADING
# ============================================================================

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
# MAIN APP
# ============================================================================

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
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
    
    # Check model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
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
                with st.spinner("Generating Grad-CAM..."):
                    # Prepare input
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    device = next(model.parameters()).device
                    image_tensor = image_tensor.to(device)
                    
                    # Generate Grad-CAM
                    grad_cam = GradCAM(model)
                    heatmap = grad_cam.generate(image_tensor)
                    
                    # Overlay
                    visualization = overlay_heatmap(image_np, heatmap, alpha=heatmap_alpha)
                    
                    # Display
                    st.image(
                        Image.fromarray(visualization),
                        caption="Grad-CAM Heatmap",
                        use_column_width=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Grad-CAM error: {str(e)[:100]}")
                st.info("Try disabling Grad-CAM or checking your model architecture")

else:
    st.info("👆 Upload an image to get started")
    
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        ### Features
        - **YOLO Classification**: Fast, accurate predictions
        - **Grad-CAM Visualization**: Understand model decisions
        - **Top-K Predictions**: See alternative classifications
        - **Confidence Display**: Know how certain the model is
        
        ### How it works
        1. Upload an image (JPG, JPEG, or PNG)
        2. Model classifies the image
        3. Grad-CAM shows which regions influenced the decision
        4. Red = high importance, Blue = low importance
        
        ### Tips
        - Adjust heatmap opacity for better visibility
        - Check confidence scores to verify predictions
        - Use threshold to filter low-confidence predictions
        """)