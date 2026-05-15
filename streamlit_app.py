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
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index (if None, uses predicted class)
        """
        self.model.eval()
        
        # Ensure input has correct shape (B, C, H, W) and divisible by 32
        if input_tensor.dim() != 4:
            raise ValueError(f"Input must be 4D tensor, got {input_tensor.dim()}D")
        
        b, c, h, w = input_tensor.shape
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(f"Image dimensions ({h}, {w}) must be divisible by 32")
        
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            # Handle different output formats
            if hasattr(output, 'probs'):
                # Classification output with probs attribute
                probs = output.probs.data  # Shape: (batch_size, num_classes)
                if target_class is None:
                    target_class = probs.argmax(dim=1)[0].item()
                score = probs[0, target_class]  # Get score for target class
            elif isinstance(output, torch.Tensor):
                # Raw tensor output
                if output.dim() > 1:
                    if target_class is None:
                        target_class = output.argmax(dim=1)[0].item()
                    score = output[0, target_class]
                else:
                    score = output.max()
            else:
                raise ValueError(f"Unexpected output type: {type(output)}")
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        # Ensure gradients and activations are 4D (B, C, H, W)
        if self.gradients.dim() == 4 and self.activations.dim() == 4:
            # Compute weighted activations
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1)
            cam = F.relu(cam[0]).cpu().detach().numpy()
        else:
            raise RuntimeError(f"Unexpected tensor shapes: grad {self.gradients.shape}, activ {self.activations.shape}")
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
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
        2. Upload an image (JPG, JPEG, or PNG)
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