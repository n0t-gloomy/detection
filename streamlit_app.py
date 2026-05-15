import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# ============================================================================
# GRAD-CAM IMPLEMENTATION (Fixed for YOLO)
# ============================================================================

class GradCAM:
    """Gradient-weighted Class Activation Mapping for YOLO Classification"""
    
    def __init__(self, yolo_model):
        # Target the underlying PyTorch nn.Module
        self.model = yolo_model.model
        self.hooks = []
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to the last convolutional layer"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; the gradient is the first element
            self.gradients = grad_output[0].detach()
        
        # Find the last Conv2d layer in the architecture
        target_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        
        if target_layer:
            self.hooks.append(target_layer.register_forward_hook(forward_hook))
            self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
        else:
            raise RuntimeError("Could not find a Conv2d layer in the model.")

    def generate(self, input_tensor, target_class=None):
        """Generates the heatmap"""
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Enable gradients for the backward pass
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            # FIX: YOLO internal models often return a list [logits]
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            score = output[0, target_class]
            
            self.model.zero_grad()
            score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured. Check hooks.")

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam[0]).cpu().detach().numpy()
        
        # Normalize heatmap to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def release(self):
        """Remove hooks to prevent memory leaks and duplication"""
        for hook in self.hooks:
            hook.remove()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def overlay_heatmap(image, heatmap, alpha=0.5):
    """Resizes and overlays heatmap on the original image"""
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    result = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return result

def prepare_input(image_np, size=640):
    """Prepares numpy image for YOLO input"""
    image_pil = Image.fromarray(image_np).resize((size, size))
    img_resized = np.array(image_pil)
    # Normalize and convert to (B, C, H, W) tensor
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_resized, tensor

@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

st.set_page_config(page_title="YOLO Grad-CAM", layout="wide")
st.title("🧠 YOLO Classification & Grad-CAM")

# Sidebar
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model Path", "weights/oral.pt")
heatmap_alpha = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.5)
img_size = st.sidebar.selectbox("Inference Size", [320, 640, 1280], index=1)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if os.path.exists(model_path):
        model = load_yolo_model(model_path)
        
        if model:
            # 1. Standard Inference for Results
            results = model(image)
            result = results[0]
            
            if hasattr(result, 'probs') and result.probs is not None:
                top1_idx = result.probs.top1
                top1_conf = result.probs.top1conf.item()
                top1_name = result.names[top1_idx]

                with col2:
                    st.subheader("Prediction")
                    st.metric("Class", top1_name)
                    st.metric("Confidence", f"{top1_conf:.1%}")

                # 2. Grad-CAM Generation
                st.divider()
                st.subheader("🔍 Explainable AI (Grad-CAM)")
                
                try:
                    # Prepare data
                    img_resized, input_tensor = prepare_input(img_np, size=img_size)
                    
                    # Initialize GradCAM
                    cam_engine = GradCAM(model)
                    heatmap = cam_engine.generate(input_tensor, target_class=top1_idx)
                    
                    # Create visualization
                    viz = overlay_heatmap(img_resized, heatmap, alpha=heatmap_alpha)
                    
                    st.image(viz, caption=f"Attention area for '{top1_name}'", use_container_width=True)
                    
                    # Cleanup hooks
                    cam_engine.release()
                    
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
            else:
                st.warning("This model does not appear to be a Classification model.")
    else:
        st.error(f"Model file not found at {model_path}")