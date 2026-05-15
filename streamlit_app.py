import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# ============================================================================
# GRAD-CAM
# ============================================================================

class GradCAM:
    def __init__(self, model):
        self.yolo_model = model
        self.model = model.model  # actual PyTorch model

        self.device = next(self.model.parameters()).device
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise RuntimeError("No Conv2d layer found")

        last_conv.register_forward_hook(forward_hook)
        last_conv.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()

        if input_tensor.dim() != 4:
            raise ValueError("Input must be 4D")

        b, c, h, w = input_tensor.shape
        if h % 32 != 0 or w % 32 != 0:
            raise ValueError("Image must be divisible by 32")

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        with torch.enable_grad():
            output = self.yolo_model(input_tensor)

            # ✅ unwrap YOLO output
            if isinstance(output, list):
                output = output[0]

            # ✅ FIXED probs handling
            if hasattr(output, 'probs') and output.probs is not None:
                probs = output.probs.data

                if probs.dim() > 1:
                    probs = probs.squeeze()

                if target_class is None:
                    target_class = probs.argmax().item()

                score = probs[target_class]

            elif isinstance(output, torch.Tensor):
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
            raise RuntimeError("Hooks failed")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam[0]).detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


# ============================================================================
# HELPERS
# ============================================================================

def overlay_heatmap(image, heatmap, alpha=0.4):
    h, w = image.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(image.astype(np.float32), 1 - alpha,
                           heatmap.astype(np.float32), alpha, 0).astype(np.uint8)


def resize_image_for_yolo(img, size=640):
    return np.array(
        Image.fromarray(img).resize((size, size), Image.Resampling.LANCZOS)
    )


@st.cache_resource
def load_model(path):
    return YOLO(path)


# ============================================================================
# UI
# ============================================================================

st.set_page_config(layout="wide")
st.title("🧠 YOLO + Grad-CAM")

model_path = st.sidebar.text_input("Model Path", "weights/oral.pt")
show_cam = st.sidebar.checkbox("Show Grad-CAM", True)
alpha = st.sidebar.slider("Opacity", 0.0, 1.0, 0.5)
size = st.sidebar.selectbox("Grad-CAM Size", [320, 416, 512, 640], index=3)

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ============================================================================
# MAIN
# ============================================================================

if file:
    image = Image.open(file)
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    if not os.path.exists(model_path):
        st.error("Model not found")
        st.stop()

    model = load_model(model_path)

    results = model(image)
    result = results[0]

    if hasattr(result, 'probs'):
        probs = result.probs.data

        if probs.dim() > 1:
            probs = probs.squeeze()

        idx = probs.argmax().item()
        conf = probs[idx].item()
        name = result.names[idx]

        with col2:
            st.metric("Class", name)
            st.metric("Confidence", f"{conf:.2%}")

    if show_cam:
        try:
            resized = resize_image_for_yolo(img_np, size)

            tensor = torch.from_numpy(resized)\
                .permute(2, 0, 1)\
                .unsqueeze(0)\
                .float() / 255.0

            device = next(model.model.parameters()).device
            tensor = tensor.to(device)

            cam = GradCAM(model)
            heatmap = cam.generate(tensor, target_class=idx)

            vis = overlay_heatmap(resized, heatmap, alpha)

            st.image(vis, caption="Grad-CAM")

        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

else:
    st.info("Upload an image to begin")