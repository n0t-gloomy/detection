import streamlit as st
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================

def get_target_layer(yolo_model):
    """
    Auto-discover the last Conv2d in the YOLO backbone.
    Works for YOLO classification models (YOLOv8-cls, etc.).
    Returns the layer module and its name, or (None, None) if not found.
    """
    nn_model = yolo_model.model  # underlying nn.Sequential
    target_layer = None
    target_name = None

    for name, module in nn_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            target_name = name

    return target_layer, target_name


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for YOLO classification models.

    Usage:
        gcam = GradCAM(yolo_model)
        heatmap = gcam.generate(image_tensor, target_class=idx)
    """

    def __init__(self, yolo_model, target_layer=None):
        self.yolo_model = yolo_model
        self.nn_model = yolo_model.model  # raw nn.Sequential

        # Auto-detect layer if not provided
        if target_layer is None:
            target_layer, layer_name = get_target_layer(yolo_model)
            if target_layer is None:
                raise ValueError(
                    "Could not find a Conv2d layer in the model. "
                    "Pass target_layer explicitly."
                )
            self._layer_name = layer_name
        else:
            self._layer_name = "custom"

        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._hooks = []

    def _register_hooks(self):
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _save_activation(self, module, input, output):
        # Keep a version WITH grad_fn so backward can flow through it
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class=None):
        """
        Generate a Grad-CAM heatmap.

        Args:
            image_tensor: (1, C, H, W) float32 tensor in [0, 1]
            target_class: class index to explain (uses argmax if None)

        Returns:
            heatmap: (H_cam, W_cam) numpy array in [0, 1]
            target_class: int, the class that was explained
        """
        # Temporarily re-enable gradients for all model parameters
        # (YOLO wraps inference in torch.no_grad() by default)
        for p in self.nn_model.parameters():
            p.requires_grad_(True)

        self.nn_model.eval()
        self._register_hooks()

        try:
            # torch.enable_grad() overrides any outer no_grad context
            with torch.enable_grad():
                # Give the input a grad_fn so the graph is built
                image_tensor = image_tensor.detach().requires_grad_(True)

                # ---- forward pass through the raw nn model ----
                output = self.nn_model(image_tensor)

                # YOLO may return a list (e.g. [logits, extra]) — take the first
                if isinstance(output, (list, tuple)):
                    output = output[0]

                # Squeeze batch & spatial dims if needed → shape (num_classes,)
                if output.dim() == 4:          # (B, C, 1, 1)
                    output = output.squeeze(-1).squeeze(-1)
                if output.dim() == 2:          # (B, num_classes)
                    output = output[0]

                if target_class is None:
                    target_class = int(output.argmax().item())

                # ---- backward pass ----
                self.nn_model.zero_grad()
                one_hot = torch.zeros_like(output)
                one_hot[target_class] = 1.0
                output.backward(gradient=one_hot, retain_graph=False)

            # ---- compute CAM ----
            # gradients / activations: (1, C, H, W)
            activations = self._activations.detach()
            gradients  = self._gradients          # already detached in hook

            weights = gradients.mean(dim=(2, 3), keepdim=True)   # global avg pool
            cam = (weights * activations).sum(dim=1, keepdim=True)  # weighted sum
            cam = F.relu(cam)                                         # keep positives

            # Normalise to [0, 1]
            cam = cam.squeeze().cpu().numpy()
            cam = cam - cam.min()
            denom = cam.max()
            if denom > 1e-8:
                cam = cam / denom

            return cam, target_class

        finally:
            self._remove_hooks()
            # Restore inference-safe state: disable grads on params
            for p in self.nn_model.parameters():
                p.requires_grad_(False)


def overlay_heatmap(image_np, heatmap, alpha=0.45):
    """
    Blend a Grad-CAM heatmap over an RGB image.

    Args:
        image_np: (H, W, 3) uint8 numpy array
        heatmap:  float numpy array in [0, 1] (any spatial size)
        alpha:    heatmap opacity

    Returns:
        blended (H, W, 3) uint8 array
    """
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(
        image_np.astype(np.float32), 1 - alpha,
        heatmap_color.astype(np.float32), alpha,
        0,
    )
    return blended.astype(np.uint8)


def resize_for_yolo(image_np, target_size=640):
    """
    Resize to (target_size × target_size), clamped to a multiple of 32.
    """
    target_size = max(32, (target_size // 32) * 32)
    pil = Image.fromarray(image_np).resize(
        (target_size, target_size), Image.Resampling.LANCZOS
    )
    return np.array(pil)


# ============================================================================
# CACHED HELPERS
# ============================================================================

@st.cache_resource
def find_model_files(directory="weights"):
    models = []
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.endswith(".pt"):
                models.append(f)
    return sorted(models)


@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="YOLO + Grad-CAM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 YOLO Classification with Grad-CAM")
st.write("Upload an image to classify it and visualise which regions drove the decision.")

# ---- Sidebar ----
st.sidebar.header("⚙️ Settings")

available_models = find_model_files("weights")
if available_models:
    selected_model = st.sidebar.selectbox(
        "Select Model", available_models,
        help="Choose a YOLO .pt model from the weights/ folder"
    )
    model_path = f"weights/{selected_model}"
else:
    st.sidebar.warning("⚠️ No .pt models found in weights/")
    model_path = st.sidebar.text_input("Model Path", value="weights/oral.pt")

show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)
heatmap_alpha = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.45, 0.05)
conf_threshold = st.sidebar.slider("Min Confidence to Show", 0.0, 1.0, 0.0, 0.05)
grad_cam_size = st.sidebar.selectbox(
    "Grad-CAM Resolution",
    [320, 416, 512, 640, 704, 768],
    index=3,
    help="Larger = better quality, slower. Must be divisible by 32.",
)

st.sidebar.divider()
st.sidebar.caption("Model is cached after first load.")

# ---- Main ----
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        st.caption(f"Size: {image.size[0]} × {image.size[1]} px")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at `{model_path}`")
        st.stop()

    model = load_yolo_model(model_path)
    if model is None:
        st.stop()

    # ---- Inference ----
    with st.spinner("🔄 Running inference…"):
        results = model(image)
        result = results[0]

    if not (hasattr(result, "probs") and result.probs is not None):
        st.error("❌ This model doesn't return classification probabilities. "
                 "Make sure it's a YOLO classification model.")
        st.stop()

    probs = result.probs.data          # tensor of shape (num_classes,)
    top1_idx = int(probs.argmax().item())
    top1_conf = float(probs[top1_idx].item())
    top1_name = result.names[top1_idx]

    with col2:
        st.subheader("🎯 Prediction")
        st.metric("Class", top1_name)
        st.metric("Confidence", f"{top1_conf:.1%}")

        st.subheader("Top-5 Predictions")
        top_k = min(5, len(result.names))
        top_confs, top_indices = torch.topk(probs.squeeze(), k=top_k)
        for conf, idx in zip(top_confs, top_indices):
            name = result.names[int(idx.item())]
            val = float(conf.item())
            if val >= conf_threshold:
                st.progress(val, text=f"{name}: {val:.1%}")

    # ---- Grad-CAM ----
    if show_gradcam:
        st.divider()
        st.subheader("🔍 Grad-CAM Heatmap")
        st.write("🔴 **Red** = high influence on decision &nbsp; | &nbsp; 🔵 **Blue** = low influence")

        try:
            with st.spinner(f"Generating Grad-CAM at {grad_cam_size}×{grad_cam_size}…"):
                # Prepare input
                image_resized = resize_for_yolo(image_np, target_size=grad_cam_size)
                tensor = (
                    torch.from_numpy(image_resized)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                device = next(model.model.parameters()).device
                tensor = tensor.to(device)

                # Run Grad-CAM
                gcam = GradCAM(model)           # auto-finds last Conv2d
                heatmap, explained_class = gcam.generate(tensor, target_class=top1_idx)

                # Overlay
                vis = overlay_heatmap(image_resized, heatmap, alpha=heatmap_alpha)

            gcam_col1, gcam_col2 = st.columns(2)
            with gcam_col1:
                st.image(image_resized, caption="Resized Input", use_column_width=True)
            with gcam_col2:
                st.image(
                    Image.fromarray(vis),
                    caption=f"Grad-CAM for '{result.names[explained_class]}' "
                            f"({grad_cam_size}×{grad_cam_size})",
                    use_column_width=True,
                )

            st.success("✅ Grad-CAM generated successfully!")

            # Download button
            _, buf_col, _ = st.columns([1, 2, 1])
            with buf_col:
                import io
                buf = io.BytesIO()
                Image.fromarray(vis).save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Grad-CAM Image",
                    data=buf.getvalue(),
                    file_name="gradcam.png",
                    mime="image/png",
                )

        except Exception as e:
            st.error(f"❌ Grad-CAM failed: {e}")
            with st.expander("Troubleshooting"):
                st.markdown("""
- Confirm the model is a **YOLO classification** model (task=`classify`), not detection/segmentation.
- Try a different **Grad-CAM Resolution** in the sidebar.
- If the model returns a list from its backbone, the auto-layer detection may need adjusting — open an issue on the repo.
""")

else:
    st.info("👆 Upload an image to get started.")

# ---- About expander ----
with st.expander("ℹ️ About this app"):
    st.markdown("""
### How Grad-CAM works
Grad-CAM computes the gradient of the target class score with respect to the feature maps
of the **last convolutional layer** in the network. The gradient magnitudes tell us how
important each spatial region is — high values appear red in the overlay.

### Steps
1. Select a YOLO `.pt` classification model from the `weights/` folder.
2. Upload an image (JPG / PNG / WEBP).
3. The model classifies the image and returns top-5 predictions.
4. Grad-CAM highlights the regions that drove the top prediction.

### Tuning tips
| Setting | Effect |
|---|---|
| **Heatmap Opacity** | 0.3–0.6 gives a good balance |
| **Grad-CAM Resolution** | Higher = sharper map, slower |
| **Min Confidence** | Hides very uncertain classes |

### Supported models
Any YOLO `.pt` file trained for **classification** (`yolov8n-cls`, `yolov8s-cls`, etc.).
""")

# ---- Available models list ----
st.divider()
st.subheader("📦 Available Models")
if available_models:
    st.success(f"Found {len(available_models)} model(s) in `weights/`:")
    for m in available_models:
        st.write(f"• `{m}`")
else:
    st.warning("No models found in `weights/`. Place your `.pt` files there.")