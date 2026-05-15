"""
models.py — utility helpers for the YOLO + Grad-CAM Streamlit app.

The main Grad-CAM logic lives in streamlit_app.py (self-contained),
but helpers here can be imported by other scripts if needed.
"""

from ultralytics import YOLO
import torch


def load_model(model_path: str) -> YOLO | None:
    """Load a YOLO model from a .pt file. Returns None on failure."""
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        return None


def get_device() -> torch.device:
    """Return GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model: YOLO, image):
    """Run inference and return the first result object."""
    return model(image)[0]


def get_last_conv_layer(yolo_model: YOLO):
    """
    Traverse the YOLO backbone and return the last Conv2d layer.
    Useful for manually specifying the Grad-CAM target layer.

    Example:
        layer, name = get_last_conv_layer(model)
        gcam = GradCAM(model, target_layer=layer)
    """
    last_layer = None
    last_name = None
    for name, module in yolo_model.model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_layer = module
            last_name = name
    return last_layer, last_name