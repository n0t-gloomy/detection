"""
Optional utility functions for model management
Not required - streamlit_app.py works standalone
"""

from ultralytics import YOLO
import torch


def load_model(model_path: str):
    """Load YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_device():
    """Get available device (GPU or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def predict(model, image):
    """Run inference"""
    results = model(image)
    return results[0]