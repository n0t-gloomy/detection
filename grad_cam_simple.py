import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
 
 
class SimpleGradCAM:
    """Lightweight Grad-CAM implementation"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = None
        self.activations = None
        
        # Hook into the last convolutional layer
        self._hook_layers()
    
    def _hook_layers(self):
        """Register hooks on last conv layer"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find the last conv layer in the model
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        
        if last_conv:
            last_conv.register_forward_hook(forward_hook)
            last_conv.register_full_backward_hook(backward_hook)
    
    def __call__(self, input_image: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor (1, 3, H, W)
        
        Returns:
            Heatmap (H, W) normalized to 0-1
        """
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            input_image.requires_grad_(True)
            output = self.model(input_image)
            
            # Get the score for predicted class
            if hasattr(output, 'probs'):
                scores = output.probs.data
                class_score = scores.max()
            else:
                class_score = output.max()
        
        # Backward pass
        self.model.zero_grad()
        class_score.backward(retain_graph=True)
        
        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)
        
        # Global average pooling over gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted sum of activations
        weighted_acts = (weights * activations).sum(dim=1)  # (B, H, W)
        
        # ReLU and normalize
        heatmap = F.relu(weighted_acts[0]).cpu().detach().numpy()
        
        # Normalize to 0-1
        hm_min, hm_max = heatmap.min(), heatmap.max()
        if hm_max > hm_min:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
        
        return heatmap
 
 
def overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image_np: Original image (H, W, 3) RGB
        heatmap: Grad-CAM (H, W) values 0-1
        alpha: Blending factor
    
    Returns:
        Blended image (H, W, 3)
    """
    # Ensure heatmap matches image size
    if heatmap.shape != image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    # Convert heatmap to color (0-255)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend
    result = cv2.addWeighted(
        image_np.astype(np.float32), 1 - alpha,
        heatmap_color.astype(np.float32), alpha,
        0
    )
    
    return result.astype(np.uint8)
 
 
# Example usage
if __name__ == "__main__":
    from ultralytics import YOLO
    
    # Load model and image
    model = YOLO("path/to/model.pt")
    image = Image.open("path/to/image.jpg")
    
    # Prepare input
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Generate Grad-CAM
    grad_cam = SimpleGradCAM(model)
    heatmap = grad_cam(image_tensor.to(next(model.parameters()).device))
    
    # Visualize
    image_np = np.array(image)
    result = overlay_heatmap(image_np, heatmap, alpha=0.5)
    
    # Display
    result_image = Image.fromarray(result)
    result_image.show()