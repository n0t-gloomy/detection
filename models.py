import torch
import numpy as np
import cv2
from torchvision import transforms

# --- Load model ---
model = torch.load("weights/oral.pt", map_location="cpu")
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Hooks ---
features = []
gradients = []

def forward_hook(module, input, output):
    features.append(output)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

# ⚠️ You may need to adjust this layer
target_layer = list(model.modules())[-2]

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


def classifier(img, weights_name):
    features.clear()
    gradients.clear()

    input_tensor = transform(img).unsqueeze(0)

    # --- Forward ---
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].detach().numpy()

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    # --- Backward ---
    model.zero_grad()
    output[0, class_id].backward()

    # --- Grad-CAM ---
    grads = gradients[0][0].detach().numpy()
    fmap = features[0][0].detach().numpy()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.width, img.height))
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    return class_id, confidence, overlay