from ultralytics import YOLO
import numpy as np

# Load model once (important for performance)
model = YOLO("weights/exp-3.pt")

def classifier(img, weights_name, class_names):
    # YOLO accepts PIL directly, no resize needed
    results = model(img)

    # Get first result
    r = results[0]

    # If classification model
    if hasattr(r, "probs") and r.probs is not None:
        probs = r.probs.data.cpu().numpy()
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        return class_names[class_id], confidence

    # If detection/segmentation model
    elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        best_idx = int(np.argmax(confs))
        class_id = class_ids[best_idx]
        confidence = float(confs[best_idx])

        return class_names[class_id], confidence

    return "No detection", 0.0