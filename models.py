from ultralytics import YOLO
import numpy as np

models_cache = {}

def get_model(weights_name):
    if weights_name not in models_cache:
        models_cache[weights_name] = YOLO(f"weights/{weights_name}")
    return models_cache[weights_name]

def classifier(img, weights_name, class_names):
    model = get_model(weights_name)
    results = model(img)
    r = results[0]

    if hasattr(r, "probs") and r.probs is not None:
        class_id = int(r.probs.top1)
        confidence = float(r.probs.top1conf)

        return model.names[class_id], confidence

    elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        best_idx = int(np.argmax(confs))
        class_id = class_ids[best_idx]
        confidence = float(confs[best_idx])

        return model.names[class_id], confidence

    return "No detection", 0.0
