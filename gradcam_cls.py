"""
Grad-CAM for Ultralytics YOLO classification models (.pt).

Designed to match default YOLO-cls inference: RGB, square resize to imgsz, /255.
Top-1 only. Layer can be auto (last Conv2d with spatial maps) or user-selected for debug.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO


def _imgsz_from_yolo(model: YOLO) -> int:
    """Resolve training image size; default 640 matches common Ultralytics Cloud cls jobs."""
    args = getattr(getattr(model, "model", None), "args", None)
    if args is not None:
        z = getattr(args, "imgsz", None)
        if z is not None:
            if isinstance(z, (list, tuple)):
                return int(z[0])
            return int(z)
    ov = getattr(model, "overrides", None) or {}
    z = ov.get("imgsz")
    if z is not None:
        if isinstance(z, (list, tuple)):
            return int(z[0])
        return int(z)
    return 640


def _pil_to_model_input(pil_image: Image.Image, imgsz: int, device: torch.device) -> torch.Tensor:
    """Match typical YOLO-cls predict: RGB, square resize, float in [0, 1]."""
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    resized = cv2.resize(arr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return t.to(device)


def _collect_conv2d_modules(net: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    out: list[tuple[str, nn.Conv2d]] = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            out.append((name, m))
    return out


def _pick_target_layer(
    net: nn.Module,
    imgsz: int,
    device: torch.device,
    layer_name: str | None,
) -> tuple[nn.Module, str]:
    convs = _collect_conv2d_modules(net)
    if not convs:
        raise RuntimeError("No Conv2d layers found in model.model")

    if layer_name:
        for name, m in convs:
            if name == layer_name:
                return m, name
        raise ValueError(f"Layer '{layer_name}' not found or not Conv2d")

    # Auto: probe each Conv2d output shape; prefer the last conv with enough spatial
    # resolution for a meaningful map (avoids tiny tail 1×1 / head convs that differ by build).
    probe = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=next(net.parameters()).dtype)
    handles: list = []
    # (module, name, H, W) in forward order
    spatial_trace: list[tuple[nn.Module, str, int, int]] = []

    def make_hook(n: str):
        def hook(_m, _inp, out):
            if torch.is_tensor(out) and out.dim() == 4:
                _b, _c, h, w = out.shape
                if h > 1 and w > 1:
                    spatial_trace.append((_m, n, int(h), int(w)))

        return hook

    try:
        for n, m in convs:
            handles.append(m.register_forward_hook(make_hook(n)))
        with torch.no_grad():
            net(probe)
    finally:
        for h in handles:
            h.remove()

    if spatial_trace:
        # e.g. 640 → min side ≥ 10: skip only the very deepest 1–2 px maps if any slipped through
        min_side = max(4, imgsz // 64)
        good = [t for t in spatial_trace if min(t[2], t[3]) >= min_side]
        pick = good[-1] if good else spatial_trace[-1]
        return pick[0], pick[1]

    return convs[-1][1], convs[-1][0]


class _GradCAM:
    """
    Uses forward hook + tensor grad hook. Avoids register_full_backward_hook, which breaks
    Ultralytics/YOLO backbones that use SiLU(inplace=True) (PyTorch view+inplace autograd error).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._tensor_hook_handles: list = []

        def forward_hook(_mod, _inputs, out):
            if not torch.is_tensor(out):
                return
            self.activations = out.detach()

            def save_grad(g):
                if g is not None:
                    self.gradients = g.detach()

            self._tensor_hook_handles.append(out.register_hook(save_grad))

        self._forward_hook_handle = target_layer.register_forward_hook(forward_hook)

    def close(self) -> None:
        for h in self._tensor_hook_handles:
            h.remove()
        self._tensor_hook_handles.clear()
        self._forward_hook_handle.remove()

    def compute(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        out = self.model(x)
        if isinstance(out, dict):
            d = out
            out = d.get("logits") or d.get("cls") or d.get("pred")
            if out is None:
                for v in d.values():
                    if torch.is_tensor(v) and v.dim() in (1, 2):
                        out = v
                        break
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not torch.is_tensor(out):
            raise RuntimeError(f"Unexpected model output type for Grad-CAM: {type(out)}")
        if out.dim() == 1:
            out = out.unsqueeze(0)
        score = out[0, class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        acts = self.activations[0]
        grads = self.gradients[0]
        weights = grads.mean(dim=(1, 2), keepdim=False)
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin > 1e-8:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        return cam


def list_debug_conv_layer_names(model: YOLO) -> list[str]:
    net = model.model
    return [n for n, _ in _collect_conv2d_modules(net)]


def gradcam_overlay(
    yolo: YOLO,
    pil_image: Image.Image,
    top1_class_idx: int,
    layer_name: str | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Returns HxWx3 uint8 RGB image: original-sized overlay for top-1 class.
    """
    device = next(yolo.model.parameters()).device
    dtype = next(yolo.model.parameters()).dtype
    imgsz = _imgsz_from_yolo(yolo)

    net = yolo.model
    net.eval()

    promoted_fp32 = False
    if device.type == "cpu" and dtype == torch.float16:
        net.float()
        promoted_fp32 = True

    try:
        x = _pil_to_model_input(pil_image, imgsz, device)
        if dtype == torch.float16 and not promoted_fp32:
            x = x.half()

        target_mod, _picked = _pick_target_layer(net, imgsz, device, layer_name)
        cam_engine = _GradCAM(net, target_mod)
        try:
            with torch.enable_grad():
                cam = cam_engine.compute(x.requires_grad_(True), int(top1_class_idx))
        finally:
            cam_engine.close()
    finally:
        if promoted_fp32:
            net.half()

    orig = np.array(pil_image.convert("RGB"))
    h, w = orig.shape[:2]
    heat = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heat_color.astype(np.float32) + (1.0 - alpha) * orig.astype(np.float32)).clip(0, 255)
    return overlay.astype(np.uint8)
