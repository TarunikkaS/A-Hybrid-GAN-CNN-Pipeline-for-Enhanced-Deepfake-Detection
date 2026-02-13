#!/usr/bin/env python3
"""
gradcam.py

Lightweight Grad-CAM implementation for PyTorch models.

Usage examples (see README_gradcam.md for details):
  python gradcam.py --checkpoint best_xception.pth --timm-model xception --image split_dataset/test/real/0001.jpg --output-dir gradcam_out

This script will try to load a model using timm if --timm-model is provided.
If no timm model is provided, it will attempt to load the checkpoint's state_dict into a model passed via code.

The script finds the requested target layer by name (e.g. "features.layer4" or simply "block14") and registers hooks to compute Grad-CAM.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Import our local Xception model
from xception_model import create_xception_model, load_trained_model


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model.eval()
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _find_target_layer(self) -> torch.nn.Module:
        # Find module by dotted name or partial match. If not found, fallback to the
        # last Conv2d module (good heuristic for convolutional backbones such as Xception).
        modules = dict(self.model.named_modules())
        if self.target_layer_name in modules:
            return modules[self.target_layer_name]

        # fallback: partial match of last token
        target_token = self.target_layer_name.split(".")[-1]
        for name, m in modules.items():
            if name.endswith(target_token):
                return m

        # fallback: pick the last Conv2d module
        import torch.nn as nn

        for name, m in reversed(list(modules.items())):
            if isinstance(m, nn.Conv2d):
                print(f"Info: target layer '{self.target_layer_name}' not found; using last Conv2d: '{name}'")
                return m

        raise ValueError(f"Target layer '{self.target_layer_name}' not found and no Conv2d fallback available. Available layers: {list(modules.keys())[:20]}")

    def _register_hooks(self):
        target_layer = self._find_target_layer()

        def forward_hook(module, inp, out):
            # out shape: (N, C, H, W)
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] shape: (N, C, H, W)
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Compute Grad-CAM for input tensor. Returns heatmap (H, W) normalized 0-1."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Handle single-output models (shape [N,1]) as in your notebook (BCEWithLogits)
        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            score = logits.view(-1) if logits.dim() == 1 else logits[:, 0]
            target_val = score[0] if target_class is None else score[target_class]
            target_val.backward(retain_graph=False)
        else:
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())
            one_hot = torch.zeros_like(logits)
            one_hot[0, target_class] = 1.0
            logits.backward(gradient=one_hot, retain_graph=False)

        # gradients: (1, C, H, W), activations: (1, C, H, W)
        grads = self.gradients[0]  # (C, H, W)
        acts = self.activations[0]  # (C, H, W)

        # Global average pooling over spatial dims
        weights = grads.view(grads.size(0), -1).mean(dim=1)  # (C,)

        cam = (weights[:, None, None] * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def load_image(image_path: str, img_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    pil = Image.open(image_path).convert("RGB")
    orig = np.array(pil)[:, :, ::-1].copy()  # BGR for cv2

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Use same normalization used during training in the notebook
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    tensor = preprocess(pil).unsqueeze(0)
    return tensor, orig


def overlay_heatmap_on_image(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Ensure both images have the same dimensions
    if heatmap_color.shape[:2] != img_bgr.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (img_bgr.shape[1], img_bgr.shape[0]))
    
    overlay = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)
    return overlay


def find_checkpoint_state_dict(checkpoint_path: str):
    data = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(data, dict):
        # common keys used by different training scripts
        if "state_dict" in data:
            return data["state_dict"]
        if "model_state_dict" in data:
            return data["model_state_dict"]
        if "model" in data:
            return data["model"]
        # maybe the checkpoint is itself a state_dict
        return data
    else:
        return data


def build_model_from_timm(timm_name: str, num_classes: int = 1):
    """Legacy function for backward compatibility - redirects to xception_model."""
    if timm_name.lower() == "xception":
        return create_xception_model(num_classes=num_classes, pretrained=False)
    else:
        try:
            import timm
            model = timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to create timm model '{timm_name}': {e}")


def try_load_weights(model: torch.nn.Module, checkpoint_path: str):
    sd = find_checkpoint_state_dict(checkpoint_path)
    # Attempt to adapt prefixes if necessary
    if all(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    return model


def process_image_file(model: torch.nn.Module, gradcam: GradCAM, image_path: str, out_path: str, img_size: int, device: str):
    tensor, orig_bgr = load_image(image_path, img_size)
    tensor = tensor.to(device)
    cam = gradcam.generate_cam(tensor)
    overlay = overlay_heatmap_on_image(orig_bgr, cam)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, overlay)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for PyTorch models")
    parser.add_argument("--checkpoint", type=str, default="./best_xception.pth", help="Path to checkpoint (pth) file (default: ./best_xception.pth)")
    parser.add_argument("--timm-model", type=str, default=None, help="If provided, build model from timm with this name")
    parser.add_argument("--image", type=str, help="Image file to process")
    parser.add_argument("--image-dir", type=str, help="Directory of images to process (will process all jpg/png)" )
    parser.add_argument("--output-dir", type=str, default="gradcam_out", help="Output directory")
    parser.add_argument("--target-layer", type=str, default="features", help="Target layer name to attach hooks to (module name)")
    parser.add_argument("--img-size", type=int, default=299, help="Image size for model input (default 299)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    if not args.image and not args.image_dir:
        raise SystemExit("Provide --image or --image-dir")

    # Build and load model
    try:
        if args.timm_model and args.timm_model.lower() != "xception":
            # Use specified timm model
            model = build_model_from_timm(args.timm_model, num_classes=1)
            model = try_load_weights(model, args.checkpoint)
            model.to(device)
        else:
            # Use our Xception model directly - this is the preferred path
            model, metadata = load_trained_model(args.checkpoint, device=device)
            print(f"✅ Loaded Xception model from {args.checkpoint}")
            print(f"   Classes: {metadata['class_to_idx']}")
            print(f"   Image size: {metadata['img_size']}")
            print(f"   Threshold: {metadata['threshold']}")
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")

    gradcam = GradCAM(model, args.target_layer)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    if args.image:
        paths.append((args.image, out_dir / (Path(args.image).stem + "_gradcam.jpg")))
    if args.image_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in Path(args.image_dir).glob(ext):
                paths.append((str(p), out_dir / (p.stem + "_gradcam.jpg")))

    for img_path, out_path in paths:
        print(f"Processing {img_path} -> {out_path}")
        process_image_file(model, gradcam, img_path, str(out_path), args.img_size, device)


if __name__ == "__main__":
    main()
