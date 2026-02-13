"""
Grad-CAM Utilities for Visual Explanation

Minimal implementation of Gradient-weighted Class Activation Mapping (Grad-CAM)
for visualizing which regions of an image influence the model's prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    
    Computes gradient-weighted class activation maps to visualize
    which spatial regions contribute most to a prediction.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model in eval mode
            target_layer: Specific layer to compute CAM from (nn.Module)
                         If None, automatically finds last Conv2d layer
        """
        self.model = model
        self.target_layer = target_layer or self._find_last_conv_layer()
        
        # Storage for forward/backward pass
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_last_conv_layer(self):
        """
        Automatically find the last Conv2d layer in the model.
        
        Returns:
            nn.Module: The last convolutional layer
        """
        last_conv = None
        
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model")
        
        print(f"🎯 Auto-selected target layer: {last_conv}")
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate class activation map for given input and class.
        
        For single-output models (binary classification with sigmoid),
        class_idx should be 0 or can be omitted.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (for single-output models, use 0)
        
        Returns:
            cam: Activation map normalized to [0, 1], shape (H, W)
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle both single-output and multi-output models
        if output.dim() == 0 or (output.dim() == 1 and output.shape[0] == 1):
            # Single output (binary classification with sigmoid)
            # Just backprop the output itself
            target_output = output
        else:
            # Multi-output (multi-class with softmax)
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            # Create one-hot for target class
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1.0
            target_output = (output * one_hot).sum()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target_output.backward(retain_graph=True)
        
        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def __call__(self, input_tensor, class_idx=None):
        """Convenience method for generate_cam."""
        return self.generate_cam(input_tensor, class_idx)


def overlay_cam_on_image(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay a class activation map on an image.
    
    Args:
        image: PIL Image or numpy array (H, W, 3) in RGB, values [0, 255]
        cam: Activation map (H, W) normalized to [0, 1]
        alpha: Blending factor (0 = original image, 1 = full heatmap)
        colormap: OpenCV colormap (default: COLORMAP_JET)
    
    Returns:
        overlay: PIL Image with CAM overlay
    """
    # Convert image to numpy if PIL
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # Ensure image is RGB uint8
    if image_np.dtype != np.uint8:
        image_np = (image_np * 255).astype(np.uint8)
    
    # Resize CAM to match image dimensions
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert CAM to heatmap
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend heatmap with original image
    overlay = (image_np * (1 - alpha) + heatmap * alpha).astype(np.uint8)
    
    return Image.fromarray(overlay)


def find_last_conv_layer(model):
    """
    Find the last convolutional layer in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        nn.Module: Last Conv2d layer
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model")
    
    return last_conv


if __name__ == "__main__":
    print("Grad-CAM utilities loaded successfully")
    print("Available functions:")
    print("  - GradCAM: Main class for generating activation maps")
    print("  - overlay_cam_on_image: Blend CAM with image")
    print("  - find_last_conv_layer: Auto-detect target layer")
