#!/usr/bin/env python3
"""
Robust Deepfake Detection with Trust-Aware Prediction
======================================================
Loads a pretrained Xception model and performs inference with perturbation-based
confidence estimation to detect unreliable predictions.

Model: timm Xception (num_classes=1, binary classification)
Checkpoint: xception_gan_augmented.pth
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import io
from typing import Tuple, List

# ============================================================================
# Configuration
# ============================================================================

# Thresholds for trust-aware prediction
VARIANCE_THRESHOLD = 0.01      # Variance in p_fake across perturbations
MAX_SWING_THRESHOLD = 0.15     # Max difference between any two p_fake values
UNCERTAINTY_ZONE = (0.45, 0.55)  # Range where predictions are considered uncertain

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Google Drive Mounting (Colab)
# ============================================================================

def mount_drive():
    """Mount Google Drive in Colab environment"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("✅ Google Drive mounted successfully")
    except ImportError:
        print("⚠️  Not running in Colab, skipping Drive mount")
    except Exception as e:
        print(f"⚠️  Drive mount failed: {e}")

# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path: str) -> nn.Module:
    """
    Load pretrained Xception model from checkpoint.
    Handles both direct state_dict and nested dict formats.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        
    Returns:
        Loaded model in eval mode on DEVICE
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model architecture
    model = timm.create_model('xception', pretrained=False, num_classes=1)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Try common keys for state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            # Could be nested model object or state_dict
            if isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            else:
                # Full model object
                state_dict = checkpoint['model'].state_dict()
        else:
            # Assume entire checkpoint is state_dict
            state_dict = checkpoint
    else:
        # Direct state_dict or model object
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights (strict=False to handle minor mismatches)
    model.load_state_dict(state_dict, strict=False)
    
    # Move to device and set to eval mode
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"📍 Device: {DEVICE}")
    
    return model

# ============================================================================
# Preprocessing
# ============================================================================

def get_transform():
    """Standard ImageNet preprocessing for Xception (299x299)"""
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_image(img_path: str) -> Image.Image:
    """Load image and convert to RGB"""
    img = Image.open(img_path).convert('RGB')
    return img

# ============================================================================
# Basic Inference
# ============================================================================

def predict_image(model: nn.Module, img_path: str) -> Tuple[str, float, float]:
    """
    Basic prediction without perturbations.
    
    Args:
        model: Loaded model
        img_path: Path to image
        
    Returns:
        (label, p_fake, logit) where:
            - label: "FAKE" or "REAL"
            - p_fake: Probability of being fake [0,1]
            - logit: Raw model output
    """
    transform = get_transform()
    img = load_image(img_path)
    
    # Preprocess
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logit = model(tensor).squeeze()
        p_fake = torch.sigmoid(logit).item()
    
    label = "FAKE" if p_fake > 0.5 else "REAL"
    logit_val = logit.item()
    
    return label, p_fake, logit_val

# ============================================================================
# Image Perturbations (Inference-Time Only)
# ============================================================================

def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression at specified quality level"""
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def apply_gaussian_blur(img: Image.Image, kernel_size: int) -> Image.Image:
    """Apply Gaussian blur with specified kernel size"""
    if kernel_size == 0:
        return img
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=kernel_size//2))

def apply_downscale(img: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale and upscale back to original size"""
    if scale_factor >= 1.0:
        return img
    w, h = img.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    return img.resize((new_w, new_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)

# ============================================================================
# Trust-Aware Prediction
# ============================================================================

def trust_aware_predict(model: nn.Module, img_path: str) -> dict:
    """
    Perform trust-aware prediction using inference-time perturbations.
    
    Tests model robustness by applying various perturbations and measuring
    prediction consistency. Flags uncertain predictions.
    
    Args:
        model: Loaded model
        img_path: Path to image
        
    Returns:
        dict containing:
            - original_label: Original prediction label
            - original_p_fake: Original fake probability
            - perturbed_p_fakes: List of probabilities for each perturbation
            - variance: Variance across all predictions
            - max_swing: Maximum difference between predictions
            - final_label: "FAKE", "REAL", or "UNTRUSTWORTHY"
            - is_trustworthy: Boolean flag
    """
    transform = get_transform()
    img = load_image(img_path)
    
    # Original prediction
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        original_p_fake = torch.sigmoid(logit).item()
    
    original_label = "FAKE" if original_p_fake > 0.5 else "REAL"
    
    # Collect predictions from perturbed variants
    perturbed_p_fakes = []
    
    # JPEG compression variants
    for quality in [95, 85, 75, 65]:
        perturbed_img = apply_jpeg_compression(img, quality)
        tensor = transform(perturbed_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            p_fake = torch.sigmoid(logit).item()
        perturbed_p_fakes.append(p_fake)
    
    # Gaussian blur variants
    for kernel_size in [0, 3, 5]:
        perturbed_img = apply_gaussian_blur(img, kernel_size)
        tensor = transform(perturbed_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            p_fake = torch.sigmoid(logit).item()
        perturbed_p_fakes.append(p_fake)
    
    # Downscale variants
    for scale_factor in [1.0, 0.9, 0.8]:
        perturbed_img = apply_downscale(img, scale_factor)
        tensor = transform(perturbed_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            p_fake = torch.sigmoid(logit).item()
        perturbed_p_fakes.append(p_fake)
    
    # Compute confidence metrics
    all_predictions = [original_p_fake] + perturbed_p_fakes
    variance = np.var(all_predictions)
    max_swing = max(all_predictions) - min(all_predictions)
    
    # Determine trustworthiness
    in_uncertainty_zone = UNCERTAINTY_ZONE[0] <= original_p_fake <= UNCERTAINTY_ZONE[1]
    high_variance = variance > VARIANCE_THRESHOLD
    high_swing = max_swing > MAX_SWING_THRESHOLD
    
    is_trustworthy = not (high_variance or high_swing or in_uncertainty_zone)
    
    if is_trustworthy:
        final_label = original_label
    else:
        final_label = "UNTRUSTWORTHY"
    
    return {
        'original_label': original_label,
        'original_p_fake': original_p_fake,
        'perturbed_p_fakes': perturbed_p_fakes,
        'variance': variance,
        'max_swing': max_swing,
        'final_label': final_label,
        'is_trustworthy': is_trustworthy
    }

# ============================================================================
# Reporting
# ============================================================================

def print_trust_aware_report(result: dict, img_path: str):
    """Print clean formatted report of trust-aware prediction"""
    print("\n" + "="*70)
    print("TRUST-AWARE DEEPFAKE DETECTION REPORT")
    print("="*70)
    print(f"📁 Image: {img_path}")
    print(f"\n🎯 Original Prediction:")
    print(f"   Label: {result['original_label']}")
    print(f"   P(Fake): {result['original_p_fake']:.4f}")
    
    print(f"\n🔄 Perturbed Predictions (n={len(result['perturbed_p_fakes'])}):")
    for i, p_fake in enumerate(result['perturbed_p_fakes'], 1):
        print(f"   [{i:2d}] P(Fake): {p_fake:.4f}")
    
    print(f"\n📊 Robustness Metrics:")
    print(f"   Variance:  {result['variance']:.6f} (threshold: {VARIANCE_THRESHOLD})")
    print(f"   Max Swing: {result['max_swing']:.4f} (threshold: {MAX_SWING_THRESHOLD})")
    
    print(f"\n✅ Final Decision:")
    if result['is_trustworthy']:
        print(f"   Status: TRUSTWORTHY ✓")
        print(f"   Label:  {result['final_label']}")
    else:
        print(f"   Status: UNTRUSTWORTHY ⚠️")
        print(f"   Label:  {result['final_label']}")
        print(f"   Reason: High prediction variance or uncertainty zone")
    
    print("="*70 + "\n")

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Mount Google Drive (Colab only)
    mount_drive()
    
    # ========================================================================
    # CONFIGURATION - Update these paths for your setup
    # ========================================================================
    
    MODEL_PATH = "/Users/tarunikkasuresh/Desktop/FINAL DEEPFAKE PROJECT MODEL/xception_gan_augmented.pth"
    IMG_PATH = "/Users/tarunikkasuresh/Desktop/FINAL DEEPFAKE PROJECT MODEL/reduced_dataset/test/fake/044_945_12.jpg"
    
    # Alternative: Use local paths if not in Colab
    # MODEL_PATH = "./xception_gan_augmented.pth"
    # IMG_PATH = "./test.jpg"
    
    # ========================================================================
    # Load Model
    # ========================================================================
    
    print("\n🚀 Loading model...")
    model = load_model(MODEL_PATH)
    
    # ========================================================================
    # Basic Prediction
    # ========================================================================
    
    print("\n📸 Running basic prediction...")
    label, p_fake, logit = predict_image(model, IMG_PATH)
    print(f"   Label: {label}")
    print(f"   P(Fake): {p_fake:.4f}")
    print(f"   Logit: {logit:.4f}")
    
    # ========================================================================
    # Trust-Aware Prediction
    # ========================================================================
    
    print("\n🔍 Running trust-aware prediction with perturbations...")
    result = trust_aware_predict(model, IMG_PATH)
    print_trust_aware_report(result, IMG_PATH)
    
    # ========================================================================
    # Batch Example (Optional)
    # ========================================================================
    
    # Uncomment to test multiple images
    """
    test_images = [
        "/content/drive/MyDrive/test_images/fake1.jpg",
        "/content/drive/MyDrive/test_images/real1.jpg",
        "/content/drive/MyDrive/test_images/suspicious1.jpg"
    ]
    
    for img_path in test_images:
        result = trust_aware_predict(model, img_path)
        print_trust_aware_report(result, img_path)
    """
