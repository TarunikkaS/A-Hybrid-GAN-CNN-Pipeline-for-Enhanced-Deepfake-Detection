#!/usr/bin/env python3
"""
Display Grad-CAM results in a grid to visualize model attention.
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

def show_gradcam_grid(gradcam_dir, original_dir, num_samples=6, figsize=(15, 10)):
    """Show a grid of original images vs Grad-CAM overlays."""
    
    gradcam_dir = Path(gradcam_dir)
    original_dir = Path(original_dir)
    
    # Get available Grad-CAM images
    gradcam_files = list(gradcam_dir.glob("*_gradcam.jpg"))[:num_samples]
    
    if not gradcam_files:
        print(f"No Grad-CAM images found in {gradcam_dir}")
        return
    
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, gradcam_path in enumerate(gradcam_files):
        # Extract original filename (remove _gradcam suffix)
        orig_name = gradcam_path.stem.replace("_gradcam", "") + ".jpg"
        orig_path = original_dir / orig_name
        
        # Load images
        try:
            gradcam_img = Image.open(gradcam_path)
            if orig_path.exists():
                orig_img = Image.open(orig_path)
            else:
                # Create placeholder if original not found
                orig_img = Image.new('RGB', gradcam_img.size, color='gray')
                print(f"Warning: Original image not found: {orig_path}")
        except Exception as e:
            print(f"Error loading {gradcam_path}: {e}")
            continue
            
        # Display original
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original\n{orig_name}", fontsize=10)
        axes[0, i].axis('off')
        
        # Display Grad-CAM overlay
        axes[1, i].imshow(gradcam_img)
        axes[1, i].set_title(f"Grad-CAM\n{gradcam_path.name}", fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle("Grad-CAM Attention Visualization", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Show results from the test batch (fake images)
    print("🔍 Showing Grad-CAM results for fake images...")
    show_gradcam_grid(
        gradcam_dir="gradcam_test_batch", 
        original_dir="split_dataset/test/fake",
        num_samples=6
    )
    
    # Show the single real image result
    print("\n🔍 Showing Grad-CAM result for real image...")
    show_gradcam_grid(
        gradcam_dir="gradcam_test", 
        original_dir="split_dataset/test/real",
        num_samples=1
    )