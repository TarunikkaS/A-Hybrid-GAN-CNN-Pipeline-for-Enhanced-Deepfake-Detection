#!/usr/bin/env python3
"""
Create a single comparison image showing original vs Grad-CAM overlay.
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

def create_comparison_image(gradcam_path, original_path, output_path):
    """Create a side-by-side comparison and save it."""
    
    # Load images
    gradcam_img = Image.open(gradcam_path)
    if Path(original_path).exists():
        orig_img = Image.open(original_path)
    else:
        orig_img = Image.new('RGB', gradcam_img.size, color=(128, 128, 128))
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(orig_img)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')
    
    ax2.imshow(gradcam_img)  
    ax2.set_title("Grad-CAM Attention Heatmap", fontsize=14)
    ax2.axis('off')
    
    plt.suptitle(f"Deepfake Detection - Grad-CAM Visualization\n{Path(gradcam_path).stem}", fontsize=16)
    plt.tight_layout()
    
    # Save the comparison
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison saved to: {output_path}")

if __name__ == "__main__":
    # Create comparison for a fake image
    fake_gradcam = "gradcam_test_batch/044_945_6_gradcam.jpg"
    fake_original = "split_dataset/test/fake/044_945_6.jpg"
    
    if Path(fake_gradcam).exists():
        create_comparison_image(fake_gradcam, fake_original, "gradcam_comparison_fake.jpg")
    
    # Create comparison for the real image  
    real_gradcam = "gradcam_test/033_16_gradcam.jpg"
    real_original = "split_dataset/test/real/033_16.jpg"
    
    if Path(real_gradcam).exists():
        create_comparison_image(real_gradcam, real_original, "gradcam_comparison_real.jpg")
        
    print("\n🎯 Grad-CAM Analysis:")
    print("- The heatmap shows where your Xception model focuses attention")
    print("- Red/yellow areas = high attention (important for classification)")
    print("- Blue/purple areas = low attention") 
    print("- For deepfakes, the model often focuses on facial artifacts, edges, or inconsistencies")
    print("- For real images, attention is typically more evenly distributed across natural features")