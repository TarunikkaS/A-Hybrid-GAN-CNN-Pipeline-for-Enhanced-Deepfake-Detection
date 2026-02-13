#!/usr/bin/env python3
"""
show_gradcam_grid.py

Create a grid showing 5 fake vs 5 real images with their Grad-CAM visualizations.
Shows original images on top row and Grad-CAM heatmaps on bottom row.
"""

import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import timm
import torchvision.transforms as T
import glob
import random
from pathlib import Path
import os

# Import our Grad-CAM tools
try:
    from xception_model import load_trained_model
    from gradcam import GradCAM, load_image, overlay_heatmap_on_image
    gradcam_available = True
    print("✅ Grad-CAM tools imported successfully")
except ImportError as e:
    gradcam_available = False
    print(f"⚠️ Grad-CAM tools not available: {e}")
    exit(1)


def predict_image_standalone(img_path, checkpoint_path="./best_xception.pth"):
    """Standalone prediction function"""
    try:
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Setup model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model('xception', pretrained=False, num_classes=1).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        
        # Get metadata
        thresh = ckpt.get("threshold", 0.5)
        img_size = ckpt.get("img_size", 299)
        classes = [k for k,_ in sorted(ckpt["class_to_idx"].items(), key=lambda x:x[1])]
        
        # Define transforms
        tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        
        # Make prediction
        img = Image.open(img_path).convert("RGB")
        x = tfms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x).squeeze(1)).item()
        pred = int(prob >= thresh)
        label = classes[pred] if len(classes)==2 else str(pred)
        
        return label, prob
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "unknown", 0.0


def generate_gradcam_for_image(img_path, target_layer="conv4"):
    """Generate Grad-CAM visualization for a single image"""
    try:
        # Load model for Grad-CAM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gradcam_model, metadata = load_trained_model("./best_xception.pth", device=device)
        
        # Create Grad-CAM instance
        gradcam = GradCAM(gradcam_model, target_layer)
        
        # Process image for Grad-CAM
        tensor, orig_bgr = load_image(img_path, metadata['img_size'])
        tensor = tensor.to(device)
        cam = gradcam.generate_cam(tensor)
        overlay = overlay_heatmap_on_image(orig_bgr, cam)
        
        # Convert overlay back to RGB
        overlay_rgb = overlay[:, :, ::-1]  # BGR to RGB
        
        return overlay_rgb
        
    except Exception as e:
        print(f"❌ Grad-CAM error for {img_path}: {e}")
        return None


def create_gradcam_grid():
    """Create a grid showing 5 fake vs 5 real images with Grad-CAM"""
    
    print("🔍 Collecting images...")
    
    # Get random images
    real_imgs = glob.glob("split_dataset/test/real/*.jpg")
    fake_imgs = glob.glob("split_dataset/test/fake/*.jpg")
    
    if len(real_imgs) < 5 or len(fake_imgs) < 5:
        # Try final_dataset if split_dataset doesn't have enough
        real_imgs = glob.glob("final_dataset/real/*.jpg")
        fake_imgs = glob.glob("final_dataset/fake/*.jpg")
    
    if len(real_imgs) < 5 or len(fake_imgs) < 5:
        print("❌ Not enough images found. Need at least 5 real and 5 fake images.")
        return
    
    # Select 5 random images from each category
    selected_real = random.sample(real_imgs, 5)
    selected_fake = random.sample(fake_imgs, 5)
    
    all_images = selected_real + selected_fake
    labels = ["REAL"] * 5 + ["FAKE"] * 5
    
    print("🎯 Generating predictions and Grad-CAM visualizations...")
    
    # Create the grid: 4 rows x 10 columns
    # Row 1: Original real images (5)
    # Row 2: Grad-CAM real images (5) 
    # Row 3: Original fake images (5)
    # Row 4: Grad-CAM fake images (5)
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    
    # Process real images
    for i, img_path in enumerate(selected_real):
        # Get prediction
        label, prob = predict_image_standalone(img_path)
        
        # Load original image
        orig_img = Image.open(img_path)
        
        # Generate Grad-CAM
        gradcam_img = generate_gradcam_for_image(img_path)
        
        # Plot original image (top row)
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"REAL\nPred: {label}\n({prob:.3f})", fontsize=10, 
                            color="green" if label.lower() == "real" else "red")
        axes[0, i].axis('off')
        
        # Plot Grad-CAM (second row)
        if gradcam_img is not None:
            axes[1, i].imshow(gradcam_img)
        else:
            axes[1, i].imshow(np.zeros((299, 299, 3)))
        axes[1, i].set_title("Grad-CAM", fontsize=10)
        axes[1, i].axis('off')
        
        print(f"✅ Real image {i+1}: {Path(img_path).name} -> {label} ({prob:.3f})")
    
    # Process fake images
    for i, img_path in enumerate(selected_fake):
        # Get prediction
        label, prob = predict_image_standalone(img_path)
        
        # Load original image
        orig_img = Image.open(img_path)
        
        # Generate Grad-CAM
        gradcam_img = generate_gradcam_for_image(img_path)
        
        # Plot original image (third row)
        axes[2, i].imshow(orig_img)
        axes[2, i].set_title(f"FAKE\nPred: {label}\n({prob:.3f})", fontsize=10,
                            color="green" if label.lower() == "fake" else "red")
        axes[2, i].axis('off')
        
        # Plot Grad-CAM (fourth row)
        if gradcam_img is not None:
            axes[3, i].imshow(gradcam_img)
        else:
            axes[3, i].imshow(np.zeros((299, 299, 3)))
        axes[3, i].set_title("Grad-CAM", fontsize=10)
        axes[3, i].axis('off')
        
        print(f"✅ Fake image {i+1}: {Path(img_path).name} -> {label} ({prob:.3f})")
    
    # Add row labels
    fig.text(0.02, 0.85, "REAL\nImages", fontsize=14, fontweight='bold', 
             ha='center', va='center', rotation=90, color='green')
    fig.text(0.02, 0.65, "REAL\nGrad-CAM", fontsize=14, fontweight='bold',
             ha='center', va='center', rotation=90, color='green')
    fig.text(0.02, 0.35, "FAKE\nImages", fontsize=14, fontweight='bold',
             ha='center', va='center', rotation=90, color='red') 
    fig.text(0.02, 0.15, "FAKE\nGrad-CAM", fontsize=14, fontweight='bold',
             ha='center', va='center', rotation=90, color='red')
    
    plt.suptitle("Deepfake Detection: 5 Real vs 5 Fake Images with Grad-CAM Visualization", 
                 fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.02)
    
    # Save the grid
    output_file = "gradcam_comparison_grid.jpg"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved grid visualization: {output_file}")
    
    # Show the plot
    plt.show()
    
    return output_file


if __name__ == "__main__":
    print("🎯 Creating 5 Real vs 5 Fake Grad-CAM Grid...")
    
    if not os.path.exists("./best_xception.pth"):
        print("❌ best_xception.pth not found in current directory")
        exit(1)
    
    create_gradcam_grid()
    print("\n🎉 Grid visualization complete!")