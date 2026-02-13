#!/usr/bin/env python3
"""
test_image_gradcam.py

Standalone script to test images with your trained Xception model and Grad-CAM visualization.
Works independently of the notebook - just run this script directly.

Usage:
    python test_image_gradcam.py --image path/to/image.jpg
    python test_image_gradcam.py --random-test  # Test random image from dataset
    python test_image_gradcam.py --batch-test   # Test multiple images
"""

import argparse
import os
import random
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import timm
import torchvision.transforms as T

# Import our Grad-CAM tools
try:
    from xception_model import load_trained_model
    from gradcam import GradCAM, load_image, overlay_heatmap_on_image
    gradcam_available = True
    print("✅ Grad-CAM tools imported successfully")
except ImportError as e:
    gradcam_available = False
    print(f"⚠️ Grad-CAM tools not available: {e}")


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
        
        return label, prob, ckpt
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "unknown", 0.0, None


def test_image_with_gradcam(img_path, show_gradcam=True, target_layer="conv4", save_output=False):
    """
    Test an image with prediction and optional Grad-CAM visualization.
    
    Args:
        img_path: Path to the image file
        show_gradcam: Whether to show Grad-CAM visualization
        target_layer: Layer name for Grad-CAM
        save_output: Whether to save the visualization to file
    """
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None
    
    print(f"🔍 Testing image: {img_path}")
    
    # Make prediction
    label, prob, ckpt = predict_image_standalone(img_path)
    
    if ckpt is None:
        print("❌ Failed to load model")
        return None
    
    # Load and display original image
    img = Image.open(img_path)
    
    if show_gradcam and gradcam_available:
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
            
            # Convert overlay back to RGB for display
            overlay_rgb = overlay[:, :, ::-1]  # BGR to RGB
            
            # Show side-by-side comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(img)
            ax1.set_title("Original Image", fontsize=14)
            ax1.axis('off')
            
            ax2.imshow(overlay_rgb)
            ax2.set_title("Grad-CAM Attention Heatmap", fontsize=14)
            ax2.axis('off')
            
            color = "red" if label.lower() in ["fake", "deepfake"] else "green"
            fig.suptitle(f"Prediction: {label.upper()} ({prob:.3f})", fontsize=16, color=color)
            
            if save_output:
                output_name = f"{Path(img_path).stem}_gradcam_result.jpg"
                plt.savefig(output_name, dpi=150, bbox_inches='tight')
                print(f"💾 Saved visualization: {output_name}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Grad-CAM error: {e}")
            # Fallback to simple display
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.axis("off")
            color = "red" if label.lower() in ["fake", "deepfake"] else "green"
            plt.title(f"Prediction: {label.upper()} ({prob:.3f})", fontsize=14, color=color)
            plt.show()
    else:
        # Simple display without Grad-CAM
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis("off")
        color = "red" if label.lower() in ["fake", "deepfake"] else "green"
        plt.title(f"Prediction: {label.upper()} ({prob:.3f})", fontsize=14, color=color)
        plt.show()
    
    print(f"✅ Prediction: {label.upper()} | Confidence: {prob:.3f}")
    if show_gradcam and gradcam_available:
        print("🎯 Grad-CAM shows model attention (red=high, blue=low)")
    
    return {"label": label, "confidence": prob, "image_path": img_path}


def test_random_image():
    """Test a random image from the test dataset"""
    choice = random.choice(["real", "fake"])
    test_dir = f"split_dataset/test/{choice}"
    
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if images:
            random_img = random.choice(images)
            img_path = os.path.join(test_dir, random_img)
            print(f"🎲 Testing random {choice} image: {random_img}")
            return test_image_with_gradcam(img_path, show_gradcam=True, save_output=True)
        else:
            print(f"No images found in {test_dir}")
    else:
        print(f"Directory not found: {test_dir}")
    return None


def test_multiple_images(num_images=4):
    """Test multiple images from the dataset"""
    print(f"🧪 Testing {num_images} images from the test dataset...")
    
    real_imgs = glob.glob("split_dataset/test/real/*.jpg")[:num_images//2]
    fake_imgs = glob.glob("split_dataset/test/fake/*.jpg")[:num_images//2]
    
    results = []
    
    for i, img_path in enumerate(real_imgs):
        print(f"\n--- Real Image {i+1} ---")
        result = test_image_with_gradcam(img_path, show_gradcam=True)
        if result:
            results.append(result)
    
    for i, img_path in enumerate(fake_imgs):
        print(f"\n--- Fake Image {i+1} ---")
        result = test_image_with_gradcam(img_path, show_gradcam=True)
        if result:
            results.append(result)
    
    # Summary
    if results:
        correct = sum(1 for r in results if 
                     (r["label"].lower() in ["real"] and "real" in r["image_path"]) or
                     (r["label"].lower() in ["fake"] and "fake" in r["image_path"]))
        print(f"\n📊 Summary: {correct}/{len(results)} correct predictions ({correct/len(results)*100:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test images with Xception + Grad-CAM")
    parser.add_argument("--image", type=str, help="Path to specific image to test")
    parser.add_argument("--random-test", action="store_true", help="Test a random image from dataset")
    parser.add_argument("--batch-test", action="store_true", help="Test multiple images")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images for batch test")
    parser.add_argument("--target-layer", type=str, default="conv4", help="Target layer for Grad-CAM")
    parser.add_argument("--save", action="store_true", help="Save visualization outputs")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists("./best_xception.pth"):
        print("❌ best_xception.pth not found in current directory")
        print("   Please ensure the checkpoint file is available")
        return
    
    print("✅ Found checkpoint: best_xception.pth")
    
    if args.image:
        # Test specific image
        test_image_with_gradcam(args.image, show_gradcam=True, 
                               target_layer=args.target_layer, save_output=args.save)
    elif args.random_test:
        # Test random image
        test_random_image()
    elif args.batch_test:
        # Test multiple images
        test_multiple_images(args.num_images)
    else:
        # Interactive mode - ask what to do
        print("\n🎯 Xception Deepfake Detection + Grad-CAM Tester")
        print("=" * 50)
        print("1. Test specific image")
        print("2. Test random image from dataset") 
        print("3. Test multiple images")
        print("4. Exit")
        
        while True:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                img_path = input("Enter image path: ").strip()
                test_image_with_gradcam(img_path, show_gradcam=True, save_output=True)
            elif choice == "2":
                test_random_image()
            elif choice == "3":
                num = input("How many images to test? (default 4): ").strip()
                num = int(num) if num.isdigit() else 4
                test_multiple_images(num)
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()