#!/usr/bin/env python3
"""
test_ffpp2_gradcam.py

Test images with the new FFPP2.pt model and Grad-CAM visualization.
Uses the retrained model from XCEPTION_NET(RETRAINING).ipynb
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
import torchvision.transforms as transforms
import cv2

# Import our Grad-CAM tools
try:
    from gradcam import GradCAM, load_image, overlay_heatmap_on_image
    gradcam_available = True
    print("✅ Grad-CAM tools imported successfully")
except ImportError as e:
    gradcam_available = False
    print(f"⚠️ Grad-CAM tools not available: {e}")


def predict_image_ffpp2(image_path, save_viz=False):
    """Predict image using FFPP2 model and generate Grad-CAM"""
    try:
        # Load FFPP2 model
        print("🔄 Loading FFPP2 model...")
        model = load_ffpp2_model()
        if model is None:
            return
        
        print(f"✅ FFPP2 model loaded successfully: {type(model)}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image_rgb).unsqueeze(0)
        
        # Make prediction
        print("🔄 Making prediction...")
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output).item()
        
        print("✅ Prediction complete, generating Grad-CAM...")
        
        # Generate Grad-CAM
        gradcam = GradCAM(model, target_layer_name='features')
        cam = gradcam.generate_cam(input_tensor, target_class=None)
        
        # Create visualization
        from gradcam import overlay_heatmap_on_image
        viz_image = overlay_heatmap_on_image(image_rgb, cam)
        
        # Display results
        print(f"🎯 Prediction: {prediction:.6f}")
        print(f"📊 Classification: {'FAKE' if prediction < 0.5 else 'REAL'}")
        print(f"📈 Confidence: {(1-prediction)*100 if prediction < 0.5 else prediction*100:.2f}%")
        
        if save_viz:
            # Save visualization
            filename = f"ffpp2_gradcam_{os.path.basename(image_path)}"
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image_rgb)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cam, cmap='jet')
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(viz_image)
            plt.title("Overlay Visualization")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved: {filename}")
            
        # Return in format expected by test_image_with_gradcam_ffpp2
        label = 'FAKE' if prediction < 0.5 else 'REAL'
        return label, prediction, 'FFPP2.pt'
        
    except Exception as e:
        import traceback
        print(f"❌ Prediction error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None


def load_ffpp2_model():
    """Load the FFPP2 retrained model"""
    try:
        # Load with weights_only=False to handle full model objects
        checkpoint = torch.load('FFPP2.pt', map_location='cpu', weights_only=False)
        
        # Handle FFPP2 checkpoint format
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                # Use the pre-trained model directly
                model = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # Create new model and load state dict
                model = timm.create_model('xception', num_classes=1, pretrained=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model = timm.create_model('xception', num_classes=1, pretrained=False)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Direct state dict
                model = timm.create_model('xception', num_classes=1, pretrained=False)
                model.load_state_dict(checkpoint)
        else:
            # Complete model object - use it directly
            model = checkpoint
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Error loading FFPP2 model: {e}")
        return None


def test_image_with_gradcam_ffpp2(img_path, show_gradcam=True, target_layer="conv4", save_output=False):
    """
    Test an image with FFPP2 model prediction and optional Grad-CAM visualization.
    """
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None
    
    print(f"🔍 Testing image with FFPP2 model: {img_path}")
    
    # Make prediction
    label, prob, ckpt = predict_image_ffpp2(img_path)
    
    if ckpt is None:
        print("❌ Failed to load FFPP2 model")
        return None
    
    # Load and display original image
    img = Image.open(img_path)
    
    if show_gradcam and gradcam_available:
        try:
            # Load FFPP2 model for Grad-CAM
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gradcam_model = load_ffpp2_model()
            if gradcam_model is None:
                print("❌ Failed to load FFPP2 model for Grad-CAM")
                return None
            gradcam_model = gradcam_model.to(device)
            
            # Create Grad-CAM instance
            gradcam = GradCAM(gradcam_model, target_layer)
            
            # Process image for Grad-CAM (use standard 299x299 for Xception)
            from gradcam import load_image
            tensor, orig_bgr = load_image(img_path, 299)
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
            fig.suptitle(f"FFPP2 Model Prediction: {label.upper()} ({prob:.3f})", fontsize=16, color=color)
            
            if save_output:
                output_name = f"{Path(img_path).stem}_ffpp2_gradcam_result.jpg"
                plt.savefig(output_name, dpi=150, bbox_inches='tight')
                print(f"💾 Saved FFPP2 visualization: {output_name}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Grad-CAM error: {e}")
            # Fallback to simple display
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.axis("off")
            color = "red" if label.lower() in ["fake", "deepfake"] else "green"
            plt.title(f"FFPP2 Prediction: {label.upper()} ({prob:.3f})", fontsize=14, color=color)
            plt.show()
    else:
        # Simple display without Grad-CAM
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.axis("off")
        color = "red" if label.lower() in ["fake", "deepfake"] else "green"
        plt.title(f"FFPP2 Prediction: {label.upper()} ({prob:.3f})", fontsize=14, color=color)
        plt.show()
    
    print(f"✅ FFPP2 Prediction: {label.upper()} | Confidence: {prob:.3f}")
    if show_gradcam and gradcam_available:
        print("🎯 Grad-CAM shows model attention (red=high, blue=low)")
    
    return {"label": label, "confidence": prob, "image_path": img_path}


def main():
    parser = argparse.ArgumentParser(description="Test images with FFPP2 model + Grad-CAM")
    parser.add_argument("--image", type=str, help="Path to specific image to test")
    parser.add_argument("--target-layer", type=str, default="conv4", help="Target layer for Grad-CAM")
    parser.add_argument("--save", action="store_true", help="Save visualization outputs")
    
    args = parser.parse_args()
    
    # Check if FFPP2 checkpoint exists
    if not os.path.exists("./FFPP2.pt"):
        print("❌ FFPP2.pt not found in current directory")
        print("   Please ensure the FFPP2 checkpoint file is available")
        return
    
    print("✅ Found FFPP2 checkpoint: FFPP2.pt")
    
    if args.image:
        # Test specific image
        test_image_with_gradcam_ffpp2(args.image, show_gradcam=True, 
                                     target_layer=args.target_layer, save_output=args.save)
    else:
        # Interactive mode
        print("\n🎯 FFPP2 Deepfake Detection + Grad-CAM Tester")
        print("=" * 50)
        img_path = input("Enter image path: ").strip()
        test_image_with_gradcam_ffpp2(img_path, show_gradcam=True, save_output=True)


if __name__ == "__main__":
    main()