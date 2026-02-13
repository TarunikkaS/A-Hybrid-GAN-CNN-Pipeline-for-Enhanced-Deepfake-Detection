#!/usr/bin/env python3
"""
test_original_gradcam.py

Test images with the original best_xception.pth model and Grad-CAM visualization.
For comparison with the FFPP2 retrained model.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import cv2

# Import our Grad-CAM tools
try:
    from gradcam import GradCAM, overlay_heatmap_on_image, load_image
    from xception_model import load_trained_model
    print("✅ Grad-CAM tools imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def predict_image_original(image_path, save_viz=False):
    """Predict image using original best_xception.pth model and generate Grad-CAM"""
    try:
        # Load original model
        print("🔄 Loading original best_xception.pth model...")
        model, metadata = load_trained_model('best_xception.pth')
        if model is None:
            print("❌ Failed to load original model")
            return None
        
        print(f"✅ Original model loaded successfully: {type(model)}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess for model
        import torchvision.transforms as transforms
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
        viz_image = overlay_heatmap_on_image(image_rgb, cam)
        
        # Display results
        print(f"🎯 Prediction: {prediction:.6f}")
        print(f"📊 Classification: {'FAKE' if prediction < 0.5 else 'REAL'}")
        print(f"📈 Confidence: {(1-prediction)*100 if prediction < 0.5 else prediction*100:.2f}%")
        
        if save_viz:
            # Save visualization
            filename = f"original_{os.path.basename(image_path).split('.')[0]}_gradcam_result.jpg"
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
            
        # Return in format expected by test function
        label = 'FAKE' if prediction < 0.5 else 'REAL'
        return label, prediction, 'best_xception.pth'
        
    except Exception as e:
        import traceback
        print(f"❌ Prediction error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

def test_image_with_gradcam_original(img_path, show_gradcam=True, save_viz=False):
    """Test a single image with original model and show Grad-CAM visualization"""
    
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return
        
    print(f"🔍 Testing image with original model: {img_path}")
    
    # Get prediction and Grad-CAM
    result = predict_image_original(img_path, save_viz=save_viz)
    if result is None:
        print("❌ Failed to process image")
        return
        
    label, prob, checkpoint = result
    print(f"✅ Original Model Prediction: {label} | Confidence: {prob:.3f}")
    print(f"🎯 Grad-CAM shows model attention (red=high, blue=low)")

def main():
    parser = argparse.ArgumentParser(description='Test images with original best_xception.pth model and Grad-CAM')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--save', action='store_true', help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists('best_xception.pth'):
        print("❌ best_xception.pth not found!")
        return
    
    print("✅ Found original checkpoint: best_xception.pth")
    
    # Test the image
    test_image_with_gradcam_original(args.image, show_gradcam=True, save_viz=args.save)

if __name__ == "__main__":
    main()