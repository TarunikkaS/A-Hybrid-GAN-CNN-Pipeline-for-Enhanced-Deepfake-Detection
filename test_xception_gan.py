#!/usr/bin/env python3
"""
Test xception_gan_augmented.pth model on reduced_dataset
"""

import os
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_PATH = "/Users/tarunikkasuresh/Desktop/FINAL DEEPFAKE PROJECT MODEL/xception_gan_augmented.pth"
DATASET_PATH = "/Users/tarunikkasuresh/Desktop/FINAL DEEPFAKE PROJECT MODEL/reduced_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    """Load the xception_gan_augmented model"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = timm.create_model('xception', pretrained=False, num_classes=1)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            if isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint['model'].state_dict()
        else:
            state_dict = checkpoint
    else:
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded from {os.path.basename(checkpoint_path)}")
    print(f"📍 Device: {DEVICE}")
    
    return model

def get_transform():
    """Standard preprocessing for Xception"""
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_image(model, img_path):
    """Predict single image"""
    transform = get_transform()
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logit = model(tensor).squeeze()
        p_fake = torch.sigmoid(logit).item()
    
    label = "FAKE" if p_fake > 0.5 else "REAL"
    return label, p_fake

def test_dataset(model, dataset_path, split='test'):
    """Test model on a dataset split"""
    fake_dir = os.path.join(dataset_path, split, 'fake')
    real_dir = os.path.join(dataset_path, split, 'real')
    
    results = {
        'fake': {'correct': 0, 'total': 0, 'predictions': []},
        'real': {'correct': 0, 'total': 0, 'predictions': []}
    }
    
    # Test fake images
    if os.path.exists(fake_dir):
        fake_images = [f for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\n🔍 Testing {len(fake_images)} fake images...")
        
        for img_name in tqdm(fake_images, desc="Fake"):
            img_path = os.path.join(fake_dir, img_name)
            try:
                label, p_fake = predict_image(model, img_path)
                results['fake']['total'] += 1
                results['fake']['predictions'].append(p_fake)
                if label == 'FAKE':
                    results['fake']['correct'] += 1
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    
    # Test real images
    if os.path.exists(real_dir):
        real_images = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"\n🔍 Testing {len(real_images)} real images...")
        
        for img_name in tqdm(real_images, desc="Real"):
            img_path = os.path.join(real_dir, img_name)
            try:
                label, p_fake = predict_image(model, img_path)
                results['real']['total'] += 1
                results['real']['predictions'].append(p_fake)
                if label == 'REAL':
                    results['real']['correct'] += 1
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    
    return results

def print_results(results, split_name):
    """Print formatted results"""
    print("\n" + "="*70)
    print(f"RESULTS FOR {split_name.upper()} SET")
    print("="*70)
    
    # Fake images
    if results['fake']['total'] > 0:
        fake_acc = results['fake']['correct'] / results['fake']['total'] * 100
        fake_preds = results['fake']['predictions']
        print(f"\n📊 FAKE Images:")
        print(f"   Total: {results['fake']['total']}")
        print(f"   Correct: {results['fake']['correct']}")
        print(f"   Accuracy: {fake_acc:.2f}%")
        print(f"   Avg P(Fake): {np.mean(fake_preds):.4f} ± {np.std(fake_preds):.4f}")
        print(f"   Range: [{min(fake_preds):.4f}, {max(fake_preds):.4f}]")
    
    # Real images
    if results['real']['total'] > 0:
        real_acc = results['real']['correct'] / results['real']['total'] * 100
        real_preds = results['real']['predictions']
        print(f"\n📊 REAL Images:")
        print(f"   Total: {results['real']['total']}")
        print(f"   Correct: {results['real']['correct']}")
        print(f"   Accuracy: {real_acc:.2f}%")
        print(f"   Avg P(Fake): {np.mean(real_preds):.4f} ± {np.std(real_preds):.4f}")
        print(f"   Range: [{min(real_preds):.4f}, {max(real_preds):.4f}]")
    
    # Overall
    total = results['fake']['total'] + results['real']['total']
    correct = results['fake']['correct'] + results['real']['correct']
    if total > 0:
        overall_acc = correct / total * 100
        print(f"\n🎯 OVERALL:")
        print(f"   Total Images: {total}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {overall_acc:.2f}%")
    
    print("="*70 + "\n")

def main():
    # Load model
    print("\n🚀 Loading xception_gan_augmented model...")
    model = load_model(MODEL_PATH)
    
    # Test on test set
    print(f"\n📁 Dataset: {DATASET_PATH}")
    results_test = test_dataset(model, DATASET_PATH, split='test')
    print_results(results_test, 'test')
    
    # Test on validation set if exists
    val_fake = os.path.join(DATASET_PATH, 'val', 'fake')
    val_real = os.path.join(DATASET_PATH, 'val', 'real')
    if os.path.exists(val_fake) or os.path.exists(val_real):
        results_val = test_dataset(model, DATASET_PATH, split='val')
        print_results(results_val, 'validation')

if __name__ == "__main__":
    main()
