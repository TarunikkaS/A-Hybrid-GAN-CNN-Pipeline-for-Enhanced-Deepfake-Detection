#!/usr/bin/env python3
"""
Local Manipulation Attribution via Region-Wise Occlusion Sensitivity
=====================================================================
Forensic module for identifying which facial regions most contribute to
deepfake detection decisions through systematic occlusion analysis.

Model: xception_gan_augmented.pth (Xception, num_classes=1)
Method: Occlusion-based attribution (no gradients, inference only)
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import timm
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Facial region definitions (relative coordinates for 299x299 images)
FACE_REGIONS = {
    'forehead': (0.2, 0.0, 0.8, 0.25),      # (x1, y1, x2, y2) as fractions
    'left_eye': (0.2, 0.25, 0.45, 0.4),
    'right_eye': (0.55, 0.25, 0.8, 0.4),
    'nose': (0.4, 0.35, 0.6, 0.55),
    'left_cheek': (0.1, 0.4, 0.35, 0.65),
    'right_cheek': (0.65, 0.4, 0.9, 0.65),
    'mouth': (0.3, 0.6, 0.7, 0.75),
    'jawline': (0.15, 0.7, 0.85, 0.95),
}

OUTPUT_DIR = "./attribution_results"

# ============================================================================
# Model Loading
# ============================================================================

def load_model(checkpoint_path: str) -> nn.Module:
    """Load pretrained Xception model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = timm.create_model('xception', pretrained=False, num_classes=1)
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict') or checkpoint.get('model', checkpoint)
        if not isinstance(state_dict, dict):
            state_dict = state_dict.state_dict() if hasattr(state_dict, 'state_dict') else checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()
    
    return model

# ============================================================================
# Preprocessing
# ============================================================================

def get_transform():
    """Standard ImageNet preprocessing for Xception."""
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(model: nn.Module, img: Image.Image) -> float:
    """Run inference on PIL image, return P(fake)."""
    transform = get_transform()
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logit = model(tensor).squeeze()
        p_fake = torch.sigmoid(logit).item()
    
    return p_fake

# ============================================================================
# Region Occlusion
# ============================================================================

def get_region_bbox(region_coords: Tuple[float, float, float, float], img_size: int = 299) -> Tuple[int, int, int, int]:
    """Convert relative coordinates to absolute pixel coordinates."""
    x1, y1, x2, y2 = region_coords
    return (
        int(x1 * img_size),
        int(y1 * img_size),
        int(x2 * img_size),
        int(y2 * img_size)
    )

def occlude_region(img: Image.Image, region_coords: Tuple[float, float, float, float], method: str = 'blur') -> Image.Image:
    """
    Create occluded version by masking specified region.
    
    Args:
        img: PIL Image (RGB)
        region_coords: (x1, y1, x2, y2) as fractions
        method: 'blur' or 'mean' for occlusion type
        
    Returns:
        Occluded PIL Image
    """
    img_copy = img.copy()
    bbox = get_region_bbox(region_coords, img.size[0])
    
    if method == 'blur':
        # Extract region, blur it heavily, paste back
        region = img_copy.crop(bbox)
        blurred = region.filter(ImageFilter.GaussianBlur(radius=15))
        img_copy.paste(blurred, bbox)
    elif method == 'mean':
        # Fill region with mean color
        region = img_copy.crop(bbox)
        mean_color = tuple(int(c) for c in np.array(region).mean(axis=(0, 1)))
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle(bbox, fill=mean_color)
    
    return img_copy

# ============================================================================
# Attribution Analysis
# ============================================================================

def local_manipulation_attribution(model: nn.Module, img_path: str, occlusion_method: str = 'blur') -> Dict:
    """
    Perform region-wise occlusion sensitivity analysis.
    
    Systematically occludes each facial region and measures impact on
    deepfake prediction to identify manipulation-sensitive areas.
    
    Args:
        model: Loaded Xception model in eval mode
        img_path: Path to face image
        occlusion_method: 'blur' or 'mean' for region masking
        
    Returns:
        dict with attribution results and metadata
    """
    img = Image.open(img_path).convert('RGB').resize((299, 299))
    
    # Get original prediction
    p_fake_original = predict_image(model, img)
    label_original = "FAKE" if p_fake_original > 0.5 else "REAL"
    
    # Analyze each region
    region_results = []
    
    for region_name, region_coords in FACE_REGIONS.items():
        # Occlude region
        occluded_img = occlude_region(img, region_coords, method=occlusion_method)
        
        # Predict on occluded version
        p_fake_occluded = predict_image(model, occluded_img)
        
        # Compute attribution: positive Δ means region contributes to fake detection
        delta = p_fake_original - p_fake_occluded
        
        region_results.append({
            'region': region_name,
            'coords': region_coords,
            'p_fake_occluded': p_fake_occluded,
            'delta': delta,
            'abs_delta': abs(delta)
        })
    
    # Rank by absolute contribution
    region_results.sort(key=lambda x: x['abs_delta'], reverse=True)
    
    return {
        'img_path': img_path,
        'original_prediction': {
            'p_fake': p_fake_original,
            'label': label_original
        },
        'regions': region_results,
        'occlusion_method': occlusion_method
    }

# ============================================================================
# Visualization
# ============================================================================

def create_attribution_map(img_path: str, attribution_results: Dict, output_path: str):
    """
    Create Grad-CAM-style heatmap showing ONLY evidence supporting the final decision.
    
    Filters regions based on final prediction:
    - FAKE: show only positive Δ (supports fake)
    - REAL: show only negative Δ (supports real)
    """
    import cv2
    
    # Load image
    img = Image.open(img_path).convert('RGB').resize((299, 299))
    img_array = np.array(img)
    
    # Determine final prediction
    original_pred = attribution_results['original_prediction']
    p_fake = original_pred['p_fake']
    is_fake = p_fake >= 0.5
    
    # Filter regions based on final decision
    if is_fake:
        # Show only regions that support FAKE (positive Δ)
        relevant_regions = [r for r in attribution_results['regions'] if r['delta'] > 0]
        colormap = cv2.COLORMAP_HOT  # Warm colors for FAKE
        evidence_type = "FAKE"
    else:
        # Show only regions that support REAL (negative Δ)
        relevant_regions = [r for r in attribution_results['regions'] if r['delta'] < 0]
        colormap = cv2.COLORMAP_WINTER  # Cool colors for REAL
        evidence_type = "REAL"
    
    # Initialize heatmap
    heatmap = np.zeros((299, 299), dtype=np.float32)
    
    # Generate Gaussian blob for each relevant region
    if len(relevant_regions) > 0:
        for result in relevant_regions:
            x1, y1, x2, y2 = get_region_bbox(result['coords'], 299)
            
            # Center point of region
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Gaussian spread
            sigma_x = (x2 - x1) / 2.5
            sigma_y = (y2 - y1) / 2.5
            
            # Generate 2D Gaussian blob
            y_coords, x_coords = np.ogrid[:299, :299]
            gaussian = np.exp(-((x_coords - cx)**2 / (2 * sigma_x**2) + 
                               (y_coords - cy)**2 / (2 * sigma_y**2)))
            
            # Scale by attribution strength (use absolute value)
            contribution = abs(result['delta'])
            heatmap += gaussian * contribution
        
        # Normalize heatmap to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Alpha blend with original image
    alpha = 0.5
    blended = (img_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    
    # Create final image with text panel
    panel_width = 220
    final_width = 299 + panel_width
    final_img = np.ones((299, final_width, 3), dtype=np.uint8) * 255
    
    # Place blended heatmap
    final_img[:, :299] = blended
    
    # Convert to PIL for text rendering
    final_pil = Image.fromarray(final_img)
    draw = ImageDraw.Draw(final_pil)
    
    # Add prediction badge on top-left corner of heatmap
    badge_x, badge_y = 10, 10
    badge_width, badge_height = 180, 70
    
    # Badge color based on prediction
    if is_fake:
        badge_color = (220, 50, 50, 200)  # Red
        text_color = (255, 255, 255)
    else:
        badge_color = (50, 200, 80, 200)  # Green
        text_color = (255, 255, 255)
    
    # Draw semi-transparent badge background
    badge_overlay = Image.new('RGBA', final_pil.size, (0, 0, 0, 0))
    badge_draw = ImageDraw.Draw(badge_overlay)
    badge_draw.rounded_rectangle(
        [(badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height)],
        radius=8,
        fill=badge_color
    )
    final_pil = Image.alpha_composite(final_pil.convert('RGBA'), badge_overlay)
    draw = ImageDraw.Draw(final_pil)
    
    # Add text panel
    try:
        from PIL import ImageFont
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_badge_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        font_badge_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_badge_title = ImageFont.load_default()
        font_badge_label = ImageFont.load_default()
    
    # Draw badge content
    draw.text((badge_x + 10, badge_y + 5), "FINAL PREDICTION", fill=text_color, font=font_badge_title)
    draw.text((badge_x + 15, badge_y + 22), original_pred['label'], fill=text_color, font=font_badge_label)
    conf_text = f"P(fake) = {original_pred['p_fake']:.4f}"
    draw.text((badge_x + 10, badge_y + 50), conf_text, fill=text_color, font=font_badge_title)
    
    # Right panel title
    panel_title = f"Local Evidence"
    draw.text((310, 20), panel_title, fill=(0, 0, 0), font=font_large)
    draw.text((310, 38), f"Supporting {evidence_type}", fill=(0, 0, 0), font=font_small)
    
    # Top 3 relevant regions
    y_offset = 70
    top_regions = relevant_regions[:3] if len(relevant_regions) > 0 else []
    
    if len(top_regions) == 0:
        # No evidence found (edge case)
        draw.text((310, y_offset), "No strong evidence", fill=(100, 100, 100), font=font_small)
        draw.text((310, y_offset + 15), "detected in regions", fill=(100, 100, 100), font=font_small)
    else:
        for i, region in enumerate(top_regions, 1):
            # Region name
            region_name = region['region'].replace('_', ' ').title()
            text = f"#{i}: {region_name}"
            draw.text((310, y_offset), text, fill=(0, 0, 0), font=font_small)
            
            # Delta value
            delta_text = f"Δ = {region['delta']:+.4f}"
            draw.text((320, y_offset + 18), delta_text, fill=(100, 100, 100), font=font_small)
            
            y_offset += 50
    
    # Strongest indicator callout
    if len(relevant_regions) > 0:
        strongest = relevant_regions[0]
        strongest_name = strongest['region'].replace('_', ' ').title()
        draw.text((310, 220), "Strongest Indicator:", fill=(0, 0, 0), font=font_small)
        
        # Color based on evidence type
        indicator_color = (200, 0, 0) if is_fake else (0, 150, 0)
        draw.text((315, 240), strongest_name, fill=indicator_color, font=font_large)
        draw.text((315, 258), f"Δ = {strongest['delta']:+.4f}", fill=(100, 100, 100), font=font_small)
    
    # Color legend
    draw.text((310, 280), "Heatmap Colors:", fill=(0, 0, 0), font=font_small)
    if is_fake:
        draw.text((315, 293), "🔴 Red: Fake evidence", fill=(0, 0, 0), font=font_small)
    else:
        draw.text((315, 293), "🔵 Blue: Real evidence", fill=(0, 0, 0), font=font_small)
    
    # Convert back to RGB before saving as JPEG
    final_pil = final_pil.convert('RGB')
    
    # Save
    final_pil.save(output_path, quality=95)

def _hsv_to_rgb(h: int, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV to RGB color space."""
    import colorsys
    h_norm = h / 360.0
    r, g, b = colorsys.hsv_to_rgb(h_norm, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

# ============================================================================
# Reporting
# ============================================================================

def print_attribution_report(results: Dict):
    """Print formatted attribution analysis report."""
    print("\n" + "=" * 80)
    print("LOCAL MANIPULATION ATTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"📁 Image: {results['img_path']}")
    print(f"🎯 Original Prediction: {results['original_prediction']['label']} "
          f"(P(fake) = {results['original_prediction']['p_fake']:.4f})")
    print(f"🔧 Occlusion Method: {results['occlusion_method']}")
    
    print(f"\n📊 REGION-WISE ATTRIBUTION (ranked by contribution)")
    print(f"   {'Rank':<6} {'Region':<15} {'P(fake)_occluded':<18} {'Δ Contribution':<18} {'Effect'}")
    print(f"   {'-'*6} {'-'*15} {'-'*18} {'-'*18} {'-'*20}")
    
    for i, result in enumerate(results['regions'], 1):
        effect = "↑ Supports FAKE" if result['delta'] > 0 else "↓ Suppresses FAKE"
        print(f"   {i:<6} {result['region']:<15} {result['p_fake_occluded']:<18.6f} "
              f"{result['delta']:+18.6f} {effect}")
    
    print("\n💡 INTERPRETATION:")
    top_region = results['regions'][0]
    if top_region['delta'] > 0:
        print(f"   '{top_region['region']}' shows strongest fake indicators (Δ = {top_region['delta']:+.4f})")
        print(f"   Occluding this region reduces fake detection confidence")
    else:
        print(f"   '{top_region['region']}' suppresses fake detection (Δ = {top_region['delta']:+.4f})")
        print(f"   Occluding this region increases fake detection confidence")
    
    print("\n🎨 COLOR LEGEND:")
    print("   🔴 Red/Orange:  Strongest fake indicators (Rank 1-2)")
    print("   🟡 Yellow:      Moderate indicators (Rank 3-4)")
    print("   🟢 Green:       Weak indicators (Rank 5-6)")
    print("   🔵 Blue:        Weakest indicators (Rank 7-8)")
    
    print("=" * 80 + "\n")

# ============================================================================
# Main Pipeline
# ============================================================================

def run_attribution_analysis(model_path: str, img_path: str, save_viz: bool = True):
    """Execute complete local manipulation attribution pipeline."""
    
    # Load model
    print(f"🔧 Loading model: {Path(model_path).name}")
    print(f"📍 Device: {DEVICE}")
    model = load_model(model_path)
    print(f"✅ Model loaded\n")
    
    # Run attribution analysis
    print(f"🔍 Analyzing facial regions for manipulation indicators...")
    results = local_manipulation_attribution(model, img_path, occlusion_method='blur')
    
    # Print report
    print_attribution_report(results)
    
    # Save visualization
    if save_viz:
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        img_name = Path(img_path).stem
        viz_path = Path(OUTPUT_DIR) / f"{img_name}_attribution_map.jpg"
        create_attribution_map(img_path, results, str(viz_path))
        print(f"💾 Attribution map saved: {viz_path}\n")
    
    return results

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    parser = argparse.ArgumentParser(description="Local manipulation attribution analysis")
    parser.add_argument("image_path", nargs='?', 
                       default="/Users/tarunikkasuresh/Desktop/FINAL DEEPFAKE PROJECT MODEL/final_dataset/real/033_7.jpg",
                       help="Path to input image")
    parser.add_argument("--model", "-m", 
                       default="/Users/tarunikkasuresh/Desktop/FINAL DEEPFAKE PROJECT MODEL/xception_gan_augmented.pth",
                       help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    MODEL_PATH = args.model
    IMG_PATH = args.image_path
    
    # ========================================================================
    # Run Analysis
    # ========================================================================
    
    results = run_attribution_analysis(
        model_path=MODEL_PATH,
        img_path=IMG_PATH,
        save_viz=True
    )
    
    # ========================================================================
    # Batch Analysis Example (Optional)
    # ========================================================================
    
    """
    test_images = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    
    for img_path in test_images:
        results = run_attribution_analysis(MODEL_PATH, img_path, save_viz=True)
        print(f"Top contributing region: {results['regions'][0]['region']}\n")
    """
