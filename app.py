"""
Trust-Aware Deepfake Detector - Streamlit App

A comprehensive deepfake detection application with:
- Image and video analysis
- Grad-CAM visual explanations
- Multiple verdict methods for video
- Suspicion timeline visualization
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path

# Import custom modules
from model_def import build_model, load_checkpoint
from gradcam_utils import GradCAM, overlay_cam_on_image

# ============================================================================
# CONFIGURATION
# ============================================================================

# Image preprocessing constants (MUST match training)
IMG_SIZE = 299
MEAN = (0.485, 0.456, 0.406)  # ImageNet normalization
STD = (0.229, 0.224, 0.225)   # ImageNet normalization

# Class labels
LABELS = ["REAL", "FAKE"]

# Model checkpoint path
CHECKPOINT_PATH = "weights/xception_gan_augmented.pth"

# Grad-CAM alpha blending
GRADCAM_ALPHA = 0.5

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


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(checkpoint_path):
    """
    Load the deepfake detection model.
    
    Cached to avoid reloading on every interaction.
    
    Args:
        checkpoint_path: Path to model checkpoint
    
    Returns:
        tuple: (model, device)
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with st.spinner("🔧 Loading model..."):
        try:
            # Build model architecture (num_classes=1 for binary with sigmoid)
            model = build_model(num_classes=1)
            
            # Load checkpoint
            model = load_checkpoint(model, checkpoint_path, device=device)
            
            # Set to evaluation mode
            model.eval()
            model.to(device)
            
            st.success(f"✅ Model loaded on {device}")
            return model, device
            
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.info("""
            **Troubleshooting:**
            1. Ensure `weights/xception_gan_augmented.pth` exists
            2. Update `model_def.py` to match your training architecture
            3. Check that checkpoint format matches (state_dict vs full model)
            """)
            raise


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_pil(pil_image, img_size=IMG_SIZE, mean=MEAN, std=STD):
    """
    Preprocess a PIL image for model input.
    
    Args:
        pil_image: PIL Image in any mode
        img_size: Target size (square)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        torch.Tensor: Preprocessed image (1, 3, H, W) as float32
    """
    # Convert to RGB
    img = pil_image.convert('RGB')
    
    # Resize
    img = img.resize((img_size, img_size), Image.BILINEAR)
    
    # To numpy array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Convert mean and std to numpy arrays
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    # Normalize using mean and std
    img_array = (img_array - mean) / std
    
    # Convert to CHW format
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


# ============================================================================
# PREDICTION
# ============================================================================

def predict_proba(model, input_tensor, device):
    """
    Get prediction probabilities for input.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed image (1, 3, H, W)
        device: torch.device
    
    Returns:
        tuple: (probs, label, confidence)
            - probs: numpy array [p_real, p_fake]
            - label: predicted class string
            - confidence: confidence score [0, 1]
    """
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass - model outputs single logit
        logit = model(input_tensor).squeeze()  # scalar
        
        # Apply sigmoid to get P(fake)
        p_fake = torch.sigmoid(logit).cpu().item()
        p_real = 1 - p_fake
        
        # Create probability array [P(real), P(fake)]
        probs = np.array([p_real, p_fake])
        
        # Get prediction
        pred_idx = np.argmax(probs)
        label = LABELS[pred_idx]
        confidence = probs[pred_idx]
    
    return probs, label, confidence


# ============================================================================
# GRAD-CAM
# ============================================================================

def analyze_facial_regions(model, pil_image, device, original_prob):
    """
    Analyze which facial regions contribute most to the prediction via occlusion.
    
    Args:
        model: PyTorch model
        pil_image: PIL Image (will be resized to 299x299)
        device: torch.device
        original_prob: Original P(fake) before occlusion
    
    Returns:
        List of dicts with keys: region, delta, occluded_prob
        Sorted by absolute delta (impact on prediction)
    """
    results = []
    img_resized = pil_image.resize((299, 299), Image.BILINEAR)
    
    for region_name, (x1, y1, x2, y2) in FACE_REGIONS.items():
        # Create occluded version
        img_copy = img_resized.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Convert relative coords to absolute
        x1_abs = int(x1 * 299)
        y1_abs = int(y1 * 299)
        x2_abs = int(x2 * 299)
        y2_abs = int(y2 * 299)
        
        # Draw gray rectangle to occlude region
        draw.rectangle([x1_abs, y1_abs, x2_abs, y2_abs], fill=(128, 128, 128))
        
        # Get prediction with occluded region
        occluded_tensor = preprocess_pil(img_copy)
        with torch.no_grad():
            probs, _, _ = predict_proba(model, occluded_tensor, device)
        occluded_prob = probs[1]  # P(fake)
        
        # Delta: positive means region contributes to FAKE, negative means REAL
        delta = original_prob - occluded_prob
        
        results.append({
            'region': region_name,
            'delta': delta,
            'occluded_prob': occluded_prob
        })
    
    # Sort by absolute impact
    results.sort(key=lambda x: abs(x['delta']), reverse=True)
    return results


def generate_gradcam(model, input_tensor, device):
    """
    Generate Grad-CAM heatmap for input.
    
    For single-output models, we don't need class_idx - we just backprop the output.
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed image (1, 3, H, W)
        device: torch.device
    
    Returns:
        cam: Activation map (H, W) normalized to [0, 1]
    """
    try:
        gradcam = GradCAM(model)
        # For single-output model, pass class_idx=0 (only one output neuron)
        cam = gradcam(input_tensor.to(device), class_idx=0)
        return cam
    except Exception as e:
        st.warning(f"⚠️ Grad-CAM failed: {e}")
        return None


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def sample_video_frames(video_path, sample_every=15, max_frames=40):
    """
    Sample frames from video at regular intervals.
    
    Args:
        video_path: Path to video file
        sample_every: Sample every N frames
        max_frames: Maximum number of frames to sample
    
    Returns:
        list: List of (frame_idx, frame_rgb) tuples
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error(f"❌ Failed to open video: {video_path}")
        return []
    
    frames = []
    frame_idx = 0
    sampled_count = 0
    
    while cap.isOpened() and sampled_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Sample at regular intervals
        if frame_idx % sample_every == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_idx, frame_rgb))
            sampled_count += 1
        
        frame_idx += 1
    
    cap.release()
    return frames


def compute_video_verdict(p_fakes, method="Mean FAKE prob"):
    """
    Compute overall video verdict from frame predictions.
    
    Args:
        p_fakes: List of P(fake) values for sampled frames
        method: Verdict method ("Mean FAKE prob", "Max FAKE prob", "Majority vote")
    
    Returns:
        tuple: (verdict, score)
            - verdict: "FAKE" or "REAL"
            - score: numeric score [0, 1]
    """
    if method == "Mean FAKE prob":
        score = np.mean(p_fakes)
    elif method == "Max FAKE prob":
        score = np.max(p_fakes)
    elif method == "Majority vote":
        # Fraction of frames predicted as FAKE
        score = np.mean([p >= 0.5 for p in p_fakes])
    else:
        score = np.mean(p_fakes)
    
    verdict = "FAKE" if score >= 0.5 else "REAL"
    return verdict, score


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_suspicion_timeline(frame_indices, p_fakes):
    """
    Plot frame-by-frame suspicion timeline.
    
    Args:
        frame_indices: List of frame indices
        p_fakes: List of P(fake) values
    
    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot line
    ax.plot(frame_indices, p_fakes, marker='o', linewidth=2, markersize=6, color='#FF4B4B')
    
    # Add threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Threshold')
    
    # Styling
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('P(FAKE)', fontsize=12)
    ax.set_title('Suspicion Timeline', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def soft_explanation(label, confidence):
    """
    Generate plain-English explanation of prediction.
    
    Args:
        label: Predicted class ("REAL" or "FAKE")
        confidence: Confidence score [0, 1]
    
    Returns:
        str: Explanation text
    """
    if label == "FAKE":
        return f"""
        **Model Assessment:** The model predicts this content is **{label}** with **{confidence:.1%} confidence**.
        
        The highlighted regions in the heatmap show areas that most influenced this decision. 
        Red/yellow areas indicate regions with suspicious patterns commonly associated with synthetic or manipulated content.
        """
    else:
        return f"""
        **Model Assessment:** The model predicts this content is **{label}** with **{confidence:.1%} confidence**.
        
        The heatmap highlights regions the model examined for authenticity. 
        These areas show characteristics consistent with genuine, unmanipulated content.
        """


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Trust-Aware Deepfake Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    # Title
    st.title("🔍 Trust-Aware Deepfake Detector")
    st.markdown("*Explainable AI for media authenticity verification*")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True, 
                                       help="Display visual explanations highlighting influential regions")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Video Processing")
    
    sample_every = st.sidebar.slider("Sample every N frames", min_value=1, max_value=60, value=15,
                                     help="Extract one frame every N frames")
    
    max_frames = st.sidebar.slider("Max sampled frames", min_value=5, max_value=120, value=40,
                                   help="Maximum number of frames to analyze")
    
    verdict_method = st.sidebar.selectbox(
        "Overall verdict method",
        ["Mean FAKE prob", "Max FAKE prob", "Majority vote"],
        help="How to combine frame-level predictions"
    )
    
    # Load model
    try:
        model, device = load_model(CHECKPOINT_PATH)
    except Exception:
        st.stop()
    
    # Main tabs
    tab_image, tab_video = st.tabs(["📷 Image Analysis", "🎬 Video Analysis"])
    
    # ========================================================================
    # IMAGE TAB
    # ========================================================================
    
    with tab_image:
        st.header("Image Analysis")
        
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, PNG, BMP, WebP"
        )
        
        if uploaded_image is not None:
            # Load image
            pil_image = Image.open(uploaded_image).convert('RGB')
            
            # Display original
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(pil_image, use_container_width=True)
            
            # Preprocess
            input_tensor = preprocess_pil(pil_image)
            
            # Predict
            with st.spinner("🔍 Analyzing..."):
                probs, label, confidence = predict_proba(model, input_tensor, device)
            
            # Display prediction
            with col2:
                st.subheader("Prediction")
                
                # Color-coded result
                color = "red" if label == "FAKE" else "green"
                st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
                
                # Confidence
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Probability breakdown
                st.markdown("**Probability Distribution**")
                prob_data = {
                    "Class": LABELS,
                    "Probability": [f"{p:.4f}" for p in probs]
                }
            # Grad-CAM
            if show_gradcam:
                st.subheader("🔥 Grad-CAM Visualization")
                
                with st.spinner("Generating heatmap..."):
                    cam = generate_gradcam(model, input_tensor, device)
                
                if cam is not None:
                    # Resize original image to match model input
                    resized_image = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                    
                    # Overlay CAM
                    overlay = overlay_cam_on_image(resized_image, cam, alpha=GRADCAM_ALPHA)
                    
                    st.image(overlay, caption="Regions influencing the prediction", use_container_width=True)
                    
                    # Facial Region Attribution Analysis
                    st.subheader("📊 Facial Feature Analysis")
                    
                    with st.spinner("Analyzing facial regions..."):
                        region_results = analyze_facial_regions(model, pil_image, device, probs[1])
                    
                    # Show strongest indicators
                    if region_results:
                        strongest = region_results[0]
                        strongest_name = strongest['region'].replace('_', ' ').title()
                        
                        # Highlight box
                        if strongest['delta'] > 0:
                            st.success(f"**🎯 Strongest Indicator: {strongest_name}**")
                            st.write(f"This region contributes **+{strongest['delta']:.4f}** to the FAKE prediction.")
                            st.write(f"When occluded, P(fake) drops from **{probs[1]:.4f}** → **{strongest['occluded_prob']:.4f}**")
                        else:
                            st.info(f"**🎯 Strongest Indicator: {strongest_name}**")
                            st.write(f"This region supports the REAL prediction (Δ = {strongest['delta']:.4f}).")
                        
                        # Top 3 regions
                        st.write("\n**Top 3 Contributing Regions:**")
                        
                        cols = st.columns(3)
                        for i, result in enumerate(region_results[:3]):
                            with cols[i]:
                                region_name = result['region'].replace('_', ' ').title()
                                delta = result['delta']
                                
                                # Color based on contribution
                                if abs(delta) > 0.05:
                                    emoji = "🔴" if delta > 0 else "🔵"
                                else:
                                    emoji = "⚪"
                                
                                st.metric(
                                    label=f"{emoji} {region_name}",
                                    value=f"{delta:+.4f}",
                                    delta="High Impact" if abs(delta) > 0.05 else "Low Impact"
                                )
                        
                        # Explanation
                        st.markdown("---")
                        st.markdown("**How to read this:**")
                        st.markdown("""
                        - **Δ (Delta)**: Change in P(fake) when region is occluded
                        - **Positive Δ**: Region contributes evidence for FAKE
                        - **Negative Δ**: Region contributes evidence for REAL
                        - **🔴 Red**: Strong fake indicator (Δ > +0.05)
                        - **🔵 Blue**: Strong real indicator (Δ < -0.05)
                        - **⚪ White**: Minimal impact (|Δ| < 0.05)
                        """)
            
            # Explanation
            st.markdown("---")
            st.markdown(soft_explanation(label, confidence))
    
    # ========================================================================
    # VIDEO TAB
    # ========================================================================
    
    with tab_video:
        st.header("Video Analysis")
        
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv"],
            help="Supported formats: MP4, MOV, AVI, MKV"
        )
        
        if uploaded_video is not None:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            try:
                # Display video
                st.video(uploaded_video)
                
                # Sample frames
                with st.spinner(f"📽️ Sampling frames (every {sample_every} frames, max {max_frames})..."):
                    sampled_frames = sample_video_frames(video_path, sample_every, max_frames)
                
                if len(sampled_frames) == 0:
                    st.error("❌ No frames extracted from video. Please check the file.")
                    st.stop()
                
                st.info(f"✅ Sampled {len(sampled_frames)} frames")
                
                # Analyze frames
                frame_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (frame_idx, frame_rgb) in enumerate(sampled_frames):
                    status_text.text(f"Analyzing frame {i+1}/{len(sampled_frames)}...")
                    
                    # Convert to PIL
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Preprocess
                    input_tensor = preprocess_pil(frame_pil)
                    
                    # Predict
                    probs, frame_label, frame_conf = predict_proba(model, input_tensor, device)
                    p_fake = probs[1]  # P(FAKE)
                    
                    # Generate Grad-CAM if enabled
                    overlay_img = None
                    # Generate Grad-CAM if enabled
                    overlay_img = None
                    if show_gradcam:
                        cam = generate_gradcam(model, input_tensor, device)
                        if cam is not None:
                            resized_frame = frame_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                            overlay_img = overlay_cam_on_image(resized_frame, cam, alpha=GRADCAM_ALPHA)
                    frame_results.append({
                        'frame_idx': frame_idx,
                        'p_fake': p_fake,
                        'label': frame_label,
                        'confidence': frame_conf,
                        'overlay': overlay_img,
                        'original': frame_pil
                    })
                    
                    progress_bar.progress((i + 1) / len(sampled_frames))
                
                status_text.text("✅ Analysis complete!")
                progress_bar.empty()
                
                # Compute overall verdict
                p_fakes = [r['p_fake'] for r in frame_results]
                verdict, score = compute_video_verdict(p_fakes, verdict_method)
                
                # Display verdict
                st.markdown("---")
                st.header("📊 Overall Video Verdict")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    verdict_color = "red" if verdict == "FAKE" else "green"
                    st.markdown(f"<h1 style='color: {verdict_color}; text-align: center;'>{verdict}</h1>", 
                               unsafe_allow_html=True)
                
                with col2:
                    st.metric("Score", f"{score:.2%}", help=f"Using {verdict_method}")
                
                with col3:
                    st.metric("Frames Analyzed", len(sampled_frames))
                
                # Suspicion timeline
                st.subheader("📈 Suspicion Timeline")
                frame_indices = [r['frame_idx'] for r in frame_results]
                fig = plot_suspicion_timeline(frame_indices, p_fakes)
                st.pyplot(fig)
                
                # Top suspicious frames
                st.subheader("🔍 Most Suspicious Frames")
                
                # Sort by P(FAKE) descending
                sorted_results = sorted(frame_results, key=lambda x: x['p_fake'], reverse=True)
                top_frames = sorted_results[:8]
                
                # Display in grid (4 columns)
                cols = st.columns(4)
                for i, result in enumerate(top_frames):
                    with cols[i % 4]:
                        display_img = result['overlay'] if show_gradcam and result['overlay'] else result['original']
                        st.image(display_img, 
                                caption=f"Frame {result['frame_idx']} | P(fake)={result['p_fake']:.3f}",
                                use_container_width=True)
                
                # Statistics
                st.markdown("---")
                st.subheader("📊 Frame-Level Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Mean P(FAKE)", f"{np.mean(p_fakes):.3f}")
                
                with col2:
                    st.metric("Max P(FAKE)", f"{np.max(p_fakes):.3f}")
                
                with col3:
                    fake_count = sum(1 for p in p_fakes if p >= 0.5)
                    st.metric("Frames Predicted FAKE", f"{fake_count}/{len(p_fakes)}")
                
            finally:
                # Cleanup temp file
                try:
                    os.unlink(video_path)
                except:
                    pass


if __name__ == "__main__":
    main()
