"""
FastAPI Backend for Deepfake Detection Terminal UI
===================================================
Provides REST API endpoints for the Next.js frontend.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from PIL import Image
import io
import time
import base64
import cv2
import tempfile
import os

# Import model utilities
from model_def import build_model, load_checkpoint
from gradcam_utils import GradCAM, overlay_cam_on_image

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = 299
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
LABELS = ["REAL", "FAKE"]
CHECKPOINT_PATH = "weights/xception_gan_augmented.pth"

FACE_REGIONS = {
    'forehead': (0.2, 0.0, 0.8, 0.25),
    'left_eye': (0.2, 0.25, 0.45, 0.4),
    'right_eye': (0.55, 0.25, 0.8, 0.4),
    'nose': (0.4, 0.35, 0.6, 0.55),
    'left_cheek': (0.1, 0.4, 0.35, 0.65),
    'right_cheek': (0.65, 0.4, 0.9, 0.65),
    'mouth': (0.3, 0.6, 0.7, 0.75),
    'jawline': (0.15, 0.7, 0.85, 0.95),
}

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Deepfake Detector API",
    description="Neural network-powered deepfake detection with visual attribution",
    version="1.0.0"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL MODEL
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def get_model():
    global model
    if model is None:
        print("🔧 Loading model...")
        model = build_model(num_classes=1)
        model = load_checkpoint(model, CHECKPOINT_PATH, device=device)
        model.eval()
        model.to(device)
        print(f"✅ Model loaded on {device}")
    return model

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_pil(pil_image: Image.Image) -> torch.Tensor:
    """Preprocess PIL image for model inference."""
    img = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Normalize
    img_array = (img_array - MEAN) / STD
    
    # HWC -> CHW -> NCHW
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

def predict_proba(model, input_tensor: torch.Tensor, device):
    """Run inference and return probabilities."""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        
        # Single output with sigmoid
        p_fake = torch.sigmoid(output).item()
        p_real = 1.0 - p_fake
        
        probs = [p_real, p_fake]
        pred_idx = np.argmax(probs)
        label = LABELS[pred_idx]
        confidence = probs[pred_idx]
        
    return probs, label, confidence

def generate_gradcam(model, input_tensor: torch.Tensor, device) -> Optional[np.ndarray]:
    """Generate Grad-CAM heatmap."""
    try:
        gradcam = GradCAM(model)
        cam = gradcam(input_tensor.to(device), class_idx=0)
        return cam
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        return None

def analyze_facial_regions(model, pil_image: Image.Image, device, original_prob: float) -> List[dict]:
    """Analyze facial regions via occlusion."""
    from PIL import ImageDraw
    
    results = []
    img_resized = pil_image.resize((299, 299), Image.BILINEAR)
    
    for region_name, (x1, y1, x2, y2) in FACE_REGIONS.items():
        img_copy = img_resized.copy()
        draw = ImageDraw.Draw(img_copy)
        
        x1_abs = int(x1 * 299)
        y1_abs = int(y1 * 299)
        x2_abs = int(x2 * 299)
        y2_abs = int(y2 * 299)
        
        draw.rectangle([x1_abs, y1_abs, x2_abs, y2_abs], fill=(128, 128, 128))
        
        occluded_tensor = preprocess_pil(img_copy)
        probs, _, _ = predict_proba(model, occluded_tensor, device)
        occluded_prob = probs[1]
        
        delta = original_prob - occluded_prob
        
        results.append({
            'name': region_name,
            'delta': float(delta),
            'occludedProb': float(occluded_prob)
        })
    
    results.sort(key=lambda x: abs(x['delta']), reverse=True)
    return results

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict
    processingTime: float
    facialRegions: Optional[List[dict]] = None
    gradcamImage: Optional[str] = None

class VideoFrameResponse(BaseModel):
    frameIndex: int
    pFake: float
    label: str
    confidence: float

class VideoAnalysisResponse(BaseModel):
    verdict: str
    score: float
    method: str
    framesAnalyzed: int
    frames: List[VideoFrameResponse]
    processingTime: float

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Deepfake Detector API",
        "version": "1.0.0",
        "device": str(device)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/analyze/image", response_model=PredictionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    include_gradcam: bool = True,
    include_regions: bool = True
):
    """Analyze an uploaded image for deepfake detection."""
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load model
        model = get_model()
        
        # Read and preprocess image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_pil(pil_image)
        
        # Predict
        probs, label, confidence = predict_proba(model, input_tensor, device)
        
        # Facial region analysis
        facial_regions = None
        if include_regions:
            facial_regions = analyze_facial_regions(model, pil_image, device, probs[1])
        
        # Grad-CAM
        gradcam_image = None
        if include_gradcam:
            cam = generate_gradcam(model, input_tensor, device)
            if cam is not None:
                resized_image = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                overlay = overlay_cam_on_image(resized_image, cam, alpha=0.5)
                gradcam_image = image_to_base64(overlay)
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            label=label,
            confidence=float(confidence),
            probabilities={"real": float(probs[0]), "fake": float(probs[1])},
            processingTime=processing_time,
            facialRegions=facial_regions,
            gradcamImage=gradcam_image
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/video", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    sample_every: int = 15,
    max_frames: int = 40,
    method: str = "mean"
):
    """Analyze an uploaded video for deepfake detection."""
    start_time = time.time()
    
    # Validate file type
    valid_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]
    if file.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Load model
        model = get_model()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Sample frames
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        frame_idx = 0
        sampled_count = 0
        
        while cap.isOpened() and sampled_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_every == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                input_tensor = preprocess_pil(pil_frame)
                
                probs, label, conf = predict_proba(model, input_tensor, device)
                
                frames.append(VideoFrameResponse(
                    frameIndex=frame_idx,
                    pFake=float(probs[1]),
                    label=label,
                    confidence=float(conf)
                ))
                sampled_count += 1
            
            frame_idx += 1
        
        cap.release()
        os.unlink(tmp_path)
        
        if len(frames) == 0:
            raise HTTPException(status_code=400, detail="No frames extracted from video")
        
        # Compute verdict
        p_fakes = [f.pFake for f in frames]
        
        if method == "mean":
            score = float(np.mean(p_fakes))
        elif method == "max":
            score = float(np.max(p_fakes))
        elif method == "majority":
            score = float(np.mean([p >= 0.5 for p in p_fakes]))
        else:
            score = float(np.mean(p_fakes))
        
        verdict = "FAKE" if score >= 0.5 else "REAL"
        processing_time = time.time() - start_time
        
        return VideoAnalysisResponse(
            verdict=verdict,
            score=score,
            method=method,
            framesAnalyzed=len(frames),
            frames=frames,
            processingTime=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
