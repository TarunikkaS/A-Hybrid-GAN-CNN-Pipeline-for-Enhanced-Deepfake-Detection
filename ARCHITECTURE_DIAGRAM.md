# 🔍 Trust-Aware Deepfake Detector - Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APPLICATION (app.py)                       │
│                          http://localhost:8501                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                               │
        ▼                                                               ▼
┌───────────────────┐                                       ┌───────────────────┐
│   📷 IMAGE TAB    │                                       │   🎬 VIDEO TAB    │
│   ─────────────   │                                       │   ────────────    │
│ • Upload image    │                                       │ • Upload video    │
│ • View original   │                                       │ • Frame sampling  │
│ • Get prediction  │                                       │ • Batch analysis  │
│ • See Grad-CAM    │                                       │ • Timeline plot   │
│ • Read explanation│                                       │ • Top frames grid │
└─────────┬─────────┘                                       └─────────┬─────────┘
          │                                                           │
          │                                                           │
          └─────────────────────┬───────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   ⚙️  SIDEBAR         │
                    │   ───────────────     │
                    │ ☑️ Show Grad-CAM      │
                    │ 🎚️ Sample every N     │
                    │ 🎚️ Max frames         │
                    │ 📊 Verdict method     │
                    └───────────┬───────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────────┐
        │           CORE PROCESSING PIPELINE                │
        └───────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ model_def.py │ │gradcam_utils │ │   app.py     │
        │              │ │     .py      │ │   helpers    │
        └──────────────┘ └──────────────┘ └──────────────┘
                │               │               │
                ├───────────────┼───────────────┤
                │                               │
                ▼                               ▼
        ┌─────────────┐               ┌─────────────────┐
        │ build_model │               │ preprocess_pil  │
        │ load_ckpt   │               │ predict_proba   │
        └─────────────┘               │ generate_gradcam│
                                      │ sample_frames   │
                                      └─────────────────┘
                                              │
                                              ▼
                                ┌─────────────────────────┐
                                │  🧠 MODEL INFERENCE     │
                                │  ─────────────────      │
                                │  Input: (1, 3, 299, 299)│
                                │  Output: (1, 2) logits  │
                                │  → softmax → probs      │
                                └─────────────────────────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        │                     │                     │
                        ▼                     ▼                     ▼
            ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
            │   PREDICTION     │  │    GRAD-CAM      │  │   EXPLANATION    │
            │   ──────────     │  │    ────────      │  │   ───────────    │
            │ • Class: FAKE    │  │ • Activation map │  │ • Plain English  │
            │ • Confidence: 95%│  │ • Heatmap overlay│  │ • Context-aware  │
            │ • Probs table    │  │ • JET colormap   │  │ • Concise text   │
            └──────────────────┘  └──────────────────┘  └──────────────────┘
```

## 📊 Data Flow Diagrams

### Image Analysis Flow

```
User Upload (JPG/PNG)
        │
        ▼
preprocess_pil()
   │ ├─ RGB conversion
   │ ├─ Resize 299×299
   │ ├─ Normalize [0,1]
   │ ├─ Apply mean/std
   │ └─ Add batch dim
   ▼
input_tensor (1,3,299,299)
        │
        ▼
model.forward()
        │
        ▼
logits (1, 2)
        │
        ├─ softmax ──────────┐
        │                    │
        ▼                    ▼
probs [P(real), P(fake)]  GradCAM
        │                    │
        │                    ├─ backward(one_hot)
        │                    ├─ grad weights
        │                    ├─ weighted activations
        │                    ├─ ReLU + normalize
        │                    └─ overlay on image
        │                    │
        ▼                    ▼
Display Prediction    Display Heatmap
```

### Video Analysis Flow

```
User Upload (MP4/MOV)
        │
        ▼
Save to temp file
        │
        ▼
cv2.VideoCapture()
        │
        ├─ Loop frames
        │  │
        │  ├─ Sample every N frames
        │  │  │
        │  │  ▼
        │  │ frame_rgb
        │  │  │
        │  │  ▼
        │  │ preprocess_pil()
        │  │  │
        │  │  ▼
        │  │ predict_proba()
        │  │  │
        │  │  ├─ Store p_fake
        │  │  ├─ Store label
        │  │  └─ Store overlay (if Grad-CAM)
        │  │
        │  └─ Repeat for max_frames
        │
        ▼
frame_results = [
   {frame_idx, p_fake, label, confidence, overlay},
   ...
]
        │
        ├────────────────┬────────────────┬─────────────────┐
        │                │                │                 │
        ▼                ▼                ▼                 ▼
compute_verdict   plot_timeline   sort by p_fake   statistics
   │                │                │                 │
   │ Mean/Max/Vote  │ matplotlib     │ Top 8 frames    │ mean/max/count
   │                │                │                 │
   ▼                ▼                ▼                 ▼
Overall FAKE/REAL  Suspicion plot  Grid display     Metrics panel
```

## 🏗️ File Dependencies

```
app.py
  ├── imports: streamlit, torch, numpy, cv2, PIL, matplotlib
  ├── from model_def import: build_model, load_checkpoint
  └── from gradcam_utils import: GradCAM, overlay_cam_on_image

model_def.py
  ├── imports: torch, torch.nn
  └── from timm import: create_model

gradcam_utils.py
  ├── imports: torch, torch.nn, torch.nn.functional
  ├── imports: numpy, cv2, PIL
  └── no external gradcam library needed

requirements.txt
  └── specifies all package versions
```

## 🔄 Model Checkpoint Loading Logic

```
load_checkpoint()
        │
        ▼
torch.load(path)
        │
        ├─ Case 1: Has 'state_dict' key?
        │  └─ YES → model.load_state_dict(ckpt['state_dict'])
        │
        ├─ Case 2: Has 'model' key?
        │  ├─ Is dict? → model.load_state_dict(ckpt['model'])
        │  └─ Is model object? → return ckpt['model']
        │
        ├─ Case 3: Looks like state_dict?
        │  └─ YES → model.load_state_dict(ckpt)
        │
        └─ Case 4: Full model object
           └─ return ckpt

All paths → model.to(device).eval()
```

## 🎨 Grad-CAM Pipeline

```
GradCAM(model)
     │
     ├─ Auto-detect last Conv2d layer
     │  └─ Register forward/backward hooks
     │
     ▼
generate_cam(input, class_idx)
     │
     ├─ Forward pass
     │  └─ Store activations (hook)
     │
     ├─ Backward pass (one-hot target)
     │  └─ Store gradients (hook)
     │
     ├─ Compute weights
     │  └─ weights = gradients.mean(dim=(2,3))
     │
     ├─ Weighted combination
     │  └─ cam = (weights * activations).sum(dim=1)
     │
     ├─ Apply ReLU
     │  └─ cam = F.relu(cam)
     │
     └─ Normalize [0,1]
        └─ cam = cam / cam.max()

overlay_cam_on_image(image, cam)
     │
     ├─ Resize CAM to image size
     ├─ Apply cv2.COLORMAP_JET
     ├─ Alpha blend with original
     └─ Return PIL Image
```

## 📱 UI Component Hierarchy

```
Streamlit Page
├── Title: "Trust-Aware Deepfake Detector"
├── Subtitle: "Explainable AI for media authenticity..."
│
├── Sidebar
│   ├── Header: "Settings"
│   ├── Checkbox: Show Grad-CAM
│   ├── Divider
│   ├── Header: "Video Processing"
│   ├── Slider: Sample every N frames
│   ├── Slider: Max sampled frames
│   └── Selectbox: Overall verdict method
│
└── Main Area
    └── Tabs
        ├── Tab 1: Image Analysis
        │   ├── File uploader (image)
        │   ├── Columns [Original | Prediction]
        │   │   ├── Col 1: Display image
        │   │   └── Col 2: Result metrics
        │   ├── Grad-CAM visualization
        │   └── Explanation text
        │
        └── Tab 2: Video Analysis
            ├── File uploader (video)
            ├── Video player
            ├── Progress indicators
            ├── Verdict section
            │   └── Columns [Verdict | Score | Frame count]
            ├── Suspicion timeline (matplotlib)
            ├── Top frames grid (4 columns × 2 rows)
            └── Statistics section
                └── Columns [Mean | Max | Count]
```

## 🔐 Error Handling Strategy

```
Each Operation
     │
     ├─ Try block
     │  └─ Core logic
     │
     └─ Except block
        ├─ Log error to console
        ├─ Show st.error() with message
        ├─ Provide troubleshooting hints
        └─ Graceful degradation or st.stop()

Examples:
• Model loading fail → Show installation tips → st.stop()
• Grad-CAM fail → Show warning → Continue without CAM
• Video frame extraction fail → Show error → Skip video
• No frames extracted → st.error → Don't proceed
```

## 🚀 Deployment Options

```
Development
    └─ streamlit run app.py (localhost:8501)

Production Options:
    ├─ Streamlit Cloud (share.streamlit.io)
    │   └─ Push to GitHub → Connect repo → Deploy
    │
    ├─ Docker Container
    │   └─ Build image → Run container → Expose port 8501
    │
    ├─ Cloud VM (AWS/GCP/Azure)
    │   └─ SSH → Install deps → Run app → Public IP
    │
    └─ Kubernetes
        └─ Deploy pod → Service → Ingress
```

## 📊 Performance Optimization Points

```
1. Model Caching
   └─ @st.cache_resource decorator
      └─ Loads model once, reuses across sessions

2. Frame Sampling
   └─ User controls max_frames
      └─ Prevents analyzing entire video

3. GPU Acceleration
   └─ Automatic CUDA detection
      └─ 10x faster inference

4. Lazy Loading
   └─ Grad-CAM only computed if enabled
      └─ Saves ~100-200ms per image

5. Temporary File Cleanup
   └─ Delete video after processing
      └─ Prevents disk fill
```

## 🎯 Customization Entry Points

```
Easy (UI Config):
├─ Sidebar toggles
├─ Slider values
└─ Dropdown selections

Medium (Constants):
├─ IMG_SIZE (line 24)
├─ MEAN/STD (lines 25-26)
├─ LABELS (line 29)
├─ CHECKPOINT_PATH (line 32)
└─ GRADCAM_ALPHA (line 35)

Advanced (Functions):
├─ preprocess_pil() → Custom preprocessing
├─ build_model() → Different architecture
├─ predict_proba() → Multi-class output
├─ compute_video_verdict() → New aggregation
└─ soft_explanation() → Enhanced explanations
```

---

**This architecture provides:**
- ✅ Modular, maintainable code
- ✅ Clear separation of concerns
- ✅ Extensible design
- ✅ Robust error handling
- ✅ Performance optimization
- ✅ User-friendly interface

**Ready for production deployment! 🚀**
