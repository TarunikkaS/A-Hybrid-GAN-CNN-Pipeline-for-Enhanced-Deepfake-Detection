# Trust-Aware Deepfake Detector - Streamlit App

A comprehensive web application for deepfake detection with visual explainability using Grad-CAM.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Model Checkpoint

Ensure your model checkpoint is located at:
```
weights/xception_gan_augmented.pth
```

Or update the `CHECKPOINT_PATH` in `app.py`.

### 3. Update Model Architecture

**IMPORTANT:** Edit `model_def.py` to match your training code's model architecture.

The provided implementation uses `timm.create_model('legacy_xception')` as a default. 
If you used a different architecture during training, update the `build_model()` function accordingly.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📋 Features

### Image Analysis
- Upload images (JPG, PNG, BMP, WebP)
- Binary classification: REAL vs FAKE
- Confidence scores and probability distribution
- Grad-CAM heatmap visualization
- Plain-English explanations

### Video Analysis
- Upload videos (MP4, MOV, AVI, MKV)
- Configurable frame sampling (every N frames, max frames)
- Frame-by-frame predictions
- Multiple verdict methods:
  - **Mean FAKE prob**: Average P(fake) across all frames
  - **Max FAKE prob**: Maximum P(fake) encountered
  - **Majority vote**: Fraction of frames predicted as FAKE
- Suspicion timeline plot
- Top 8 most suspicious frames with Grad-CAM overlays

### Grad-CAM Explainability
- Automatic target layer detection (last Conv2d)
- Visual heatmap showing influential regions
- Configurable alpha blending
- Works for both image and video analysis

## ⚙️ Configuration

### Model Settings (in `app.py`)

```python
# Image preprocessing
IMG_SIZE = 299                    # Input size (299x299 for Xception)
MEAN = (0.5, 0.5, 0.5)           # Normalization mean
STD = (0.5, 0.5, 0.5)            # Normalization std

# Class labels
LABELS = ["REAL", "FAKE"]         # Adjust if needed

# Checkpoint path
CHECKPOINT_PATH = "weights/xception_gan_augmented.pth"

# Grad-CAM blending
GRADCAM_ALPHA = 0.5               # 0 = original, 1 = full heatmap
```

### Sidebar Controls

**All Users:**
- ☑️ Show Grad-CAM (toggle visual explanations)

**Video Analysis:**
- 🎚️ Sample every N frames (1-60, default 15)
- 🎚️ Max sampled frames (5-120, default 40)
- 📊 Overall verdict method dropdown

## 🗂️ File Structure

```
.
├── app.py                          # Main Streamlit application
├── model_def.py                    # Model architecture definition
├── gradcam_utils.py                # Grad-CAM implementation
├── requirements.txt                # Python dependencies
├── README_streamlit.md             # This file
└── weights/
    └── xception_gan_augmented.pth  # Model checkpoint
```

## 🔧 Troubleshooting

### "Failed to load model"
1. Check that `weights/xception_gan_augmented.pth` exists
2. Update `model_def.py` to match your training architecture
3. Verify checkpoint format matches expected structure

### "No Conv2d layer found in model"
Your model doesn't have convolutional layers. Grad-CAM requires CNNs.
Set `show_gradcam = False` in sidebar or update model architecture.

### Video processing errors
1. Ensure video file is not corrupted
2. Try reducing `max_frames` if memory issues occur
3. Check that OpenCV can read your video format

### GPU not detected
Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Model Checkpoint Formats Supported

The app handles multiple checkpoint formats:

1. **Training checkpoint with 'state_dict' key:**
   ```python
   {
       'state_dict': {...},
       'optimizer': {...},
       'epoch': 10
   }
   ```

2. **Direct state dict:**
   ```python
   {
       'features.0.weight': tensor(...),
       'classifier.weight': tensor(...),
       ...
   }
   ```

3. **Full model object:**
   ```python
   torch.save(model, 'checkpoint.pth')
   ```

4. **Checkpoint with 'model' key:**
   ```python
   {
       'model': state_dict or full_model
   }
   ```

## 🎯 Usage Examples

### Analyzing a Single Image

1. Click **"📷 Image Analysis"** tab
2. Upload an image
3. View prediction and Grad-CAM heatmap
4. Read plain-English explanation

### Analyzing a Video

1. Click **"🎬 Video Analysis"** tab
2. Configure sampling settings in sidebar:
   - Sample every 15 frames (for 30 fps = 2 frames/sec)
   - Max 40 frames
   - Verdict method: "Mean FAKE prob"
3. Upload video
4. Wait for frame-by-frame analysis
5. Review:
   - Overall verdict
   - Suspicion timeline plot
   - Top suspicious frames

### Comparing Verdict Methods

Try different verdict methods for the same video to understand:
- **Mean FAKE prob**: Best for overall assessment
- **Max FAKE prob**: Conservative (flags if ANY frame looks fake)
- **Majority vote**: Binary approach (most frames FAKE = video FAKE)

## 📝 Adapting to Your Model

### If you trained with different preprocessing:

Update in `app.py`:
```python
MEAN = (0.485, 0.456, 0.406)  # ImageNet mean
STD = (0.229, 0.224, 0.225)   # ImageNet std
```

### If your model outputs single logit (binary classification):

Modify `predict_proba()` in `app.py`:
```python
def predict_proba(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logit = model(input_tensor).squeeze()  # Single output
        p_fake = torch.sigmoid(logit).item()
        probs = np.array([1 - p_fake, p_fake])  # [P(real), P(fake)]
        pred_idx = np.argmax(probs)
        label = LABELS[pred_idx]
        confidence = probs[pred_idx]
    return probs, label, confidence
```

### If using multi-class (>2 classes):

Update `LABELS` and modify model building in `model_def.py`:
```python
LABELS = ["REAL", "DEEPFAKE_TYPE1", "DEEPFAKE_TYPE2", ...]
model = build_model(num_classes=len(LABELS))
```

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io)
3. Set entrypoint: `app.py`
4. Upload model checkpoint via secrets or external storage

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📚 References

- **Grad-CAM Paper:** Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
- **Streamlit Documentation:** https://docs.streamlit.io
- **PyTorch Documentation:** https://pytorch.org/docs

## 📄 License

This application is provided as-is for research and educational purposes.
