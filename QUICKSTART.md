# 🚀 Quick Start Guide - Trust-Aware Deepfake Detector

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- ✅ Streamlit (web UI framework)
- ✅ PyTorch + torchvision (deep learning)
- ✅ timm (model architectures)
- ✅ OpenCV, Pillow (image/video processing)
- ✅ NumPy, Matplotlib (computation + visualization)

### Step 2: Verify Installation

```bash
python model_def.py
```

You should see:
```
✅ Loaded legacy_xception with 2 classes
Model has 20811050 parameters
```

### Step 3: Confirm Model Checkpoint

```bash
ls -lh weights/xception_gan_augmented.pth
```

Should show your model file (typically 80-100 MB).

## 🎯 Launch the App

### Option A: Using the launcher script (recommended)

```bash
./launch_app.sh
```

### Option B: Direct Streamlit command

```bash
streamlit run app.py
```

The app will automatically open in your browser at **http://localhost:8501**

## 📸 Using the App

### Image Analysis

1. Click the **"📷 Image Analysis"** tab
2. Click **"Browse files"** or drag-and-drop an image
3. Supported formats: JPG, PNG, BMP, WebP
4. View results:
   - **Left side**: Original image
   - **Right side**: Prediction (REAL/FAKE), confidence, probabilities
   - **Bottom**: Grad-CAM heatmap showing influential regions
   - **Explanation**: Plain-English interpretation

### Video Analysis

1. Click the **"🎬 Video Analysis"** tab
2. Configure settings in **left sidebar**:
   - ☑️ **Show Grad-CAM**: Enable visual explanations
   - 🎚️ **Sample every N frames**: Extract 1 frame every N frames (default: 15)
     - For 30 fps video: 15 = 2 frames/second
   - 🎚️ **Max sampled frames**: Maximum frames to analyze (default: 40)
     - Limits analysis time for long videos
   - 📊 **Overall verdict method**: How to combine frame predictions
     - **Mean FAKE prob**: Average P(fake) across all frames (balanced)
     - **Max FAKE prob**: Highest P(fake) encountered (conservative)
     - **Majority vote**: Fraction of frames classified as FAKE (binary)
3. Upload video (MP4, MOV, AVI, MKV)
4. Wait for analysis (progress bar shows status)
5. Review results:
   - **Overall Verdict**: FAKE or REAL with confidence score
   - **Suspicion Timeline**: Frame-by-frame P(fake) plot
   - **Most Suspicious Frames**: Top 8 frames with highest fake probability
   - **Statistics**: Mean/max P(fake), frame counts

## ⚙️ Configuration

### Adjusting Preprocessing

Edit `app.py` lines 24-26 if your training used different preprocessing:

```python
IMG_SIZE = 299                    # Xception expects 299x299
MEAN = (0.5, 0.5, 0.5)           # Change to match training
STD = (0.5, 0.5, 0.5)            # Change to match training
```

**Common alternatives:**
- ImageNet normalization: `MEAN = (0.485, 0.456, 0.406)`, `STD = (0.229, 0.224, 0.225)`
- No normalization: `MEAN = (0, 0, 0)`, `STD = (1, 1, 1)`

### Changing Model Path

Edit `app.py` line 33:

```python
CHECKPOINT_PATH = "weights/xception_gan_augmented.pth"
```

### Adjusting Grad-CAM Intensity

Edit `app.py` line 36:

```python
GRADCAM_ALPHA = 0.5  # 0 = original image, 1 = full heatmap
```

## 🎨 Understanding Grad-CAM Heatmaps

**Color interpretation:**
- 🔴 **Red/Yellow**: High activation - these regions strongly influenced the prediction
- 🟢 **Green**: Moderate activation
- 🔵 **Blue/Purple**: Low activation

**For FAKE predictions:**
- Red areas show regions with suspicious patterns (e.g., blending artifacts, synthetic textures)

**For REAL predictions:**
- Red areas show regions with authentic characteristics (e.g., natural skin texture, consistent lighting)

## 🔧 Troubleshooting

### App won't start

**Error: "ModuleNotFoundError: No module named 'streamlit'"**
```bash
pip install streamlit
```

**Error: "No module named 'timm'"**
```bash
pip install timm
```

### Model loading fails

**Error: "Failed to load model"**

1. Check model file exists:
   ```bash
   ls -lh weights/xception_gan_augmented.pth
   ```

2. Verify model architecture in `model_def.py` matches your training code

3. Test model loading directly:
   ```bash
   python -c "from model_def import build_model, load_checkpoint; import torch; model = build_model(2); model = load_checkpoint(model, 'weights/xception_gan_augmented.pth', 'cpu'); print('✅ Model loaded')"
   ```

### Grad-CAM not showing

1. Check **"Show Grad-CAM"** is enabled in sidebar
2. Ensure model has convolutional layers (Grad-CAM requires CNNs)
3. Check browser console for errors (F12 → Console tab)

### Video processing slow

**Normal behavior:**
- Analyzing 40 frames takes ~30-60 seconds on CPU
- Each frame requires: preprocessing → forward pass → Grad-CAM (if enabled)

**Speed up:**
1. Reduce **"Max sampled frames"** (e.g., 20 instead of 40)
2. Increase **"Sample every N frames"** (e.g., 30 instead of 15)
3. Disable **"Show Grad-CAM"** for faster processing
4. Use GPU if available (model automatically uses CUDA)

### Out of memory

**For large videos:**
1. Reduce **"Max sampled frames"** to 20-30
2. Disable **"Show Grad-CAM"** (saves memory)
3. Close other applications

**For high-resolution images:**
- Images are automatically resized to 299×299, so this shouldn't occur

## 💡 Tips & Best Practices

### For Best Results

1. **Image quality matters**: Use clear, well-lit images
2. **Face-focused content**: Model trained on face crops works best on faces
3. **Video sampling**: Balance between coverage and speed
   - Short videos (<30s): Sample every 10 frames, max 50 frames
   - Long videos (>1 min): Sample every 30 frames, max 40 frames
4. **Verdict method selection**:
   - **Mean FAKE prob**: Best for overall assessment (recommended)
   - **Max FAKE prob**: Use when false negatives are costly
   - **Majority vote**: Use for binary decision

### Interpreting Results

**High confidence (>90%)**
- Model is very certain
- Multiple consistent cues across image

**Medium confidence (60-90%)**
- Model found some suspicious patterns
- Review Grad-CAM to understand which regions contributed

**Low confidence (50-60%)**
- Borderline case, model uncertain
- Consider getting human expert review
- May indicate edge case or novel manipulation

**For videos:**
- **Consistent timeline**: Steady high/low values indicate strong signal
- **Spiky timeline**: Inconsistent predictions may indicate:
  - Compression artifacts
  - Scene changes
  - Partial manipulation (only some frames fake)

## 📊 Example Workflows

### Workflow 1: Quick Single Image Check
1. Launch app
2. Upload image to Image Analysis tab
3. Check prediction + confidence
4. Review Grad-CAM heatmap
5. Read explanation
**Time: ~5 seconds**

### Workflow 2: Detailed Video Investigation
1. Launch app
2. Configure sidebar:
   - Show Grad-CAM: ✅
   - Sample every 15 frames
   - Max 40 frames
   - Verdict: Mean FAKE prob
3. Upload video
4. Wait for analysis (~45 seconds)
5. Review:
   - Overall verdict
   - Suspicion timeline (check for spikes)
   - Top suspicious frames
6. If uncertain, adjust verdict method and compare
**Time: ~2-3 minutes**

### Workflow 3: Batch Processing Multiple Images
1. Upload first image
2. Note prediction
3. Upload next image (replaces previous)
4. Compare results
**Time: ~5 seconds per image**

## 🚀 Advanced Usage

### Running on GPU

The app automatically uses GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

To verify GPU usage, check the terminal output when launching:
```
✅ Model loaded on cuda
```

### Custom Preprocessing

If your model was trained with custom preprocessing, update `preprocess_pil()` in `app.py`:

```python
def preprocess_pil(pil_image, img_size=IMG_SIZE, mean=MEAN, std=STD):
    img = pil_image.convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    
    # ADD YOUR CUSTOM PREPROCESSING HERE
    # Example: additional augmentation, color space conversion, etc.
    
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor
```

### Multi-Class Classification

If you trained with >2 classes, update `app.py`:

```python
LABELS = ["REAL", "DEEPFAKE_TYPE1", "DEEPFAKE_TYPE2", "GAN_GENERATED"]
```

And `model_def.py`:

```python
model = build_model(num_classes=4)  # Match number of classes
```

## 📞 Support

**Common questions:**

Q: Can I use a different model checkpoint?
A: Yes, change `CHECKPOINT_PATH` in `app.py` and ensure `model_def.py` matches the architecture.

Q: Does this work on CPU?
A: Yes, but GPU is ~10x faster. CPU analysis takes ~1-2 seconds per frame.

Q: Can I deploy this online?
A: Yes! See `README_streamlit.md` for deployment instructions (Streamlit Cloud, Docker, etc.)

Q: How accurate is the model?
A: Depends on your training. Check validation accuracy from training logs. This app displays the model's predictions as-is.

## 📚 Next Steps

- ✅ Test on known real/fake images to verify accuracy
- ✅ Compare different verdict methods on sample videos
- ✅ Adjust Grad-CAM alpha for clearer visualization
- ✅ Consider deployment for team access (see README_streamlit.md)

**Happy detecting! 🔍**
