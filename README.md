# Deepfake Detection System
## GAN-Augmented XceptionNet with Trust-Aware Inference

A state-of-the-art deepfake detection system leveraging GAN-refined XceptionNet architecture with advanced adversarial training and frequency analysis. This project achieves **95.10% accuracy** on the FaceForensics++ dataset with comprehensive facial attribution visualization.

---

 Project Overview

This system detects AI-generated deepfakes using a novel GAN-augmented training approach combined with XceptionNet architecture. The model is trained on multiple manipulation techniques including Deepfakes, Face2Face, and FaceSwap, providing robust detection across various forgery methods.

 Key Features

- **High-Performance Detection**: 95.10% accuracy with 98.5% ROC-AUC
- **Multiple Interfaces**: 
  - Streamlit Web App
  - Terminal UI (Next.js + FastAPI)
  - Command-line inference
- **Visual Explanations**: GradCAM heatmaps for facial manipulation attribution
- **Trust-Aware Inference**: Confidence scoring and uncertainty quantification
- **Real-time Processing**: Optimized for CPU and GPU inference

---

## 📊 Model Performance

### Comparative Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Baseline XceptionNet** | 90.91% | 91.5% | 91.3% | 91.0% | 91.0% |
| **GAN Refinement (Adv only)** | 92.80% | 94.1% | 91.6% | 92.8% | 94.6% |
| **GAN Refinement (Adv + L1)** | 94.60% | 95.6% | 92.4% | 94.0% | 96.8% |
| **GAN-Refined XceptionNet (Adv + L1 + Freq)** ⭐ | **95.10%** | **97.2%** | **93.3%** | **95.2%** | **98.5%** |

### Model Architecture

**Primary Model**: `xception_gan_augmented.pth`
- **Architecture**: Modified XceptionNet with GAN-based adversarial refinement
- **Training Strategy**: Adversarial loss + L1 reconstruction + Frequency domain analysis
- **Input**: 299×299 RGB face images
- **Output**: Binary classification (Real/Fake) + confidence score

**Baseline Model**: `best_xception.pth`
- Standard XceptionNet trained on FaceForensics++
- Used for comparison and fallback inference

---

## 🗂️ Dataset

**FaceForensics++ (FF++)** - c23 compression level

### Manipulation Types
- **Deepfakes**: Face swapping using autoencoders
- **Face2Face**: Facial reenactment
- **FaceSwap**: Traditional computer vision face swapping
- **Original**: Pristine YouTube videos

### Dataset Structure
```
Dataset/
├── manipulated_sequences/
│   ├── Deepfakes/c23/videos/
│   ├── Face2Face/c23/videos/
│   └── FaceSwap/c23/videos/
└── original_sequences/
    └── youtube/c23/videos/
```

---

 Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Node.js 18+ (optional, for Terminal UI)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "FINAL DEEPFAKE PROJECT MODEL"
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model weights**
Ensure the following weights are present:
- `xception_gan_augmented.pth` (primary model)
- `weights/xception_gan_augmented.pth` (for Terminal UI)
- `best_xception.pth` (baseline model)

---

## 💻 Usage

### Option 1: Streamlit Web App (Recommended)

```bash
chmod +x launch_app.sh
./launch_app.sh
```

Or manually:
```bash
streamlit run app.py
```

Access at: **http://localhost:8501**

**Features**:
- Upload images or videos
- Real-time detection
- Confidence scores
- GradCAM visualization
- Batch processing

### Option 2: Terminal UI (Advanced)

**Backend API**:
```bash
cd deepfake-terminal-ui/api
python server.py
```
API runs on: **http://localhost:8000**
API Docs: **http://localhost:8000/docs**

**Frontend** (requires Node.js):
```bash
cd deepfake-terminal-ui
npm install
npm run dev
```
Frontend: **http://localhost:3000**

### Option 3: Command-Line Inference

**Single Image**:
```bash
python xception_trust_aware_inference.py --image path/to/image.jpg
```

**With GradCAM**:
```bash
python test_image_gradcam.py --image path/to/image.jpg --output gradcam_output.jpg
```

**Batch Processing**:
```bash
python local_manipulation_attribution.py --input_dir path/to/images --output_dir results/
```

---

 GradCAM Visualization

Generate facial attribution heatmaps to understand what regions the model focuses on:

```bash
# Single image with comparison
python create_gradcam_comparison.py --image path/to/image.jpg

# Grid visualization
python show_gradcam_grid.py --input_dir path/to/images

# Batch processing
python test_ffpp2_gradcam.py --model FFPP2.pt --dataset_dir Dataset/
```

**Output**: Heatmap overlays highlighting manipulated facial regions (eyes, mouth, jawline, etc.)

---

## 📁 Project Structure

```
├── app.py                              # Streamlit web interface
├── xception_gan_augmented.pth          # Primary GAN-refined model
├── best_xception.pth                   # Baseline XceptionNet model
├── FFPP2.pt                           # FaceForensics++ trained model
│
├── model_def.py                        # Model architecture definitions
├── xception_model.py                   # XceptionNet implementation
├── gradcam_utils.py                    # GradCAM utility functions
├── xception_trust_aware_inference.py   # Trust-aware inference engine
│
├── test_image_gradcam.py              # Single image GradCAM testing
├── test_ffpp2_gradcam.py              # FFPP2 model GradCAM testing
├── create_gradcam_comparison.py        # Side-by-side comparison generator
├── show_gradcam_grid.py               # Grid visualization
├── local_manipulation_attribution.py   # Batch attribution analysis
│
├── deepfake-terminal-ui/              # Terminal-style web interface
│   ├── api/                           # FastAPI backend
│   │   ├── server.py                  # API server
│   │   └── requirements.txt           # API dependencies
│   └── src/                           # Next.js frontend
│
├── Dataset/                           # FaceForensics++ dataset
├── reduced_dataset/                   # Preprocessed training data
├── split_dataset/                     # Train/Val/Test splits
├── attribution_results/               # GradCAM output storage
│
└── requirements.txt                   # Python dependencies
```

---

 Training

### GAN-Augmented Training Pipeline

The model uses a three-stage adversarial training approach:

1. **Adversarial Training**: Generator-discriminator framework
2. **L1 Reconstruction Loss**: Pixel-level feature preservation  
3. **Frequency Domain Analysis**: DCT-based artifact detection

**Notebooks**:
- `XCEPTION_NET(RETRAINING).ipynb`: Full training pipeline
- `VIT.ipynb`: Vision Transformer experiments

**Key Hyperparameters**:
- Learning Rate: 1e-4
- Batch Size: 32
- Epochs: 50
- Optimizer: Adam
- Loss: BCE + L1 + Frequency Loss

---

## 🔬 Technical Details

### Preprocessing
- Face detection and alignment
- 299×299 resizing
- Normalization: ImageNet statistics
- Data augmentation: Rotation, flipping, color jitter

### Inference Pipeline
1. Face extraction from image/video
2. Preprocessing and normalization
3. Model prediction (logits + confidence)
4. GradCAM attribution (optional)
5. Trust score calculation

### Trust-Aware Features
- Confidence thresholding
- Uncertainty quantification
- Multi-model ensemble (optional)
- Adversarial robustness testing

---

## 📈 Results & Benchmarks

### Per-Manipulation Performance (GAN-Refined Model)

| Manipulation Type | Accuracy | Notes |
|------------------|----------|-------|
| **Deepfakes** | 96.2% | Best detection rate |
| **Face2Face** | 94.8% | Expression-based manipulation |
| **FaceSwap** | 94.3% | Traditional CV techniques |
| **Overall** | 95.1% | Cross-manipulation generalization |

### Inference Speed
- **GPU (CUDA)**: ~15ms per image
- **CPU**: ~80ms per image
- **Batch (GPU)**: ~8ms per image (batch size 32)

---

## 📚 Documentation

- [Streamlit App Guide](README_streamlit.md)
- [GradCAM Usage](README_gradcam.md)
- [Architecture Overview](ARCHITECTURE_DIAGRAM.md)
- [Quick Start](QUICKSTART.md)
- [Streamlit Package Summary](STREAMLIT_PACKAGE_SUMMARY.md)

---

## 🛠️ API Reference

### FastAPI Endpoints

**Health Check**:
```http
GET /health
```

**Predict Image**:
```http
POST /predict
Content-Type: multipart/form-data

Body: image file
```

**Response**:
```json
{
  "prediction": "fake",
  "confidence": 0.978,
  "model": "xception_gan_augmented",
  "processing_time_ms": 145
}
```

**Generate GradCAM**:
```http
POST /gradcam
Content-Type: multipart/form-data

Body: image file
```



---




---

