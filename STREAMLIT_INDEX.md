# 📑 Streamlit App Documentation Index

> **Complete Trust-Aware Deepfake Detector with Grad-CAM Explainability**

## 🚀 Quick Access

**Want to start immediately?** → Read [QUICKSTART.md](QUICKSTART.md)

**Need full documentation?** → Read [README_streamlit.md](README_streamlit.md)

**Want to understand the code?** → Read [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)

**Looking for complete overview?** → Read [STREAMLIT_PACKAGE_SUMMARY.md](STREAMLIT_PACKAGE_SUMMARY.md)

---

## 📂 File Guide

### Core Application Files (Required)

| File | Size | Purpose |
|------|------|---------|
| `app.py` | 20 KB | Main Streamlit application with UI and logic |
| `model_def.py` | 5 KB | Model architecture builder and checkpoint loader |
| `gradcam_utils.py` | 6 KB | Grad-CAM implementation for visual explanation |
| `requirements.txt` | 539 B | Python package dependencies |
| `weights/xception_gan_augmented.pth` | 80 MB | Pretrained model checkpoint |

### Supporting Files (Optional but Recommended)

| File | Purpose |
|------|---------|
| `launch_app.sh` | One-command launcher with pre-flight checks |
| `README_streamlit.md` | Comprehensive feature documentation |
| `QUICKSTART.md` | Step-by-step user guide |
| `STREAMLIT_PACKAGE_SUMMARY.md` | Complete package overview |
| `ARCHITECTURE_DIAGRAM.md` | Visual architecture reference |
| `STREAMLIT_INDEX.md` | This file - documentation index |

---

## 📚 Documentation Map

### For Different User Types

#### 🆕 **First-Time Users**
Start here to get running quickly:
1. Read: [QUICKSTART.md](QUICKSTART.md) → Installation & Usage
2. Run: `./launch_app.sh`
3. Test: Upload a sample image in the app

#### 🔧 **Developers/Customizers**
Understand the codebase structure:
1. Read: [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) → Code structure
2. Edit: `app.py` constants (IMG_SIZE, MEAN, STD)
3. Modify: `model_def.py` → `build_model()` function

#### 📖 **Power Users**
Explore advanced features:
1. Read: [README_streamlit.md](README_streamlit.md) → Full feature list
2. Read: [STREAMLIT_PACKAGE_SUMMARY.md](STREAMLIT_PACKAGE_SUMMARY.md) → Customization points
3. Experiment: Different verdict methods for videos

#### 🚀 **Deployers**
Get the app to production:
1. Read: [README_streamlit.md](README_streamlit.md) → Deployment section
2. Choose: Streamlit Cloud / Docker / Cloud VM
3. Deploy: Follow platform-specific instructions

---

## 🎯 Common Tasks

### Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python model_def.py

# Launch app
./launch_app.sh
```

**Documentation:** [QUICKSTART.md](QUICKSTART.md) → "Installation (5 minutes)"

---

### Using the Image Tab

1. Click **"📷 Image Analysis"** tab
2. Upload image (JPG/PNG/BMP/WebP)
3. View prediction + confidence
4. Check Grad-CAM heatmap
5. Read explanation

**Documentation:** [QUICKSTART.md](QUICKSTART.md) → "Using the App → Image Analysis"

---

### Using the Video Tab

1. Click **"🎬 Video Analysis"** tab
2. Configure sidebar settings:
   - ☑️ Show Grad-CAM
   - 🎚️ Sample every 15 frames
   - 🎚️ Max 40 frames
   - 📊 Verdict: Mean FAKE prob
3. Upload video (MP4/MOV/AVI/MKV)
4. Wait for analysis
5. Review timeline + top frames

**Documentation:** [QUICKSTART.md](QUICKSTART.md) → "Using the App → Video Analysis"

---

### Customizing Preprocessing

Edit `app.py` lines 24-26:

```python
IMG_SIZE = 299
MEAN = (0.485, 0.456, 0.406)  # ImageNet mean
STD = (0.229, 0.224, 0.225)   # ImageNet std
```

**Documentation:** [README_streamlit.md](README_streamlit.md) → "Configuration"

---

### Changing Model Architecture

Edit `model_def.py` → `build_model()`:

```python
def build_model(num_classes=2):
    model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
    return model
```

**Documentation:** [STREAMLIT_PACKAGE_SUMMARY.md](STREAMLIT_PACKAGE_SUMMARY.md) → "Architecture Changes"

---

### Troubleshooting

**Problem:** Model won't load
- **Solution:** Check `weights/xception_gan_augmented.pth` exists
- **Docs:** [README_streamlit.md](README_streamlit.md) → "Troubleshooting"

**Problem:** Grad-CAM not showing
- **Solution:** Ensure "Show Grad-CAM" checkbox is enabled
- **Docs:** [QUICKSTART.md](QUICKSTART.md) → "Troubleshooting → Grad-CAM not showing"

**Problem:** Video processing slow
- **Solution:** Reduce max frames, disable Grad-CAM, or use GPU
- **Docs:** [QUICKSTART.md](QUICKSTART.md) → "Troubleshooting → Video processing slow"

---

## 🔍 Feature Reference

### UI Components

| Feature | Location | Description |
|---------|----------|-------------|
| Image tab | Main area | Single image analysis |
| Video tab | Main area | Multi-frame video analysis |
| Sidebar | Left side | Configuration settings |
| Grad-CAM toggle | Sidebar | Enable/disable heatmaps |
| Frame sampling | Sidebar | Control video frame extraction |
| Verdict method | Sidebar | Choose aggregation strategy |

**Docs:** [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) → "UI Component Hierarchy"

---

### Model Features

| Feature | Implementation | Configuration |
|---------|---------------|---------------|
| Architecture | `model_def.py` | `build_model()` |
| Checkpoint loading | `model_def.py` | `load_checkpoint()` |
| Preprocessing | `app.py` | `preprocess_pil()` |
| Inference | `app.py` | `predict_proba()` |
| GPU support | Automatic | `torch.device()` |

**Docs:** [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) → "Model Checkpoint Loading Logic"

---

### Grad-CAM Features

| Feature | Implementation | Configuration |
|---------|---------------|---------------|
| CAM generation | `gradcam_utils.py` | `GradCAM` class |
| Auto layer detection | `gradcam_utils.py` | `_find_last_conv_layer()` |
| Heatmap overlay | `gradcam_utils.py` | `overlay_cam_on_image()` |
| Alpha blending | `app.py` | `GRADCAM_ALPHA = 0.5` |
| Colormap | OpenCV | `cv2.COLORMAP_JET` |

**Docs:** [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) → "Grad-CAM Pipeline"

---

### Video Processing Features

| Feature | Implementation | Configuration |
|---------|---------------|---------------|
| Frame sampling | `app.py` | `sample_video_frames()` |
| Verdict methods | `app.py` | `compute_video_verdict()` |
| Timeline plot | matplotlib | `plot_suspicion_timeline()` |
| Top frames | Streamlit grid | Sort by P(fake) |
| Statistics | Pandas-style | Mean/Max/Count |

**Docs:** [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) → "Video Analysis Flow"

---

## 📖 Documentation Deep Dive

### README_streamlit.md
**Purpose:** Comprehensive feature documentation
**Best for:** Understanding all capabilities, deployment options
**Sections:**
- Quick Start (installation)
- Features (image/video/Grad-CAM)
- Configuration (constants, settings)
- Troubleshooting (common errors)
- Model Checkpoint Formats (4 types)
- Deployment (local/cloud/Docker)
- Adaptation Examples (preprocessing, multi-class)

### QUICKSTART.md
**Purpose:** Step-by-step user guide
**Best for:** First-time users, practical tutorials
**Sections:**
- Installation (3 steps)
- Launching App (launcher script)
- Using Image Analysis (workflow)
- Using Video Analysis (workflow)
- Understanding Grad-CAM (color interpretation)
- Troubleshooting (with solutions)
- Tips & Best Practices
- Example Workflows (3 scenarios)
- Advanced Usage (GPU, custom preprocessing)

### STREAMLIT_PACKAGE_SUMMARY.md
**Purpose:** Complete package overview
**Best for:** Developers, project maintainers
**Sections:**
- Files Created (with sizes)
- Key Features Implemented (checklist)
- Code Structure (functions, constants)
- Usage Instructions (commands)
- Customization Points (easy/medium/advanced)
- Performance Characteristics (timing)
- Safety & Error Handling
- Documentation Coverage
- Testing Checklist
- Future Enhancements (ideas)

### ARCHITECTURE_DIAGRAM.md
**Purpose:** Visual architecture reference
**Best for:** Understanding code flow, data pipelines
**Sections:**
- ASCII Architecture Diagram
- Data Flow Diagrams (image/video)
- File Dependencies Graph
- Model Checkpoint Loading Logic
- Grad-CAM Pipeline
- UI Component Hierarchy
- Error Handling Strategy
- Deployment Options
- Performance Optimization
- Customization Entry Points

---

## 🎓 Learning Path

### Beginner Track (Just Want to Use It)
1. **QUICKSTART.md** → Installation
2. Launch app
3. Try image analysis
4. Try video analysis
5. **QUICKSTART.md** → Tips section

### Intermediate Track (Want to Customize)
1. **README_streamlit.md** → Configuration
2. Edit `app.py` constants
3. Test with custom preprocessing
4. **QUICKSTART.md** → Advanced Usage
5. Try different models

### Advanced Track (Want to Extend)
1. **ARCHITECTURE_DIAGRAM.md** → Full architecture
2. **STREAMLIT_PACKAGE_SUMMARY.md** → Customization points
3. Modify core functions
4. Add new features
5. Deploy to production

---

## 🔗 Quick Links

### External Resources
- **Streamlit Docs:** https://docs.streamlit.io
- **PyTorch Docs:** https://pytorch.org/docs
- **timm Docs:** https://huggingface.co/docs/timm
- **Grad-CAM Paper:** Selvaraju et al., ICCV 2017

### Internal Code References
- **Main app logic:** `app.py` → `main()` function
- **Model building:** `model_def.py` → `build_model()`
- **Grad-CAM:** `gradcam_utils.py` → `GradCAM` class
- **Preprocessing:** `app.py` → `preprocess_pil()`
- **Video processing:** `app.py` → `sample_video_frames()`

---

## ✅ Verification Checklist

Before using the app, ensure:

- [ ] All files present in project directory
- [ ] `pip install -r requirements.txt` completed
- [ ] `python model_def.py` runs successfully
- [ ] `weights/xception_gan_augmented.pth` exists
- [ ] Streamlit installed (`streamlit --version`)
- [ ] Launch script executable (`chmod +x launch_app.sh`)

**All green?** → Run `./launch_app.sh` 🚀

---

## 💡 Pro Tips

1. **Keep docs open:** Reference documentation while using the app
2. **Test incrementally:** Start with images before videos
3. **Check terminal:** Useful debug info printed to console
4. **Adjust sidebar:** Experiment with different settings
5. **GPU matters:** 10x speed boost if CUDA available
6. **Read explanations:** Plain-English text aids interpretation

---

## 📞 Support Matrix

| Question | Best Resource |
|----------|---------------|
| How do I install? | [QUICKSTART.md](QUICKSTART.md) |
| What features exist? | [README_streamlit.md](README_streamlit.md) |
| How does it work? | [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) |
| How do I customize? | [STREAMLIT_PACKAGE_SUMMARY.md](STREAMLIT_PACKAGE_SUMMARY.md) |
| Model won't load? | [README_streamlit.md](README_streamlit.md) → Troubleshooting |
| Grad-CAM issues? | [QUICKSTART.md](QUICKSTART.md) → Troubleshooting |
| How to deploy? | [README_streamlit.md](README_streamlit.md) → Deployment |

---

## 🎉 You're All Set!

**Everything you need is documented.** Choose your path:

- 🏃 **Quick Start:** Run `./launch_app.sh` now!
- 📖 **Learn First:** Read [QUICKSTART.md](QUICKSTART.md)
- 🔧 **Customize:** Review [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)
- 🚀 **Deploy:** Follow [README_streamlit.md](README_streamlit.md)

**Happy detecting! 🔍**

---

*Last updated: December 14, 2025*
