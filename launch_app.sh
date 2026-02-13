#!/bin/bash
# Quick launcher script for Trust-Aware Deepfake Detector

echo "🚀 Launching Trust-Aware Deepfake Detector..."
echo ""
echo "📋 Pre-flight checks:"

# Check if weights exist
if [ -f "weights/xception_gan_augmented.pth" ]; then
    echo "✅ Model checkpoint found"
else
    echo "❌ Model checkpoint not found at weights/xception_gan_augmented.pth"
    echo "   Please copy your checkpoint to the weights/ directory"
    exit 1
fi

# Check Python dependencies
echo "🔍 Checking dependencies..."

python -c "import streamlit" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Streamlit installed"
else
    echo "❌ Streamlit not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi

python -c "import torch" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ PyTorch installed"
else
    echo "❌ PyTorch not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi

python -c "import timm" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ timm installed"
else
    echo "❌ timm not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "🎯 All checks passed! Starting Streamlit app..."
echo ""
echo "📱 The app will open in your browser at: http://localhost:8501"
echo "   Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
streamlit run app.py
