#!/usr/bin/env python3
"""
xception_model.py

Exact model construction used in XCEPTION_NET.ipynb for deepfake detection.
This module exports the create_xception_model() function that builds the same
timm Xception model with num_classes=1 for BCE loss as used in training.
"""

import torch
import torch.nn as nn


def create_xception_model(num_classes: int = 1, pretrained: bool = False) -> torch.nn.Module:
    """
    Create the exact Xception model used in the notebook training.
    
    Args:
        num_classes: Number of output classes (default 1 for BCE)
        pretrained: Whether to use pretrained weights (default False for training from scratch)
        
    Returns:
        torch.nn.Module: The Xception model
    """
    try:
        import timm
        
        # This matches the exact call from the notebook:
        # model = timm.create_model('xception', pretrained=True, num_classes=1).to(device)
        model = timm.create_model('xception', pretrained=pretrained, num_classes=num_classes)
        
        return model
        
    except ImportError as e:
        raise RuntimeError(
            f"timm is required for Xception model creation. "
            f"Install with: pip install timm\n"
            f"Original error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create Xception model: {e}")


def load_checkpoint_metadata(checkpoint_path: str) -> dict:
    """
    Load metadata from the checkpoint saved during training.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        dict: Metadata including class_to_idx, img_size, threshold
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    metadata = {
        "class_to_idx": checkpoint.get("class_to_idx", {"fake": 0, "real": 1}),
        "img_size": checkpoint.get("img_size", 299),
        "threshold": checkpoint.get("threshold", 0.5),
    }
    
    return metadata


def load_trained_model(checkpoint_path: str, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """
    Load the trained Xception model from checkpoint with metadata.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on
        
    Returns:
        tuple: (model, metadata_dict)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Create model with same architecture
    model = create_xception_model(num_classes=1, pretrained=False)
    
    # Load weights - the notebook saves under "model" key
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint
    
    # Handle DataParallel prefixes if present
    if isinstance(state_dict, dict) and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Extract metadata
    metadata = load_checkpoint_metadata(checkpoint_path)
    
    return model, metadata


if __name__ == "__main__":
    # Quick test
    print("Testing Xception model creation...")
    
    try:
        model = create_xception_model()
        print(f"✅ Model created successfully: {type(model)}")
        
        # Test forward pass
        x = torch.randn(1, 3, 299, 299)
        with torch.no_grad():
            output = model(x)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")