"""
Model Definition for Deepfake Detection

This module provides the model architecture builder.
Adapt this to match your training code's model structure.
"""

import torch
import torch.nn as nn

def build_model(num_classes=1):
    """
    Build the Xception model for deepfake detection.
    
    This model uses num_classes=1 for binary classification with sigmoid output.
    The training checkpoint was created with a single output neuron.
    
    For Xception-based models, you likely used one of:
    1) timm.create_model('xception', pretrained=True, num_classes=num_classes)
    2) timm.create_model('legacy_xception', pretrained=True, num_classes=num_classes)
    3) Custom Xception from torchvision or other source
    
    Args:
        num_classes (int): Number of output neurons (default: 1 for binary with sigmoid)
    
    Returns:
        torch.nn.Module: The model instance
    
    Note: Your model was trained with num_classes=1 (single logit + sigmoid).
    """
    
    # ============================================================================
    # MODEL IMPLEMENTATION - Matches training configuration
    # ============================================================================
    
    try:
        # Attempt to use timm (most common for Xception)
        import timm
        
        # Use 'xception' (not legacy_xception) - matches training code
        model = timm.create_model('xception', pretrained=False, num_classes=num_classes)
        print(f"✅ Loaded xception with {num_classes} output(s)")
        return model
        
    except ImportError:
        print("⚠️  timm not installed. Install with: pip install timm")
        raise
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError(
            "Could not create model. Please update build_model() in model_def.py "
            "to match your training code's architecture."
        )


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Robustly load a checkpoint into the model.
    
    Handles multiple checkpoint formats:
    - Full model saved with torch.save(model, path)
    - State dict saved with torch.save(model.state_dict(), path)
    - Training checkpoint with 'state_dict' key
    - Checkpoint with 'model' key containing full model
    
    Args:
        model: The model instance to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load checkpoint on
    
    Returns:
        model: The model with loaded weights
    """
    
    print(f"🔧 Loading checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint (use weights_only=False for compatibility)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract state_dict using same logic as local_manipulation_attribution.py
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict') or checkpoint.get('model', checkpoint)
            if not isinstance(state_dict, dict):
                state_dict = state_dict.state_dict() if hasattr(state_dict, 'state_dict') else checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        print("   Format: Extracted and cleaned state dict")
        
        # Load with strict=False to handle minor mismatches
        model.load_state_dict(state_dict, strict=False)
        
        print("✅ Checkpoint loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise


if __name__ == "__main__":
    # Test model building
    print("Testing model creation...")
    model = build_model(num_classes=2)
    print(f"Model type: {type(model)}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
