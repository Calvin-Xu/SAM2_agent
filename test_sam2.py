"""Test SAM2 mask generation and verify setup"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import sys
from pathlib import Path

# Add the project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import SAM2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def test_sam2_setup():
    """Test that SAM2 is properly installed and configured"""

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print(f"\nLoading SAM2 model...")
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {model_cfg}")

    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    print("Model loaded successfully!")

    # Create mask generator
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    print("Mask generator created successfully!")

    # Load and process test image
    image_path = "tmp/inputs/truck.jpg"
    print(f"\nLoading test image: {image_path}")
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    print(f"Image shape: {image.shape}")

    # Generate masks
    print("\nGenerating masks...")
    masks = mask_generator.generate(image)
    print(f"Generated {len(masks)} masks!")

    # Print mask statistics
    if len(masks) > 0:
        areas = [m['area'] for m in masks]
        print(f"\nMask statistics:")
        print(f"  Min area: {min(areas)}")
        print(f"  Max area: {max(areas)}")
        print(f"  Mean area: {np.mean(areas):.1f}")
        print(f"  Median area: {np.median(areas):.1f}")

    return True

if __name__ == "__main__":
    try:
        test_sam2_setup()
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
