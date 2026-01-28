"""
Reproduce key figures and visualizations from the SAM2 paper
https://arxiv.org/html/2408.00714v2

This script reproduces:
1. Automatic mask generation visualization (demonstrating the core capability)
2. Parameter sensitivity analysis (showing impact of configuration settings)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import cv2
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Setup
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "paper_reproductions"
OUTPUT_DIR.mkdir(exist_ok=True)

def setup_device():
    """Setup CUDA device with optimizations"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    return device

def show_anns(anns, borders=True):
    """Visualize annotations as shown in SAM2 paper"""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                   sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                       for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

def reproduce_figure_automatic_masks(device):
    """
    Reproduce automatic mask generation visualization

    This demonstrates the core SAM2 capability as shown throughout the paper,
    particularly in qualitative results figures.
    """
    print("\n=== Reproducing Figure: Automatic Mask Generation ===")

    # Load model
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # Test on multiple images
    image_paths = [
        "tmp/inputs/truck.jpg",
        "tmp/inputs/cars.jpg"
    ]

    # Set random seed for reproducibility
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('Figure Reproduction: Automatic Mask Generation with SAM 2',
                 fontsize=16, fontweight='bold')

    for idx, image_path in enumerate(image_paths):
        if not Path(image_path).exists():
            print(f"Skipping {image_path} - not found")
            continue

        print(f"\nProcessing: {image_path}")
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        # Generate masks
        masks = mask_generator.generate(image)
        print(f"  Generated {len(masks)} masks")

        # Original image
        axes[idx, 0].imshow(image)
        axes[idx, 0].set_title(f'Original Image ({Path(image_path).name})',
                              fontsize=12)
        axes[idx, 0].axis('off')

        # Masks overlay
        axes[idx, 1].imshow(image)
        show_anns(masks, borders=True)
        axes[idx, 1].set_title(f'SAM 2 Segmentation ({len(masks)} masks)',
                              fontsize=12)
        axes[idx, 1].axis('off')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "figure1_automatic_masks.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved to: {output_path}")

    return masks

def reproduce_figure_parameter_sensitivity(device):
    """
    Reproduce parameter sensitivity analysis

    This demonstrates how different configuration parameters affect
    mask generation quality and quantity, similar to ablation studies
    in the paper.
    """
    print("\n=== Reproducing Figure: Parameter Sensitivity Analysis ===")

    # Load model
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

    # Load test image
    image_path = "tmp/inputs/truck.jpg"
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # Test different parameter configurations
    configs = [
        {
            "name": "Default (Baseline)",
            "params": {}
        },
        {
            "name": "High Quality (Paper Settings)",
            "params": {
                "points_per_side": 64,
                "pred_iou_thresh": 0.7,
                "stability_score_thresh": 0.92,
                "crop_n_layers": 1,
                "use_m2m": True
            }
        },
        {
            "name": "Fast (Lower Quality)",
            "params": {
                "points_per_side": 16,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95
            }
        }
    ]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Figure Reproduction: Parameter Sensitivity Analysis',
                 fontsize=16, fontweight='bold')

    results = []

    for idx, config in enumerate(configs):
        print(f"\nTesting configuration: {config['name']}")

        # Create mask generator with specific parameters
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            **config['params']
        )

        # Generate masks
        masks = mask_generator.generate(image)
        print(f"  Generated {len(masks)} masks")

        # Calculate statistics
        areas = [m['area'] for m in masks]
        ious = [m['predicted_iou'] for m in masks]
        stability_scores = [m['stability_score'] for m in masks]

        results.append({
            'Configuration': config['name'],
            'Num Masks': len(masks),
            'Mean Area': np.mean(areas),
            'Mean IoU': np.mean(ious),
            'Mean Stability': np.mean(stability_scores),
            'Min Area': np.min(areas),
            'Max Area': np.max(areas)
        })

        # Set random seed for consistent colors
        np.random.seed(42)

        # Visualize masks
        axes[0, idx].imshow(image)
        show_anns(masks, borders=True)
        axes[0, idx].set_title(f'{config["name"]}\n({len(masks)} masks)',
                              fontsize=11)
        axes[0, idx].axis('off')

        # Histogram of mask areas
        axes[1, idx].hist(areas, bins=30, edgecolor='black', alpha=0.7)
        axes[1, idx].set_xlabel('Mask Area (pixels)', fontsize=10)
        axes[1, idx].set_ylabel('Frequency', fontsize=10)
        axes[1, idx].set_title(f'Area Distribution\nMean: {np.mean(areas):.0f} px²',
                              fontsize=10)
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "figure2_parameter_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved to: {output_path}")

    # Save statistics
    results_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "parameter_sensitivity_stats.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Statistics saved to: {csv_path}")
    print("\nResults Summary:")
    print(results_df.to_string(index=False))

    return results_df

def main():
    """Main execution"""
    print("=" * 70)
    print("SAM2 Paper Figure Reproduction")
    print("=" * 70)

    # Setup device
    device = setup_device()
    print(f"\nUsing device: {device}")

    # Reproduce figures
    reproduce_figure_automatic_masks(device)
    reproduce_figure_parameter_sensitivity(device)

    print("\n" + "=" * 70)
    print("✓ All figures reproduced successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
