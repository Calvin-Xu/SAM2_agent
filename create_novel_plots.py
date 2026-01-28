"""
Create novel analysis plots related to the SAM2 paper

This script creates new plots that provide additional insights beyond
what's shown in the original paper:

1. Mask Quality vs Size Analysis: Examining the relationship between
   mask size and quality metrics (IoU, stability)

2. Hierarchical Mask Coverage Analysis: Visualizing how masks of different
   sizes contribute to overall image coverage and analyzing overlap patterns
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import pandas as pd
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import seaborn as sns

# Setup
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "novel_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

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

def novel_plot_1_quality_vs_size(device):
    """
    Novel Plot 1: Mask Quality vs Size Analysis

    This plot explores an interesting question not addressed in the paper:
    Is there a relationship between mask size and quality metrics?

    Hypothesis: Larger objects might be easier to segment with higher
    quality (IoU, stability), or conversely, smaller objects might have
    more variable quality.
    """
    print("\n=== Novel Plot 1: Mask Quality vs Size Analysis ===")

    # Load model
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

    # Test on multiple images for more data
    image_paths = [
        "tmp/inputs/truck.jpg",
        "tmp/inputs/cars.jpg"
    ]

    all_data = []

    for image_path in image_paths:
        if not Path(image_path).exists():
            continue

        print(f"\nProcessing: {image_path}")
        image = Image.open(image_path)
        image_np = np.array(image.convert("RGB"))

        # Generate masks with high-quality settings
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            use_m2m=True
        )

        masks = mask_generator.generate(image_np)
        print(f"  Generated {len(masks)} masks")

        # Collect data
        for mask in masks:
            area = mask['area']
            total_pixels = image_np.shape[0] * image_np.shape[1]
            relative_size = area / total_pixels

            all_data.append({
                'area': area,
                'relative_size': relative_size,
                'predicted_iou': mask['predicted_iou'],
                'stability_score': mask['stability_score'],
                'bbox_w': mask['bbox'][2],
                'bbox_h': mask['bbox'][3],
                'aspect_ratio': mask['bbox'][2] / max(mask['bbox'][3], 1),
                'image': Path(image_path).stem
            })

    df = pd.DataFrame(all_data)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle('Novel Analysis 1: Mask Quality Characteristics vs Size',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: IoU vs Area (scatter)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(df['area'], df['predicted_iou'],
                          c=df['stability_score'], cmap='viridis',
                          alpha=0.6, s=50)
    ax1.set_xlabel('Mask Area (pixels)', fontsize=10)
    ax1.set_ylabel('Predicted IoU', fontsize=10)
    ax1.set_title('IoU vs Area (colored by Stability)', fontsize=11)
    ax1.set_xscale('log')
    plt.colorbar(scatter1, ax=ax1, label='Stability Score')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Stability vs Area (scatter)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(df['area'], df['stability_score'],
               alpha=0.6, s=50, c='coral')
    ax2.set_xlabel('Mask Area (pixels)', fontsize=10)
    ax2.set_ylabel('Stability Score', fontsize=10)
    ax2.set_title('Stability vs Area', fontsize=11)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Quality by Size Bins
    ax3 = fig.add_subplot(gs[0, 2])
    df['size_bin'] = pd.cut(df['relative_size'],
                            bins=[0, 0.001, 0.01, 0.1, 1.0],
                            labels=['Tiny\n(<0.1%)', 'Small\n(0.1-1%)',
                                   'Medium\n(1-10%)', 'Large\n(>10%)'])
    size_quality = df.groupby('size_bin')[['predicted_iou', 'stability_score']].mean()
    x = np.arange(len(size_quality))
    width = 0.35
    ax3.bar(x - width/2, size_quality['predicted_iou'], width,
           label='Mean IoU', alpha=0.8)
    ax3.bar(x + width/2, size_quality['stability_score'], width,
           label='Mean Stability', alpha=0.8)
    ax3.set_xlabel('Mask Size (% of image)', fontsize=10)
    ax3.set_ylabel('Quality Metric', fontsize=10)
    ax3.set_title('Average Quality by Size Category', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(size_quality.index)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Aspect Ratio vs Quality
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(df['aspect_ratio'], df['predicted_iou'],
               alpha=0.6, s=50, c='steelblue')
    ax4.set_xlabel('Aspect Ratio (width/height)', fontsize=10)
    ax4.set_ylabel('Predicted IoU', fontsize=10)
    ax4.set_title('IoU vs Aspect Ratio', fontsize=11)
    ax4.set_xlim(0, 5)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Distribution of Quality Metrics
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(df['predicted_iou'], bins=30, alpha=0.5,
            label='Predicted IoU', color='blue', edgecolor='black')
    ax5.hist(df['stability_score'], bins=30, alpha=0.5,
            label='Stability Score', color='orange', edgecolor='black')
    ax5.set_xlabel('Score', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Distribution of Quality Metrics', fontsize=11)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Size Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(df['relative_size'] * 100, bins=50,
            edgecolor='black', alpha=0.7, color='green')
    ax6.set_xlabel('Mask Size (% of image)', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Distribution of Mask Sizes', fontsize=11)
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)

    # Plot 7-9: Heatmap correlation matrix
    ax7 = fig.add_subplot(gs[2, :])
    corr_data = df[['area', 'relative_size', 'predicted_iou',
                    'stability_score', 'aspect_ratio']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm',
               center=0, ax=ax7, cbar_kws={'label': 'Correlation'})
    ax7.set_title('Correlation Matrix: Mask Properties vs Quality', fontsize=11)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "novel_plot_1_quality_vs_size.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved to: {output_path}")

    # Save statistics
    stats_path = OUTPUT_DIR / "quality_vs_size_stats.csv"
    df.to_csv(stats_path, index=False)
    print(f"✓ Data saved to: {stats_path}")

    # Print insights
    print("\nKey Insights:")
    print(f"  - Total masks analyzed: {len(df)}")
    print(f"  - Mean IoU: {df['predicted_iou'].mean():.3f} (±{df['predicted_iou'].std():.3f})")
    print(f"  - Mean Stability: {df['stability_score'].mean():.3f} (±{df['stability_score'].std():.3f})")
    print(f"  - Correlation (area vs IoU): {df['area'].corr(df['predicted_iou']):.3f}")
    print(f"  - Correlation (area vs stability): {df['area'].corr(df['stability_score']):.3f}")

    return df

def novel_plot_2_hierarchical_coverage(device):
    """
    Novel Plot 2: Hierarchical Mask Coverage Analysis

    This plot analyzes how masks of different sizes contribute to overall
    image coverage and examines overlap patterns. This addresses questions
    not explored in the paper about the hierarchical nature of segmentation.
    """
    print("\n\n=== Novel Plot 2: Hierarchical Mask Coverage Analysis ===")

    # Load model
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

    # Load image
    image_path = "tmp/inputs/truck.jpg"
    print(f"\nProcessing: {image_path}")
    image = Image.open(image_path)
    image_np = np.array(image.convert("RGB"))
    h, w = image_np.shape[:2]

    # Generate masks
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        use_m2m=True
    )

    masks = mask_generator.generate(image_np)
    print(f"  Generated {len(masks)} masks")

    # Sort masks by area
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Create visualization
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Novel Analysis 2: Hierarchical Coverage and Overlap Patterns',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np)
    ax1.set_title('Original Image', fontsize=11)
    ax1.axis('off')

    # Plot 2: Cumulative coverage
    ax2 = fig.add_subplot(gs[0, 1])
    coverage_map = np.zeros((h, w), dtype=int)
    for mask in sorted_masks:
        coverage_map[mask['segmentation']] += 1

    im2 = ax2.imshow(coverage_map, cmap='hot', interpolation='nearest')
    ax2.set_title('Overlap Heatmap (higher = more overlapping masks)', fontsize=11)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Number of masks')

    # Plot 3: Coverage by size tiers
    ax3 = fig.add_subplot(gs[0, 2])
    n_tiers = 4
    tier_size = len(sorted_masks) // n_tiers
    colors = plt.cm.Set3(np.linspace(0, 1, n_tiers))

    tier_coverage = np.zeros((h, w, 3))
    for tier in range(n_tiers):
        start_idx = tier * tier_size
        end_idx = (tier + 1) * tier_size if tier < n_tiers - 1 else len(sorted_masks)
        tier_masks = sorted_masks[start_idx:end_idx]

        tier_map = np.zeros((h, w), dtype=bool)
        for mask in tier_masks:
            tier_map |= mask['segmentation']

        tier_coverage[tier_map] = colors[tier][:3]

    ax3.imshow(tier_coverage)
    ax3.set_title('Coverage by Size Tiers\n(largest to smallest)', fontsize=11)
    ax3.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'Tier {i+1}')
                      for i in range(n_tiers)]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Plot 4: Cumulative coverage curve
    ax4 = fig.add_subplot(gs[1, 0])
    cumulative_coverage = []
    covered_pixels = np.zeros((h, w), dtype=bool)

    for i, mask in enumerate(sorted_masks):
        covered_pixels |= mask['segmentation']
        coverage_pct = np.sum(covered_pixels) / (h * w) * 100
        cumulative_coverage.append(coverage_pct)

    ax4.plot(range(1, len(cumulative_coverage) + 1), cumulative_coverage,
            linewidth=2, color='steelblue')
    ax4.fill_between(range(1, len(cumulative_coverage) + 1),
                     cumulative_coverage, alpha=0.3, color='steelblue')
    ax4.set_xlabel('Number of Masks (sorted by size)', fontsize=10)
    ax4.set_ylabel('Image Coverage (%)', fontsize=10)
    ax4.set_title('Cumulative Coverage Curve', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95% coverage')
    ax4.legend(fontsize=9)

    # Plot 5: Overlap distribution
    ax5 = fig.add_subplot(gs[1, 1])
    overlap_counts = coverage_map.flatten()
    unique, counts = np.unique(overlap_counts, return_counts=True)
    ax5.bar(unique, counts, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Number of Overlapping Masks', fontsize=10)
    ax5.set_ylabel('Number of Pixels', fontsize=10)
    ax5.set_title('Overlap Distribution', fontsize=11)
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Size vs Coverage Contribution
    ax6 = fig.add_subplot(gs[1, 2])
    contributions = []
    prev_coverage = 0
    for mask in sorted_masks:
        # Calculate unique coverage contribution
        contribution = mask['area'] / (h * w) * 100
        contributions.append(contribution)

    ax6.bar(range(len(contributions)), contributions, alpha=0.7)
    ax6.set_xlabel('Mask Index (sorted by size)', fontsize=10)
    ax6.set_ylabel('Coverage Contribution (%)', fontsize=10)
    ax6.set_title('Individual Mask Coverage Contributions', fontsize=11)
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, axis='y')

    # Plot 7: Top masks visualization
    ax7 = fig.add_subplot(gs[2, 0])
    top_n = 10
    top_mask_viz = np.zeros((h, w, 4))
    colors_top = plt.cm.tab10(np.linspace(0, 1, top_n))

    for i, mask in enumerate(sorted_masks[:top_n]):
        mask_overlay = np.zeros((h, w, 4))
        mask_overlay[mask['segmentation']] = colors_top[i]
        top_mask_viz = np.maximum(top_mask_viz, mask_overlay)

    ax7.imshow(image_np)
    ax7.imshow(top_mask_viz, alpha=0.6)
    ax7.set_title(f'Top {top_n} Largest Masks', fontsize=11)
    ax7.axis('off')

    # Plot 8: Statistics table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')

    # Calculate coverage milestones safely
    def get_coverage_milestone(target):
        try:
            return next(i for i, c in enumerate(cumulative_coverage) if c >= target)
        except StopIteration:
            return None

    masks_50 = get_coverage_milestone(50)
    masks_90 = get_coverage_milestone(90)
    masks_95 = get_coverage_milestone(95)

    stats_text = f"""
    Coverage Statistics:

    Total Masks: {len(masks)}
    Total Image Coverage: {cumulative_coverage[-1]:.1f}%

    Masks for 50% coverage: {masks_50 if masks_50 is not None else 'N/A'}
    Masks for 90% coverage: {masks_90 if masks_90 is not None else 'N/A'}
    Masks for 95% coverage: {masks_95 if masks_95 is not None else 'N/A'}

    Average overlap per pixel: {coverage_map.mean():.2f} masks
    Max overlap: {coverage_map.max()} masks
    Pixels with no masks: {np.sum(coverage_map == 0)} ({np.sum(coverage_map == 0)/(h*w)*100:.1f}%)

    Top 10% of masks cover: {cumulative_coverage[len(masks)//10]:.1f}% of image
    Top 50% of masks cover: {cumulative_coverage[len(masks)//2]:.1f}% of image
    """

    ax8.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "novel_plot_2_hierarchical_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved to: {output_path}")

    print("\nKey Insights:")
    print(f"  - Total coverage: {cumulative_coverage[-1]:.1f}%")
    if masks_90 is not None:
        print(f"  - Masks for 90% coverage: {masks_90}")
    else:
        print(f"  - Masks for 90% coverage: Not reached (max {cumulative_coverage[-1]:.1f}%)")
    print(f"  - Average overlap: {coverage_map.mean():.2f} masks/pixel")
    print(f"  - Max overlap: {coverage_map.max()} masks")

def main():
    """Main execution"""
    print("=" * 70)
    print("SAM2 Novel Analysis Plots")
    print("Creating new visualizations not in the original paper")
    print("=" * 70)

    # Setup device
    device = setup_device()
    print(f"\nUsing device: {device}")

    # Create novel plots
    novel_plot_1_quality_vs_size(device)
    novel_plot_2_hierarchical_coverage(device)

    print("\n" + "=" * 70)
    print("✓ All novel analysis plots created successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
