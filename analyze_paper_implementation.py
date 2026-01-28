"""
Analyze the SAM2 implementation for potential discrepancies with the paper

This script:
1. Compares implementation parameters with paper specifications
2. Checks for any obvious bugs or inconsistencies
3. Validates that the code matches the described methodology
"""

import numpy as np
import torch
from pathlib import Path
import inspect
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

def analyze_default_parameters():
    """Analyze if default parameters match paper specifications"""
    print("\n=== Parameter Analysis ===")
    print("\nAnalyzing SAM2AutomaticMaskGenerator default parameters...")

    # Get the constructor signature
    sig = inspect.signature(SAM2AutomaticMaskGenerator.__init__)

    # Extract default values
    defaults = {}
    for param_name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty and param_name != 'self':
            defaults[param_name] = param.default

    print("\nDefault parameters in implementation:")
    for key, value in sorted(defaults.items()):
        print(f"  {key}: {value}")

    # Compare with paper specifications
    # According to the SAM2 paper, the recommended settings are:
    paper_specs = {
        'points_per_side': 64,  # Paper mentions dense sampling
        'pred_iou_thresh': 0.7,  # Lower threshold for higher recall
        'stability_score_thresh': 0.92,  # High stability requirement
        'crop_n_layers': 1,  # Multi-scale processing
        'use_m2m': True,  # Mask-to-mask refinement
    }

    print("\n\nPaper-recommended parameters:")
    for key, value in sorted(paper_specs.items()):
        impl_value = defaults.get(key, "NOT FOUND")
        match = "✓" if impl_value == value else "✗"
        print(f"  {match} {key}: Paper={value}, Implementation={impl_value}")

    return defaults, paper_specs

def check_mask_generator_consistency():
    """Check for potential bugs or inconsistencies in mask generation"""
    print("\n\n=== Consistency Checks ===")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)

    # Test 1: Reproducibility with same seed
    print("\n1. Testing reproducibility with random seed...")
    from PIL import Image
    image = Image.open("tmp/inputs/truck.jpg")
    image = np.array(image.convert("RGB"))

    np.random.seed(42)
    torch.manual_seed(42)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    masks1 = mask_generator.generate(image)

    np.random.seed(42)
    torch.manual_seed(42)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    masks2 = mask_generator.generate(image)

    if len(masks1) == len(masks2):
        print(f"  ✓ Reproducible: Generated {len(masks1)} masks both times")
    else:
        print(f"  ✗ NOT reproducible: {len(masks1)} vs {len(masks2)} masks")

    # Test 2: Validate mask properties
    print("\n2. Validating mask properties...")
    issues_found = []

    for i, mask in enumerate(masks1):
        # Check required fields
        required_fields = ['segmentation', 'area', 'bbox', 'predicted_iou',
                          'stability_score', 'point_coords']
        missing = [f for f in required_fields if f not in mask]
        if missing:
            issues_found.append(f"Mask {i} missing fields: {missing}")

        # Check area consistency
        if 'segmentation' in mask and 'area' in mask:
            actual_area = np.sum(mask['segmentation'])
            reported_area = mask['area']
            if abs(actual_area - reported_area) > 1:
                issues_found.append(
                    f"Mask {i} area mismatch: "
                    f"segmentation={actual_area}, reported={reported_area}"
                )

        # Check IoU and stability scores are in valid range
        if 'predicted_iou' in mask:
            iou = mask['predicted_iou']
            if not (0 <= iou <= 1):
                issues_found.append(f"Mask {i} invalid IoU: {iou}")

        if 'stability_score' in mask:
            stability = mask['stability_score']
            if not (0 <= stability <= 1):
                issues_found.append(f"Mask {i} invalid stability: {stability}")

    if issues_found:
        print(f"  ✗ Found {len(issues_found)} issues:")
        for issue in issues_found[:5]:  # Show first 5
            print(f"    - {issue}")
        if len(issues_found) > 5:
            print(f"    ... and {len(issues_found) - 5} more")
    else:
        print(f"  ✓ All {len(masks1)} masks have valid properties")

    # Test 3: Check threshold filtering
    print("\n3. Testing threshold filtering...")
    mask_generator_strict = SAM2AutomaticMaskGenerator(
        model=sam2,
        pred_iou_thresh=0.95,  # Very strict
        stability_score_thresh=0.98
    )
    masks_strict = mask_generator_strict.generate(image)

    mask_generator_lenient = SAM2AutomaticMaskGenerator(
        model=sam2,
        pred_iou_thresh=0.5,  # Very lenient
        stability_score_thresh=0.8
    )
    masks_lenient = mask_generator_lenient.generate(image)

    print(f"  Strict thresholds: {len(masks_strict)} masks")
    print(f"  Lenient thresholds: {len(masks_lenient)} masks")

    if len(masks_strict) < len(masks_lenient):
        print(f"  ✓ Threshold filtering works correctly")
    else:
        print(f"  ✗ Unexpected: strict >= lenient")

    return issues_found

def analyze_code_comments():
    """Look for TODO, FIXME, or other concerning comments in the implementation"""
    print("\n\n=== Code Comment Analysis ===")
    print("\nSearching for potential issues in implementation...")

    # Check the automatic_mask_generator code
    from sam2 import automatic_mask_generator
    source_file = Path(automatic_mask_generator.__file__)

    if source_file.exists():
        with open(source_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        concerns = []
        for i, line in enumerate(lines, 1):
            lower = line.lower()
            if any(keyword in lower for keyword in ['todo', 'fixme', 'hack', 'bug', 'xxx', 'warning']):
                concerns.append((i, line.strip()))

        if concerns:
            print(f"\n  Found {len(concerns)} lines with potential concerns:")
            for line_no, line in concerns[:10]:
                print(f"    Line {line_no}: {line[:80]}")
        else:
            print("  ✓ No obvious concerns in comments")

    return concerns

def main():
    """Main analysis"""
    print("=" * 70)
    print("SAM2 Implementation Analysis")
    print("Checking for discrepancies with paper and potential issues")
    print("=" * 70)

    # Analyze parameters
    defaults, paper_specs = analyze_default_parameters()

    # Check consistency
    issues = check_mask_generator_consistency()

    # Analyze code comments
    concerns = analyze_code_comments()

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    # Parameter discrepancies
    mismatches = []
    for key, paper_value in paper_specs.items():
        impl_value = defaults.get(key)
        if impl_value != paper_value:
            mismatches.append((key, paper_value, impl_value))

    if mismatches:
        print("\n⚠ Parameter Discrepancies Found:")
        for param, paper_val, impl_val in mismatches:
            print(f"  - {param}: Paper recommends {paper_val}, "
                  f"but default is {impl_val}")
    else:
        print("\n✓ All parameters match paper specifications")

    if issues:
        print(f"\n⚠ {len(issues)} validation issues found in masks")
    else:
        print("\n✓ No validation issues found")

    if concerns:
        print(f"\n⚠ {len(concerns)} concerning code comments found")
    else:
        print("\n✓ No concerning code comments found")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
