"""
Automatic mask generation with SAM 2.

This MCP Server provides 2 tools:
1. sam2_generate_masks: Generate object masks automatically from an image using SAM 2
2. sam2_generate_masks_advanced: Generate object masks with advanced configuration options

All tools extracted from facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("AUTOMATIC_MASK_GENERATOR_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("AUTOMATIC_MASK_GENERATOR_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
automatic_mask_generator_example_mcp = FastMCP(name="automatic_mask_generator_example")


def show_anns(anns, borders=True):
    """Helper function to visualize masks."""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


@automatic_mask_generator_example_mcp.tool
def sam2_generate_masks(
    image_path: Annotated[str | None,
        "Path to input image file (JPEG, PNG, or other common image formats). "
        "Image will be automatically loaded and converted to RGB format. "
        "Supports various image sizes and aspect ratios."] = None,

    sam2_checkpoint: Annotated[str,
        "Path to SAM 2 model checkpoint file (.pt file). "
        "Example: 'checkpoints/sam2.1_hiera_large.pt'. "
        "Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/"] = "checkpoints/sam2.1_hiera_large.pt",

    model_cfg: Annotated[str,
        "Path to SAM 2 model configuration YAML file. "
        "Example: 'configs/sam2.1/sam2.1_hiera_l.yaml'. "
        "Configuration must match the checkpoint model architecture."] = "configs/sam2.1/sam2.1_hiera_l.yaml",

    seed: Annotated[int,
        "Random seed for reproducibility of mask visualization colors. "
        "Default is 42 to match tutorial outputs."] = 42,

    out_prefix: Annotated[str | None,
        "Prefix for output files. If None, uses 'sam2_masks_TIMESTAMP'. "
        "Output includes: CSV with mask metadata, PNG with input image, PNG with masked overlay."] = None,
) -> dict:
    """
    Generate object masks automatically from an image using SAM 2 with default settings.
    Input is an RGB image file and output is mask metadata CSV and visualization PNGs showing the original image and masked overlay.
    """
    # Input file validation
    if image_path is None:
        raise ValueError("Path to input image file must be provided")

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    checkpoint_file = Path(sam2_checkpoint)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {sam2_checkpoint}")

    # Note: model_cfg is a Hydra config path (e.g., "configs/sam2.1/sam2.1_hiera_l.yaml")
    # It is resolved by Hydra's config search path, not as a file path
    # Therefore, we don't validate it as a file path here

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup output prefix
    if out_prefix is None:
        out_prefix = f"sam2_masks_{timestamp}"

    output_base = OUTPUT_DIR / out_prefix

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load image
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # Save original image visualization
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    original_img_path = f"{output_base}_original.png"
    plt.savefig(original_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Build SAM 2 model
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    # Generate masks
    masks = mask_generator.generate(image)

    # Save mask metadata
    mask_data = []
    for i, mask in enumerate(masks):
        mask_data.append({
            'mask_id': i,
            'area': mask['area'],
            'bbox_x': mask['bbox'][0],
            'bbox_y': mask['bbox'][1],
            'bbox_w': mask['bbox'][2],
            'bbox_h': mask['bbox'][3],
            'predicted_iou': mask['predicted_iou'],
            'stability_score': mask['stability_score'],
            'point_coords_x': mask['point_coords'][0][0],
            'point_coords_y': mask['point_coords'][0][1],
            'crop_box_x': mask['crop_box'][0],
            'crop_box_y': mask['crop_box'][1],
            'crop_box_w': mask['crop_box'][2],
            'crop_box_h': mask['crop_box'][3],
        })

    mask_df = pd.DataFrame(mask_data)
    csv_path = f"{output_base}_metadata.csv"
    mask_df.to_csv(csv_path, index=False)

    # Visualize masks
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    masked_img_path = f"{output_base}_overlay.png"
    plt.savefig(masked_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "message": f"Generated {len(masks)} masks from image using SAM 2",
        "reference": "https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb",
        "artifacts": [
            {
                "description": "Mask metadata CSV",
                "path": str(Path(csv_path).resolve())
            },
            {
                "description": "Original image",
                "path": str(Path(original_img_path).resolve())
            },
            {
                "description": "Masked overlay visualization",
                "path": str(Path(masked_img_path).resolve())
            }
        ]
    }


@automatic_mask_generator_example_mcp.tool
def sam2_generate_masks_advanced(
    image_path: Annotated[str | None,
        "Path to input image file (JPEG, PNG, or other common image formats). "
        "Image will be automatically loaded and converted to RGB format. "
        "Supports various image sizes and aspect ratios."] = None,

    sam2_checkpoint: Annotated[str,
        "Path to SAM 2 model checkpoint file (.pt file). "
        "Example: 'checkpoints/sam2.1_hiera_large.pt'. "
        "Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/"] = "checkpoints/sam2.1_hiera_large.pt",

    model_cfg: Annotated[str,
        "Path to SAM 2 model configuration YAML file. "
        "Example: 'configs/sam2.1/sam2.1_hiera_l.yaml'. "
        "Configuration must match the checkpoint model architecture."] = "configs/sam2.1/sam2.1_hiera_l.yaml",

    points_per_side: Annotated[int,
        "Number of points to sample per side of the image. "
        "Higher values (e.g., 64) sample more densely and generate more masks. "
        "Tutorial uses 64 for improved detail."] = 64,

    points_per_batch: Annotated[int,
        "Number of points to process in parallel in one batch. "
        "Higher values use more memory but can be faster. Tutorial uses 128."] = 128,

    pred_iou_thresh: Annotated[float,
        "Threshold for predicted IoU (Intersection over Union) to filter low-quality masks. "
        "Range: 0.0-1.0. Higher values (e.g., 0.7) are more selective. Tutorial uses 0.7."] = 0.7,

    stability_score_thresh: Annotated[float,
        "Stability score threshold for mask quality filtering. "
        "Range: 0.0-1.0. Higher values require more stable masks. Tutorial uses 0.92."] = 0.92,

    stability_score_offset: Annotated[float,
        "Offset value added to stability score calculation. "
        "Affects the sensitivity of stability filtering. Tutorial uses 0.7."] = 0.7,

    crop_n_layers: Annotated[int,
        "Number of crop layers to use for generating masks on image crops. "
        "0 means no crops, higher values process multiple crop scales. Tutorial uses 1."] = 1,

    box_nms_thresh: Annotated[float,
        "Non-maximal suppression (NMS) threshold for removing duplicate masks. "
        "Range: 0.0-1.0. Lower values are more aggressive at removing duplicates. Tutorial uses 0.7."] = 0.7,

    crop_n_points_downscale_factor: Annotated[int,
        "Factor to downscale the number of points sampled in each crop layer. "
        "Tutorial uses 2, meaning each crop layer has half the points of the previous."] = 2,

    min_mask_region_area: Annotated[float,
        "Minimum area (in pixels) for a mask region to be retained. "
        "Removes small disconnected regions and noise. Tutorial uses 25.0."] = 25.0,

    use_m2m: Annotated[bool,
        "Whether to use mask-to-mask refinement for improved quality. "
        "Tutorial uses True for better mask boundaries."] = True,

    seed: Annotated[int,
        "Random seed for reproducibility of mask visualization colors. "
        "Default is 42 to match tutorial outputs."] = 42,

    out_prefix: Annotated[str | None,
        "Prefix for output files. If None, uses 'sam2_masks_advanced_TIMESTAMP'. "
        "Output includes: CSV with mask metadata, PNG with input image, PNG with masked overlay."] = None,
) -> dict:
    """
    Generate object masks with advanced configuration options for densely sampling and quality filtering.
    Input is an RGB image file and output is mask metadata CSV and visualization PNGs showing the original image and masked overlay with improved detail.
    """
    # Input file validation
    if image_path is None:
        raise ValueError("Path to input image file must be provided")

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    checkpoint_file = Path(sam2_checkpoint)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"SAM 2 checkpoint not found: {sam2_checkpoint}")

    # Note: model_cfg is a Hydra config path (e.g., "configs/sam2.1/sam2.1_hiera_l.yaml")
    # It is resolved by Hydra's config search path, not as a file path
    # Therefore, we don't validate it as a file path here

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup output prefix
    if out_prefix is None:
        out_prefix = f"sam2_masks_advanced_{timestamp}"

    output_base = OUTPUT_DIR / out_prefix

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load image
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # Save original image visualization
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    original_img_path = f"{output_base}_original.png"
    plt.savefig(original_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Build SAM 2 model with advanced configuration
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_generator_2 = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        box_nms_thresh=box_nms_thresh,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        use_m2m=use_m2m,
    )

    # Generate masks
    masks2 = mask_generator_2.generate(image)

    # Save mask metadata
    mask_data = []
    for i, mask in enumerate(masks2):
        mask_data.append({
            'mask_id': i,
            'area': mask['area'],
            'bbox_x': mask['bbox'][0],
            'bbox_y': mask['bbox'][1],
            'bbox_w': mask['bbox'][2],
            'bbox_h': mask['bbox'][3],
            'predicted_iou': mask['predicted_iou'],
            'stability_score': mask['stability_score'],
            'point_coords_x': mask['point_coords'][0][0],
            'point_coords_y': mask['point_coords'][0][1],
            'crop_box_x': mask['crop_box'][0],
            'crop_box_y': mask['crop_box'][1],
            'crop_box_w': mask['crop_box'][2],
            'crop_box_h': mask['crop_box'][3],
        })

    mask_df = pd.DataFrame(mask_data)
    csv_path = f"{output_base}_metadata.csv"
    mask_df.to_csv(csv_path, index=False)

    # Visualize masks
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    masked_img_path = f"{output_base}_overlay.png"
    plt.savefig(masked_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "message": f"Generated {len(masks2)} masks from image using SAM 2 advanced configuration",
        "reference": "https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb",
        "artifacts": [
            {
                "description": "Mask metadata CSV",
                "path": str(Path(csv_path).resolve())
            },
            {
                "description": "Original image",
                "path": str(Path(original_img_path).resolve())
            },
            {
                "description": "Masked overlay visualization",
                "path": str(Path(masked_img_path).resolve())
            }
        ]
    }
