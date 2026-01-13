# [Paper2Agent](https://github.com/jmiao24/Paper2Agent): SAM 2 Demo

A demonstration of turning the [SAM 2 paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/) into an interactive AI agent. This project transforms Meta's Segment Anything Model 2 (SAM 2) for promptable visual segmentation into a conversational agent that can automatically generate object masks from images through natural language.

## Folder Structure

```
SAM2_demo/
├── mcp/
│   ├── sam2_mcp.py             # MCP server entry point
│   ├── requirements.txt        # Python dependencies
│   └── tools/
│       └── automatic_mask_generator_example.py   # Mask generation tools
└── tmp/
    ├── inputs/                 # Input images directory
    └── outputs/                # Generated masks and visualizations
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jmiao24/SAM2_demo.git
cd SAM2_demo
```

### 2. Install Gemini CLI

Install the [Google Gemini CLI](https://github.com/google-gemini/gemini-cli):

```bash
brew install gemini-cli
```

### 3. Install FastMCP

```bash
pip install fastmcp
```

### 4. Install MCP Server

Install the SAM 2 MCP server using fastmcp:

```bash
fastmcp install gemini-cli ./mcp/sam2_mcp.py --with-requirements ./mcp/requirements.txt
```

### 5. Start the Agent

Start Gemini CLI in the repository folder:

```bash
gemini
```

You will now have access to the SAM 2 agent with all available tools.

## Example Query

```
Segment all objects in the image located at ./tmp/inputs/photo.jpg and save
the mask overlay visualization.
```

## Available Agent Tools

The agent provides the following capabilities through natural language:

### Automatic Mask Generation
- `sam2_generate_masks`: Generate object masks automatically from an image using SAM 2 with default settings
- `sam2_generate_masks_advanced`: Generate object masks with advanced configuration options for fine-tuned control

### Advanced Configuration Options
The advanced tool supports customization of:
- `points_per_side`: Sampling density (higher = more masks)
- `pred_iou_thresh`: IoU threshold for mask quality filtering
- `stability_score_thresh`: Stability score threshold
- `crop_n_layers`: Number of crop layers for multi-scale processing
- `box_nms_thresh`: Non-maximal suppression threshold
- `min_mask_region_area`: Minimum mask region area
- `use_m2m`: Mask-to-mask refinement for improved boundaries

### Output Artifacts
Each tool generates:
- CSV file with mask metadata (area, bounding box, IoU, stability score)
- Original image visualization (PNG)
- Masked overlay visualization (PNG)

## About SAM 2

SAM 2 (Segment Anything Model 2) is a foundation model for promptable visual segmentation in images and videos. Key features include:

- Extends SAM to video by treating images as single-frame videos
- Transformer architecture with streaming memory for real-time processing
- Automatic mask generation without manual prompts
- High-quality segmentation across diverse object types
- Support for various image sizes and aspect ratios

For more details, see the [SAM 2 GitHub repository](https://github.com/facebookresearch/sam2).
