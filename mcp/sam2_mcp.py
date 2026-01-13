"""
Model Context Protocol (MCP) for sam2

SAM 2 (Segment Anything Model 2) is a foundation model for promptable visual segmentation in images and videos. It extends SAM to video by treating images as single-frame videos, using a transformer architecture with streaming memory for real-time processing. This MCP server provides tools for automatic mask generation from images with both default and advanced configuration options.

This MCP Server contains tools extracted from the following tutorial files:
1. automatic_mask_generator_example
    - sam2_generate_masks: Generate object masks automatically from an image using SAM 2
    - sam2_generate_masks_advanced: Generate object masks with advanced configuration options
"""

from fastmcp import FastMCP

# Import statements (alphabetical order)
from tools.automatic_mask_generator_example import automatic_mask_generator_example_mcp

# Server definition and mounting
mcp = FastMCP(name="sam2")
mcp.mount(automatic_mask_generator_example_mcp)

if __name__ == "__main__":
    mcp.run()
