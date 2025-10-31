import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from ui_parser import generate_ui_layout as _gen

mcp = FastMCP("sam-mcp-server")

@mcp.tool()
def health_check() -> str:
    ckpt = os.environ.get("SAM_CHECKPOINT_PATH", "(unset)")
    device = os.environ.get("SAM_DEVICE", "cuda")
    model = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    return f"SAM checkpoint: {ckpt}\nModel: {model}\nDevice: {device}\nEasyOCR expected installed"

@mcp.tool()
def generate_ui_layout(
    source_image_path: str,
    new_widget_name: str = "AutoWidget",
    output_directory: Optional[str] = None,
) -> dict:
    """Generate UMG layout JSON + alpha PNG layers + preview from a UI image."""
    return _gen(source_image_path, new_widget_name, output_directory)

if __name__ == "__main__":
    mcp.run()
