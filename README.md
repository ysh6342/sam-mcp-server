# SAM MCP Server & Unreal UMG Builder

This repository pairs a Claude MCP adapter with UI parsing utilities powered by Segment Anything and EasyOCR. It turns UI screenshots into structured layout data and rebuilds Unreal Engine 5.7 UMG widgets from the generated JSON.

## Features
- `generate_ui_layout` analyzes a UI image and emits alpha PNG layers, a contour preview, and a `layout.json` widget tree.
- Per-layer OCR helps classify buttons, sliders, check boxes, editable text boxes, and other common widgets.
- `build_widget_from_json.py` recreates a UserWidget blueprint inside Unreal and imports the generated layer images as Texture2D assets.

## Requirements
- Python 3.9-3.11
- CUDA-capable GPU (recommended but optional)
- Unreal Engine 5.7 for the reconstruction workflow
- A Segment Anything checkpoint (`.pth`) stored outside the repository

## Setup
```bash
python -m venv .venv
. .venv/bin/activate             # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision easyocr opencv-python
pip install -e segment-anything
```

## Environment Variables
`ui_parser.py` and `mcp_server.py` rely on:

- `SAM_CHECKPOINT_PATH` - absolute path to the SAM checkpoint, for example `D:/models/sam_vit_h_4b8939.pth`.
- `SAM_MODEL_TYPE` - model name such as `vit_h`, `vit_l`, or `vit_b`.
- `SAM_DEVICE` - `cuda` or `cpu`. Defaults to CUDA if available, otherwise CPU.

Export these variables in your shell (or load them from a `.env`). Do not commit checkpoints.

## Running the MCP Server
```bash
python mcp_server.py
```

From Claude or any MCP client call:

- `source_image_path`: path to the UI screenshot (BGRA/RGBA PNG recommended).
- `new_widget_name`: optional name that becomes the root widget when rebuilding in Unreal. Defaults to `AutoWidget`.

The tool creates a sibling directory named `*_layers_<timestamp>/` next to the source image. It contains:

- `preview.png` - the original image with segmentation contours.
- `layout.json` - widget tree metadata and canvas size.
- `layer_XXX.png` - alpha-preserving PNG layers for individual widgets.

Keep these artifacts out of Git history; they are meant for inspection only.

## Rebuilding in Unreal
Run inside the Unreal Python environment:

```python
from build_umg_from_json import build_widget_from_json
build_widget_from_json(r"D:/ui/main_menu_layers_ab12cd34/layout.json", "AutoWidget")
```

Prerequisites and behaviors:
- `BASE_WIDGET_BP` must resolve to an existing UserWidget blueprint that serves as a template.
- The script duplicates that base asset under `TARGET_FOLDER` and overwrites any widget with the same name.
- Layer PNGs are imported into `/Game/UI/Generated` as Texture2D assets and automatically wired to Image widgets.

## Development Notes
- Run `bash segment-anything/linter.sh` before committing to apply isort/black and run flake8 plus mypy.
- Validate UI output by inspecting the generated `layout.json`, `preview.png`, and PNG layers or by adding `pytest` coverage for `generate_ui_layout`.
- Adjust `BASE_WIDGET_BP` and `TARGET_FOLDER` inside `build_umg_from_json.py` to match your Unreal project structure.
