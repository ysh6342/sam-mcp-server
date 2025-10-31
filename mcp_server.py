
import asyncio, json, os
from mcp.server import Server
from mcp.types import Tool, ToolRequest, ToolResponse
from ui_parser import generate_ui_layout

server = Server("sam-mcp-server", "1.3.0")

@server.tool(Tool(
    name="health_check",
    description="Check SAM/EasyOCR environment.",
    input_schema={"type":"object","properties":{}}
))
async def health_check(req: ToolRequest) -> ToolResponse:
    ckpt = os.environ.get("SAM_CHECKPOINT_PATH", "(unset)")
    device = os.environ.get("SAM_DEVICE", "cuda")
    model = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    return ToolResponse(content=[{"type":"text","text":f"SAM checkpoint: {ckpt}\nModel: {model}\nDevice: {device}\nEasyOCR: expected installed"}])

@server.tool(Tool(
    name="generate_ui_layout",
    description="Generate UMG layout JSON from a UI image. Saves alpha PNG layers and preview.",
    input_schema={
        "type":"object",
        "required":["source_image_path"],
        "properties":{
            "source_image_path":{"type":"string"},
            "new_widget_name":{"type":"string","default":"AutoWidget"}
        }
    }
))
async def generate_ui_layout_tool(req: ToolRequest) -> ToolResponse:
    args = req.arguments or {}
    result = generate_ui_layout(args.get("source_image_path"), args.get("new_widget_name"))
    return ToolResponse(content=[{"type":"text","text":json.dumps(result, ensure_ascii=False, indent=2)}])

async def main():
    await server.run_stdio()

if __name__ == "__main__":
    asyncio.run(main())
