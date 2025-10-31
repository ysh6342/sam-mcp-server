
# UE 5.7 - Build UMG from layout.json with visual defaults
import unreal, json, os, re

BASE_WIDGET_BP = "/Game/UI/BaseUserWidget.BaseUserWidget"
TARGET_FOLDER = "/Game/UI"

WIDGET_CLASS_MAP = {
    "CanvasPanel": unreal.CanvasPanel,
    "Image": unreal.Image,
    "TextBlock": unreal.TextBlock,
    "Button": unreal.Button,
    "ProgressBar": unreal.ProgressBar,
    "CheckBox": unreal.CheckBox,
    "EditableTextBox": unreal.EditableTextBox,
    "Slider": unreal.Slider,
    "Border": unreal.Border,
    "Spacer": unreal.Spacer
}

def _sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', name)[:64] or "Widget"

def _add_to_canvas(tree, parent_canvas, child_widget, slot):
    slot_obj = tree.add_child_to_canvas(parent_canvas, child_widget)
    try:
        layout = slot.get("Position", {}); size = slot.get("Size", {})
        slot_obj.set_editor_property("position", unreal.Vector2D(float(layout.get("X",0)), float(layout.get("Y",0))))
        slot_obj.set_editor_property("size", unreal.Vector2D(float(size.get("Width",64)), float(size.get("Height",32))))
        slot_obj.set_editor_property("z_order", 0)
    except Exception as e:
        unreal.log_warning(f"Slot set failed: {e}")
    return slot_obj

def _to_linear_color(rgba):
    if not rgba: return unreal.LinearColor(1,1,1,1)
    return unreal.LinearColor(float(rgba["R"]), float(rgba["G"]), float(rgba["B"]), float(rgba["A"]))

def _ensure_font_roboto():
    obj = unreal.load_object(None, "/Engine/EngineFonts/Roboto.Roboto")
    return obj

def _apply_props(widget, node):
    wtype = node.get("WidgetType")

    # Text widgets
    if isinstance(widget, unreal.TextBlock) or isinstance(widget, unreal.EditableTextBox):
        txt = node.get("Text", "")
        try:
            widget.set_editor_property("text", unreal.Text(txt))
        except Exception: pass
        # Justification Center
        if isinstance(widget, unreal.TextBlock):
            try:
                widget.set_editor_property("justification", unreal.TextJustify.CENTER)
            except Exception: pass
        # Color
        color = _to_linear_color(node.get("TextColor")) if node.get("TextColor") else unreal.LinearColor(1,1,1,1)
        try:
            if isinstance(widget, unreal.TextBlock):
                widget.set_editor_property("color_and_opacity", color)
        except Exception: pass
        # Font Roboto
        try:
            font_obj = _ensure_font_roboto()
            if font_obj:
                font = widget.get_editor_property("font")
                font.set_editor_property("font_object", font_obj)
                widget.set_editor_property("font", font)
        except Exception: pass

    # Image brush
    img_path = node.get("ImagePath")
    if img_path and os.path.exists(img_path):
        dest_path = f"{TARGET_FOLDER}/Generated"
        tx_name = _sanitize_name(os.path.splitext(os.path.basename(img_path))[0])
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", img_path)
        task.set_editor_property("destination_path", dest_path)
        task.set_editor_property("destination_name", tx_name)
        task.set_editor_property("automated", True)
        task.set_editor_property("save", True)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        tex = unreal.load_object(None, f"{dest_path}/{tx_name}.{tx_name}")
        if tex and isinstance(widget, unreal.Image):
            brush = widget.get_editor_property("brush")
            brush.set_editor_property("resource_object", tex)
            widget.set_editor_property("brush", brush)

    # Button defaults
    if isinstance(widget, unreal.Button):
        try:
            widget.set_editor_property("is_focusable", True)
        except Exception: pass

    # ProgressBar defaults
    if isinstance(widget, unreal.ProgressBar):
        try:
            widget.set_editor_property("percent", 1.0)
        except Exception: pass

    # Slider defaults
    if isinstance(widget, unreal.Slider):
        try:
            widget.set_editor_property("value", 0.5)
        except Exception: pass

    # CheckBox defaults
    if isinstance(widget, unreal.CheckBox):
        try:
            widget.set_editor_property("is_checked", False)
        except Exception: pass

def _build_tree_rec(tree, parent, node):
    cls = WIDGET_CLASS_MAP.get(node.get("WidgetType","Image"), unreal.Image)
    child = tree.construct_widget(cls, _sanitize_name(node.get("WidgetName","Widget")))
    if isinstance(parent, unreal.CanvasPanel):
        _add_to_canvas(tree, parent, child, node.get("Slot", {}))
    else:
        tree.add_child(parent, child)
    _apply_props(child, node)
    for ch in node.get("Children", []):
        _build_tree_rec(tree, child, ch)

def build_widget_from_json(json_path, output_widget_name="AutoWidget"):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    root = data["WidgetTreeRoot"][0]
    out_name = _sanitize_name(output_widget_name)
    dst = f"{TARGET_FOLDER}/{out_name}"
    if not unreal.EditorAssetLibrary.does_asset_exist(BASE_WIDGET_BP):
        raise Exception(f"Base widget blueprint not found: {BASE_WIDGET_BP}")
    if unreal.EditorAssetLibrary.does_asset_exist(dst+"."+out_name):
        unreal.EditorAssetLibrary.delete_asset(dst)
    new_asset = unreal.EditorAssetLibrary.duplicate_asset(BASE_WIDGET_BP, dst)
    if not new_asset:
        raise Exception("Failed to duplicate base widget")
    widget_bp = unreal.load_object(None, f"{dst}.{out_name}")
    tree = widget_bp.widget_tree
    canvas = tree.root_widget
    if not isinstance(canvas, unreal.CanvasPanel):
        canvas = tree.construct_widget(unreal.CanvasPanel, "RootCanvas")
        tree.set_root_widget(canvas)
    for node in root.get("Children", []):
        _build_tree_rec(tree, canvas, node)
    unreal.EditorAssetLibrary.save_loaded_asset(widget_bp)
    unreal.log(f"✅ Built UMG: {dst} from {json_path}")
    return dst
