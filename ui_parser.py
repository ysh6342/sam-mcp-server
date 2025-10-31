import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
import easyocr
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

logger = logging.getLogger(__name__)

_SAM_GENERATOR_LOCK = threading.Lock()
_SAM_GENERATOR: Optional[SamAutomaticMaskGenerator] = None
_SAM_CONFIG: Optional[Dict[str, str]] = None

_OCR_READER_LOCK = threading.Lock()
_OCR_READER: Optional[easyocr.Reader] = None


def get_ocr_reader() -> easyocr.Reader:
    """Load (or return a cached) EasyOCR reader."""
    global _OCR_READER
    with _OCR_READER_LOCK:
        if _OCR_READER is None:
            logger.info("Loading EasyOCR reader for [en, ko]")
            _OCR_READER = easyocr.Reader(["en", "ko"])
    return _OCR_READER


MIN_OCR_AREA_PX = 400  # skip OCR on extremely small segments to save time


def _resolve_sam_device() -> str:
    override = os.environ.get("SAM_DEVICE")
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def reset_sam_model_cache() -> None:
    """Clear the memoized SAM mask generator (useful for tests)."""
    global _SAM_GENERATOR, _SAM_CONFIG
    with _SAM_GENERATOR_LOCK:
        _SAM_GENERATOR = None
        _SAM_CONFIG = None


def load_sam_model(force_reload: bool = False) -> SamAutomaticMaskGenerator:
    """Load (or return a cached) SAM automatic mask generator."""
    global _SAM_GENERATOR, _SAM_CONFIG

    ckpt = os.environ.get("SAM_CHECKPOINT_PATH")
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}")

    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    device = _resolve_sam_device()

    with _SAM_GENERATOR_LOCK:
        config = {"ckpt": ckpt, "model_type": model_type, "device": device}
        if force_reload or _SAM_GENERATOR is None or _SAM_CONFIG != config:
            try:
                sam_cls = sam_model_registry[model_type]
            except KeyError as exc:
                raise ValueError(f"Unsupported SAM model type: {model_type}") from exc
            logger.info("Loading SAM model '%s' from %s on %s", model_type, ckpt, device)
            sam = sam_cls(checkpoint=ckpt)
            sam.to(device=device)
            _SAM_GENERATOR = SamAutomaticMaskGenerator(sam)
            _SAM_CONFIG = config
    return _SAM_GENERATOR  # type: ignore[return-value]

def _avg_text_color(region_bgr, quad):
    # quad: ((x1,y1),(x2,y2),(x3,y3),(x4,y4))
    pts = np.array(quad, dtype=np.int32)
    mask = np.zeros(region_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    sel = region_bgr[mask==255]
    if sel.size == 0: return (255,255,255)
    mean = sel.mean(axis=0)
    return (float(mean[2])/255.0, float(mean[1])/255.0, float(mean[0])/255.0, 1.0)  # RGBA

def _has_round_handle(region_gray):
    circles = cv2.HoughCircles(region_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12,
                               param1=100, param2=20, minRadius=6, maxRadius=max(8, int(min(region_gray.shape)/6)))
    return circles is not None

def _has_checkbox_mark(region_gray):
    edges = cv2.Canny(region_gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=40)
    return lines is not None

def classify_widget(region_bgr, ocr_texts):
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    std_int = float(np.std(gray))
    aspect = w / max(h, 1)
    has_text = len(ocr_texts) > 0

    if has_text and h < 200:
        return "TextBlock"
    if w > 200 and h < 60 and aspect > 3.5 and std_int < 25:
        return "ProgressBar"
    if w > 200 and h < 80 and aspect > 4.0 and _has_round_handle(gray):
        return "Slider"
    if 14 <= min(w, h) <= 64 and 0.75 <= aspect <= 1.25 and _has_checkbox_mark(gray):
        return "CheckBox"
    if has_text and 0.2 < aspect < 8 and std_int < 50:
        return "EditableTextBox"
    if (1.2 <= aspect <= 5.0 and 24 <= h <= 180) and std_int < 45:
        return "Button"
    if std_int < 5 and (w > 40 and h > 40):
        return "Border"
    return "Image"

def _nest_children(widgets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    taken = set()
    by_idx = list(enumerate(widgets))
    for i, wi in by_idx:
        if wi["WidgetType"] == "Button":
            bx = wi["Slot"]["Position"]["X"]; by = wi["Slot"]["Position"]["Y"]
            bw = wi["Slot"]["Size"]["Width"]; bh = wi["Slot"]["Size"]["Height"]
            for j, wj in by_idx:
                if j == i or j in taken: continue
                if wj["WidgetType"] in ("TextBlock","Image"):
                    tx = wj["Slot"]["Position"]["X"]; ty = wj["Slot"]["Position"]["Y"]
                    tw = wj["Slot"]["Size"]["Width"]; th = wj["Slot"]["Size"]["Height"]
                    if tx >= bx and ty >= by and (tx+tw) <= (bx+bw) and (ty+th) <= (by+bh):
                        wi["Children"].append(wj)
                        taken.add(j)
    return [w for idx, w in enumerate(widgets) if idx not in taken]

def _prepare_output_dir(source_image_path: str, output_directory: Optional[str]) -> str:
    if output_directory:
        outdir = os.path.abspath(output_directory)
    else:
        outdir = os.path.join(
            os.path.dirname(source_image_path),
            f"{os.path.splitext(os.path.basename(source_image_path))[0]}_layers_{uuid.uuid4().hex[:8]}",
        )
    os.makedirs(outdir, exist_ok=True)
    return outdir


def generate_ui_layout(
    source_image_path: str,
    new_widget_name: str = "AutoWidget",
    output_directory: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.exists(source_image_path):
        raise FileNotFoundError(f"Image not found: {source_image_path}")
    img = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {source_image_path}")
    H, W = img.shape[:2]

    sam_gen = load_sam_model()
    try:
        masks = sam_gen.generate(img)
    except Exception as exc:
        logger.exception("SAM mask generation failed for %s", source_image_path)
        raise RuntimeError("SAM mask generation failed") from exc

    outdir = _prepare_output_dir(source_image_path, output_directory)

    # preview overlay
    preview = img.copy()
    for m in masks:
        seg = (m["segmentation"].astype(np.uint8))*255
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0,255,0), 2)
    cv2.imwrite(os.path.join(outdir, "preview.png"), preview)

    raw_widgets = []
    for i, mask in enumerate(masks):
        seg = (mask["segmentation"].astype(np.uint8))*255
        x,y,w,h = cv2.boundingRect(seg)
        region_bgr = img[y:y+h, x:x+w].copy()
        alpha_full = seg[y:y+h, x:x+w]
        region_bgra = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2BGRA)
        region_bgra[:, :, 3] = alpha_full
        layer_path = os.path.join(outdir, f"layer_{i:03d}.png")
        cv2.imwrite(layer_path, region_bgra)

        # OCR with mask whitening
        visible = region_bgr.copy()
        visible[alpha_full < 128] = 255
        area = int(mask.get("area", int(np.count_nonzero(alpha_full))))
        ocr_res = []
        if area >= MIN_OCR_AREA_PX:
            try:
                ocr_res = get_ocr_reader().readtext(visible, paragraph=False)
            except Exception as exc:
                logger.warning("OCR failed for mask %s: %s", i, exc)
        texts = [r[1] for r in ocr_res if r[2] > 0.6]
        colors = None
        if ocr_res:
            # estimate color using first bbox
            quad = ocr_res[0][0]
            colors = _avg_text_color(region_bgr, quad)  # RGBA 0..1

        wtype = classify_widget(region_bgr, texts)
        node = {
            "WidgetType": wtype,
            "WidgetName": f"{wtype}_{i:03d}",
            "Slot": {"Position":{"X":int(x),"Y":int(y)},"Size":{"Width":int(w),"Height":int(h)}},
            "ImagePath": layer_path.replace('\\','/'),
            "Text": texts[0] if texts else "",
            "TextColor": {"R":colors[0],"G":colors[1],"B":colors[2],"A":colors[3]} if colors else None,
            "Children":[]
        }
        raw_widgets.append(node)

    widgets = _nest_children(raw_widgets)

    layout = {
        "GeneratedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "SourceImage": os.path.abspath(source_image_path),
        "CanvasSize": {"Width": W, "Height": H},
        "WidgetTreeRoot": [{
            "WidgetType": "CanvasPanel",
            "WidgetName": new_widget_name,
            "Children": widgets
        }]
    }
    json_path = os.path.join(outdir, "layout.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, ensure_ascii=False, indent=2)
    return {"layout_json_path": json_path, "layer_directory_path": outdir, "widget_count": len(widgets)}
