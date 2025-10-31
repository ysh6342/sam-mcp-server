import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import ui_parser


class DummyMaskGenerator:
    def __init__(self, masks):
        self._masks = masks

    def generate(self, image):
        return self._masks


class DummyReader:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def readtext(self, visible, paragraph=False):
        self.calls += 1
        response = self.responses
        if isinstance(response, Exception):
            raise response
        return response


def _write_dummy_image(path: Path, size: int = 16) -> None:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture(autouse=True)
def _reset_sam_cache():
    ui_parser.reset_sam_model_cache()
    yield
    ui_parser.reset_sam_model_cache()


def test_generate_ui_layout_creates_outputs(tmp_path, monkeypatch):
    image_path = tmp_path / "input.png"
    _write_dummy_image(image_path)

    seg = np.zeros((16, 16), dtype=bool)
    seg[2:10, 2:10] = True
    masks = [{"segmentation": seg, "area": int(seg.sum())}]

    monkeypatch.setattr(ui_parser, "load_sam_model", lambda: DummyMaskGenerator(masks))
    dummy_reader = DummyReader(
        [([(0, 0), (1, 0), (1, 1), (0, 1)], "Start", 0.7)]
    )
    monkeypatch.setattr(ui_parser, "reader", dummy_reader)

    output_dir = tmp_path / "output"
    result = ui_parser.generate_ui_layout(
        str(image_path),
        new_widget_name="TestWidget",
        output_directory=str(output_dir),
    )

    layout_path = Path(result["layout_json_path"])
    layer_dir = Path(result["layer_directory_path"])

    assert layout_path.exists()
    assert layer_dir.exists()
    assert any(layer_dir.glob("layer_*.png"))
    assert (layer_dir / "preview.png").exists()
    assert dummy_reader.calls == 1

    data = json.loads(layout_path.read_text(encoding="utf-8"))
    assert data["WidgetTreeRoot"][0]["WidgetName"] == "TestWidget"
    assert data["CanvasSize"] == {"Width": 16, "Height": 16}


def test_generate_ui_layout_skips_small_masks_for_ocr(tmp_path, monkeypatch):
    image_path = tmp_path / "input_small.png"
    _write_dummy_image(image_path, size=8)

    seg = np.zeros((8, 8), dtype=bool)
    seg[0, 0] = True  # area == 1
    masks = [{"segmentation": seg, "area": 1}]

    monkeypatch.setattr(ui_parser, "load_sam_model", lambda: DummyMaskGenerator(masks))
    dummy_reader = DummyReader(
        [([(0, 0), (1, 0), (1, 1), (0, 1)], "SkipMe", 0.8)]
    )
    monkeypatch.setattr(ui_parser, "reader", dummy_reader)

    output_dir = tmp_path / "output_small"
    result = ui_parser.generate_ui_layout(
        str(image_path),
        new_widget_name="SmallWidget",
        output_directory=str(output_dir),
    )

    assert os.path.isdir(result["layer_directory_path"])
    # OCR should not have been invoked for very small masks.
    assert dummy_reader.calls == 0
