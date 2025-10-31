
# SAM MCP Server + UMG Builder (UE 5.7, Visual Defaults)

- SAM + EasyOCR: UI → 위젯 레이어(BGRA) + `layout.json` + `preview.png`
- 버튼 내부 텍스트/이미지 자식화
- UE 5.7 파이썬 스크립트: TextBlock 정렬/폰트/색, ProgressBar/Slider/CheckBox 기본값 적용

## 설치
```
pip install torch torchvision easyocr opencv-python
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Claude MCP
```
python mcp_server.py
```
Claude에서 `generate_ui_layout` 호출 후, UE 에디터에서 `build_umg_from_json.py` 실행:
```python
from build_umg_from_json import build_widget_from_json
build_widget_from_json(r"D:/ui/main_menu_layers_xxxxxxxx/layout.json", "AutoWidget")
```

## 참고
- `BASE_WIDGET_BP`는 프로젝트에 존재하는 최소 UserWidget 경로로 변경하세요.
- Texture2D는 `/Game/UI/Generated`에 자동 임포트되어 Image Brush에 연결됩니다.
