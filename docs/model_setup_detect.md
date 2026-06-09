# Detect Model Setup

Prepare YOLO weights before competition:

```bash
python scripts/prepare_detect_model.py --model yolo11n.pt --output models/detect/yolo11n.pt --allow-download
```

Then set:

```env
DETECT_BACKEND=ultralytics
MODEL_DETECT_PATH=models/detect/yolo11n.pt
ALLOW_MODEL_AUTO_DOWNLOAD=false
```

Confirm `/debug/status` reports `model_exists=true`. COCO does not cover `台灯`; add custom lamp data or use a second-stage strategy.

