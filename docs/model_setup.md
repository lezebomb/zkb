# Model Setup

Base service only:

```bash
pip install -r requirements.txt
```

Enable local YOLO detect:

```bash
pip install -r requirements-detect.txt
```

Enable local PaddleOCR:

```bash
pip install -r requirements-ocr.txt
```

Install everything:

```bash
pip install -r requirements-all.txt
```

Pre-race model preparation:

1. Put YOLO weights at `models/yolo11n.pt`.
2. Put PaddleOCR model dirs at:
   - `models/paddleocr/det`
   - `models/paddleocr/rec`
   - `models/paddleocr/cls`
3. Keep `ALLOW_MODEL_AUTO_DOWNLOAD=false` for the actual contest.

Optional helper:

```bash
bash scripts/download_models_optional.sh
```

Only set download URLs before the race. Do not auto-download during the official timed evaluation.
