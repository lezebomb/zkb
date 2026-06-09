# OCR Model Setup

Prepare PaddleOCR locally before competition:

```bash
python scripts/prepare_ocr_model.py --output models/ocr/paddleocr --allow-download
```

Expected directories:

```text
models/ocr/paddleocr/det
models/ocr/paddleocr/rec
models/ocr/paddleocr/cls
```

Competition env:

```env
OCR_BACKEND=paddleocr
MODEL_OCR_PATH=models/ocr/paddleocr
ALLOW_MODEL_AUTO_DOWNLOAD=false
```

Confirm `/debug/status` reports OCR model dirs ready.

