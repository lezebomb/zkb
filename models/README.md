# Models

Place local model artifacts here before competition. Do not commit large weights.

Expected paths:

- `models/detect/yolo11n.pt` or `models/detect/best.pt`
- `models/classify/classifier.pt`
- `models/ocr/paddleocr/det`, `rec`, `cls`

Before competition, keep `ALLOW_MODEL_AUTO_DOWNLOAD=false` and confirm `/debug/status` reports model files/directories ready.

