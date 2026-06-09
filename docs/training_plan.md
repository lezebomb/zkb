# Training Plan

1. Register candidate datasets in `data/dataset_manifest.yaml` before downloading.
2. Run detect COCO8 smoke first, then filtered COCO/custom data.
3. Build classify ImageFolder data for the 8 scene labels.
4. Generate synthetic OCR readings first; fine-tune PaddleOCR only after local setup works.
5. Export artifacts with `scripts/export_best_models.py` and verify `/debug/status`.

Default switches keep downloads and long training disabled.

