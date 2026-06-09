# Train OCR

Generate synthetic data:

```bash
python scripts/generate_synthetic_ocr_dataset.py --output data/processed/ocr/synthetic --count 500 --seed 42
```

Dry-run PaddleOCR wrapper:

```bash
python scripts/train_ocr_paddle.py --config configs/ocr/paddleocr_rec_template.yml --data data/processed/ocr/synthetic --dry-run
```

Only launch real PaddleOCR training after installing `requirements-ocr.txt`, preparing local data, and setting `RUN_FULL_TRAIN=true` or `--allow-run`.

