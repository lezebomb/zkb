# Train Detect

Smoke plan:

```bash
python scripts/prepare_detect_dataset.py --mode coco8 --output data/processed/detect/coco8_contest
python scripts/train_detect_yolo.py --model models/detect/yolo11n.pt --data data/processed/detect/coco8_contest/data.yaml --epochs 1 --imgsz 320 --batch 2 --device cpu --project runs/detect --name smoke
```

Full runs require `RUN_FULL_TRAIN=true` or `--allow-long-run`. Dataset downloads require `ALLOW_DATASET_DOWNLOAD=true` or `--allow-download`.

After training, best weights are copied to `models/detect/best.pt`. Set `DETECT_BACKEND=ultralytics` and `MODEL_DETECT_PATH=models/detect/best.pt`.

