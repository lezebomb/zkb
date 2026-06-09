# Train Classify

Prepare ImageFolder template:

```bash
python scripts/prepare_classify_dataset.py --mode template --output data/processed/classify/contest8
```

Train after adding real images:

```bash
python scripts/train_classify.py --data data/processed/classify/contest8 --model mobilenet_v3_small --epochs 1 --batch 4 --imgsz 224 --device cpu --output models/classify/classifier.pt
```

If data is empty, the script can create a tiny placeholder for smoke tests only. It is not useful for scoring.

