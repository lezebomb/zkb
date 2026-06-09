# 中控杯本地算法服务

离线优先的比赛算法服务，提供 `GET /health`、`POST /infer`、`GET /debug/status`。默认不调用任何云推理 API，模型缺失时降级 fallback，服务仍可启动。

## 安装

```bash
pip install -r requirements.txt
```

可选依赖：

- `requirements-detect.txt`: Ultralytics YOLO detect backend / training
- `requirements-ocr.txt`: PaddleOCR backend / model preparation
- `requirements-train.txt`: PyTorch classify training and smoke tooling
- `requirements-all.txt`: all optional stacks

## 启动与接口

```bash
uvicorn contest_agent.app:app --host 0.0.0.0 --port 8080
```

`/health` 示例：

```json
{
  "status": "ok",
  "supported_tasks": ["classify", "ocr", "detect"],
  "service": "contestant-algo-test-service",
  "version": "0.2.0",
  "bridge_mode": "mock-local"
}
```

检查模型与 backend：

```bash
curl http://127.0.0.1:8080/debug/status
```

## 默认安全开关

```env
OFFLINE_MODE=true
ALLOW_MODEL_AUTO_DOWNLOAD=false
ALLOW_DATASET_DOWNLOAD=false
RUN_FULL_TRAIN=false
```

正式比赛前保持下载关闭，确认所有模型权重已在 `models/` 本地存在，并确认 `/debug/status` 中 detect `model_exists=true`、OCR 目录 ready。

## 模型准备

```bash
python scripts/prepare_detect_model.py --model yolo11n.pt --output models/detect/yolo11n.pt
python scripts/prepare_ocr_model.py --output models/ocr/paddleocr
```

需要赛前自动下载时显式加 `--allow-download` 或设置 `ALLOW_MODEL_AUTO_DOWNLOAD=true`。比赛时不要自动下载。

## 训练快速入口

Detect smoke：

```bash
python scripts/prepare_detect_dataset.py --mode coco8 --output data/processed/detect/coco8_contest
python scripts/train_detect_yolo.py --model models/detect/yolo11n.pt --data data/processed/detect/coco8_contest/data.yaml --epochs 1 --imgsz 320 --batch 2 --device cpu --project runs/detect --name smoke --dry-run
```

Classify smoke/template：

```bash
python scripts/prepare_classify_dataset.py --mode template --output data/processed/classify/contest8
python scripts/train_classify.py --data data/processed/classify/contest8 --model mobilenet_v3_small --epochs 1 --batch 4 --imgsz 224 --device cpu --output models/classify/classifier.pt --dry-run
```

OCR synthetic：

```bash
python scripts/generate_synthetic_ocr_dataset.py --output data/processed/ocr/synthetic --count 500 --seed 42
python scripts/train_ocr_paddle.py --config configs/ocr/paddleocr_rec_template.yml --data data/processed/ocr/synthetic --dry-run
```

验证训练环境：

```bash
python scripts/verify_training_setup.py
```

## 数据登记

所有数据集先登记到：

- `docs/dataset_registry.md`
- `data/dataset_manifest.yaml`

刷新命令：

```bash
PYTHONPATH=src python -m contest_agent.training.dataset_registry --write-docs
```

大数据集不默认下载；不确定 license 的只登记为候选。

## 测试

```bash
PYTHONPATH=src pytest -q
```

## 打包

```bash
python scripts/package_submission.py
```

输出 `dist/contest_agent_submission.zip` 和 `.sha256`。包内包含 `configs/`、`src/`、`scripts/`、`docs/`、`tests/`、`models/` 和 requirements；排除 `.git`、`.venv`、`dist`、`runs`、`data/raw`、`data/processed`、缓存和大日志。

## 比赛提醒

- 不要使用 OpenAI/Gemini/Claude、云 OCR、Hosted inference 等外部推理服务处理题图。
- detect 坐标必须是原图 pixel 坐标。
- classify/detect 输出优先服从 `meta.class_names` 的合法中文标签。
- COCO 覆盖多数 detect 类，`台灯` 需要额外数据或二阶段策略。

