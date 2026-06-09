# Dataset Registry

Downloads are disabled by default. Large or license-unclear datasets require explicit review and `ALLOW_DATASET_DOWNLOAD=true`.

| Name | Task | Classes | Default | Manual review | Size | Use |
| --- | --- | --- | --- | --- | --- | --- |
| COCO | detect | 人, 汽车, 自行车, 手机, 水杯, 笔记本电脑, 沙发, 狗 | no | yes | Full train/val is large; do not download by default. | Detect baseline and contest-class subset. 台灯 is not covered. |
| COCO8 | detect | 人, 汽车, 自行车 | no | no | Tiny smoke dataset. | Smoke training and pipeline validation only. |
| Open Images | detect | 台灯, 灯, 手机, 狗, 沙发 | no | yes | Very large; use filtered subsets only. | Potential long-tail supplement for 台灯. |
| LVIS | detect | 台灯, 灯, 沙发, 杯子 | no | yes | Large annotations over COCO images. | Candidate long-tail labels, especially lamp-like categories. |
| Places365 | classify | 办公室, 公园, 街道, 商场, 厨房, 卧室, 图书馆, 体育馆 | no | yes | Large; do not download by default. | Candidate scene classification source. |
| Synthetic OCR Meter Readings | ocr | 中文字段, 数字, 小数点, %, ℃, MPa, V, A | yes | no | Configurable; smoke count can be 5-500 images. | OCR smoke data and fine-tuning preparation. |
| PaddleOCR official datasets | ocr | 中文, 英文, 数字, 符号 | no | yes | Varies from small to very large. | Possible OCR pretraining/fine-tuning data source. |

## Sources
- **COCO**: https://cocodataset.org/ - Public research dataset; review COCO terms before full download. Download: Use Ultralytics or COCO tools only after ALLOW_DATASET_DOWNLOAD=true.
- **COCO8**: https://docs.ultralytics.com/datasets/detect/coco8/ - Tiny sample derived from COCO; review upstream terms. Download: Ultralytics may fetch it when ALLOW_DATASET_DOWNLOAD=true or --allow-download.
- **Open Images**: https://storage.googleapis.com/openimages/web/index.html - Mixed image licenses; must review labels and image license before use. Download: Candidate only; write a filtered downloader after license review.
- **LVIS**: https://www.lvisdataset.org/ - Review LVIS and source image terms before use. Download: Candidate only; requires explicit dataset decision.
- **Places365**: http://places2.csail.mit.edu/ - Research dataset; review terms before download or redistribution. Download: Manual preparation or explicit downloader only with ALLOW_DATASET_DOWNLOAD=true.
- **Synthetic OCR Meter Readings**: local://scripts/generate_synthetic_ocr_dataset.py - Generated locally by this project. Download: Generate locally; no network required.
- **PaddleOCR official datasets**: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/dataset/overview_en.md - Multiple datasets and licenses; review each source before use. Download: Candidate only; follow PaddleOCR docs after review.
