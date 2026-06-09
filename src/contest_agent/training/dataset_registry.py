from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from contest_agent.training.common import ROOT_DIR, write_text


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    task: str
    url: str
    license_note: str
    size_note: str
    classes_covered: list[str]
    default_download: bool
    needs_manual_review: bool
    download_method: str
    project_use: str


DEFAULT_DATASETS: tuple[DatasetEntry, ...] = (
    DatasetEntry(
        name="COCO",
        task="detect",
        url="https://cocodataset.org/",
        license_note="Public research dataset; review COCO terms before full download.",
        size_note="Full train/val is large; do not download by default.",
        classes_covered=["人", "汽车", "自行车", "手机", "水杯", "笔记本电脑", "沙发", "狗"],
        default_download=False,
        needs_manual_review=True,
        download_method="Use Ultralytics or COCO tools only after ALLOW_DATASET_DOWNLOAD=true.",
        project_use="Detect baseline and contest-class subset. 台灯 is not covered.",
    ),
    DatasetEntry(
        name="COCO8",
        task="detect",
        url="https://docs.ultralytics.com/datasets/detect/coco8/",
        license_note="Tiny sample derived from COCO; review upstream terms.",
        size_note="Tiny smoke dataset.",
        classes_covered=["人", "汽车", "自行车"],
        default_download=False,
        needs_manual_review=False,
        download_method="Ultralytics may fetch it when ALLOW_DATASET_DOWNLOAD=true or --allow-download.",
        project_use="Smoke training and pipeline validation only.",
    ),
    DatasetEntry(
        name="Open Images",
        task="detect",
        url="https://storage.googleapis.com/openimages/web/index.html",
        license_note="Mixed image licenses; must review labels and image license before use.",
        size_note="Very large; use filtered subsets only.",
        classes_covered=["台灯", "灯", "手机", "狗", "沙发"],
        default_download=False,
        needs_manual_review=True,
        download_method="Candidate only; write a filtered downloader after license review.",
        project_use="Potential long-tail supplement for 台灯.",
    ),
    DatasetEntry(
        name="LVIS",
        task="detect",
        url="https://www.lvisdataset.org/",
        license_note="Review LVIS and source image terms before use.",
        size_note="Large annotations over COCO images.",
        classes_covered=["台灯", "灯", "沙发", "杯子"],
        default_download=False,
        needs_manual_review=True,
        download_method="Candidate only; requires explicit dataset decision.",
        project_use="Candidate long-tail labels, especially lamp-like categories.",
    ),
    DatasetEntry(
        name="Places365",
        task="classify",
        url="http://places2.csail.mit.edu/",
        license_note="Research dataset; review terms before download or redistribution.",
        size_note="Large; do not download by default.",
        classes_covered=["办公室", "公园", "街道", "商场", "厨房", "卧室", "图书馆", "体育馆"],
        default_download=False,
        needs_manual_review=True,
        download_method="Manual preparation or explicit downloader only with ALLOW_DATASET_DOWNLOAD=true.",
        project_use="Candidate scene classification source.",
    ),
    DatasetEntry(
        name="Synthetic OCR Meter Readings",
        task="ocr",
        url="local://scripts/generate_synthetic_ocr_dataset.py",
        license_note="Generated locally by this project.",
        size_note="Configurable; smoke count can be 5-500 images.",
        classes_covered=["中文字段", "数字", "小数点", "%", "℃", "MPa", "V", "A"],
        default_download=True,
        needs_manual_review=False,
        download_method="Generate locally; no network required.",
        project_use="OCR smoke data and fine-tuning preparation.",
    ),
    DatasetEntry(
        name="PaddleOCR official datasets",
        task="ocr",
        url="https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/dataset/overview_en.md",
        license_note="Multiple datasets and licenses; review each source before use.",
        size_note="Varies from small to very large.",
        classes_covered=["中文", "英文", "数字", "符号"],
        default_download=False,
        needs_manual_review=True,
        download_method="Candidate only; follow PaddleOCR docs after review.",
        project_use="Possible OCR pretraining/fine-tuning data source.",
    ),
)


def registry_entries() -> list[dict[str, Any]]:
    return [asdict(entry) for entry in DEFAULT_DATASETS]


def render_manifest(entries: list[dict[str, Any]] | None = None) -> str:
    items = entries or registry_entries()
    lines = ["datasets:"]
    for item in items:
        lines.append(f"  - name: {item['name']!r}")
        for key in ("task", "url", "license_note", "size_note", "download_method", "project_use"):
            lines.append(f"    {key}: {item[key]!r}")
        lines.append("    classes_covered:")
        for label in item["classes_covered"]:
            lines.append(f"      - {label!r}")
        lines.append(f"    default_download: {str(item['default_download']).lower()}")
        lines.append(f"    needs_manual_review: {str(item['needs_manual_review']).lower()}")
    lines.append("")
    return "\n".join(lines)


def render_markdown(entries: list[dict[str, Any]] | None = None) -> str:
    items = entries or registry_entries()
    header = """# Dataset Registry

Downloads are disabled by default. Large or license-unclear datasets require explicit review and `ALLOW_DATASET_DOWNLOAD=true`.

| Name | Task | Classes | Default | Manual review | Size | Use |
| --- | --- | --- | --- | --- | --- | --- |
"""
    rows = []
    for item in items:
        rows.append(
            "| {name} | {task} | {classes} | {default} | {review} | {size} | {use} |".format(
                name=item["name"],
                task=item["task"],
                classes=", ".join(item["classes_covered"]),
                default="yes" if item["default_download"] else "no",
                review="yes" if item["needs_manual_review"] else "no",
                size=item["size_note"],
                use=item["project_use"],
            )
        )
    details = ["", "## Sources"]
    for item in items:
        details.append(f"- **{item['name']}**: {item['url']} - {item['license_note']} Download: {item['download_method']}")
    details.append("")
    return header + "\n".join(rows) + "\n" + "\n".join(details)


def write_registry_files(root: Path | None = None) -> tuple[Path, Path]:
    base = root or ROOT_DIR
    manifest = write_text(base / "data" / "dataset_manifest.yaml", render_manifest())
    docs = write_text(base / "docs" / "dataset_registry.md", render_markdown())
    return manifest, docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Print or refresh dataset registry files.")
    parser.add_argument("--write-docs", action="store_true", help="Write docs/dataset_registry.md and data/dataset_manifest.yaml")
    args = parser.parse_args()
    if args.write_docs:
        manifest, docs = write_registry_files()
        print(f"wrote {manifest}")
        print(f"wrote {docs}")
    else:
        print(render_manifest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

