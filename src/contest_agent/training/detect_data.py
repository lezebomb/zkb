from __future__ import annotations

import shutil
from pathlib import Path

from contest_agent.training.common import ensure_dir, write_text


DETECT_LABELS = ["人", "汽车", "自行车", "手机", "水杯", "笔记本电脑", "台灯", "沙发", "狗"]

EN_TO_CN = {
    "person": "人",
    "car": "汽车",
    "bicycle": "自行车",
    "cell phone": "手机",
    "cellphone": "手机",
    "mobile phone": "手机",
    "phone": "手机",
    "cup": "水杯",
    "mug": "水杯",
    "laptop": "笔记本电脑",
    "notebook": "笔记本电脑",
    "couch": "沙发",
    "sofa": "沙发",
    "dog": "狗",
    "lamp": "台灯",
    "desk lamp": "台灯",
}

COCO_CONTEST_CLASSES = ["person", "car", "bicycle", "cell phone", "cup", "laptop", "couch", "dog"]


def contest_data_yaml(path: Path | str, train: str = "images/train", val: str = "images/val") -> str:
    root = Path(path)
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(DETECT_LABELS))
    return f"path: {root.as_posix()}\ntrain: {train}\nval: {val}\nnames:\n{names}\n"


def prepare_coco8(output: Path | str, allow_download: bool = False) -> Path:
    out = ensure_dir(output)
    ensure_dir(out / "images" / "train")
    ensure_dir(out / "images" / "val")
    ensure_dir(out / "labels" / "train")
    ensure_dir(out / "labels" / "val")
    write_text(out / "data.yaml", contest_data_yaml(out))
    note = [
        "COCO8 contest smoke dataset placeholder.",
        "If Ultralytics and network are available, run with --allow-download to let YOLO fetch coco8 during training.",
        "Class names are remapped to contest Chinese labels.",
        "台灯 is not covered by COCO and needs extra data or a second-stage strategy.",
    ]
    if allow_download:
        note.append("Download was allowed, but this script keeps a local YOLO yaml placeholder to avoid implicit large downloads.")
    write_text(out / "README.md", "\n".join(note) + "\n")
    return out / "data.yaml"


def prepare_coco_subset(output: Path | str, classes: list[str], max_images_per_class: int, allow_download: bool = False) -> Path:
    out = ensure_dir(output)
    ensure_dir(out / "images" / "train")
    ensure_dir(out / "images" / "val")
    ensure_dir(out / "labels" / "train")
    ensure_dir(out / "labels" / "val")
    selected = classes or COCO_CONTEST_CLASSES
    write_text(out / "data.yaml", contest_data_yaml(out))
    write_text(
        out / "README.md",
        "\n".join(
            [
                "COCO contest subset workspace.",
                f"requested_classes: {', '.join(selected)}",
                f"max_images_per_class: {max_images_per_class}",
                f"allow_download: {allow_download}",
                "Full COCO is never downloaded unless ALLOW_DATASET_DOWNLOAD=true or --allow-download is set.",
                "If raw COCO already exists, place/filter it here and keep YOLO labels in images/train,val and labels/train,val.",
                "台灯 is not covered by COCO; add Open Images/LVIS/custom data or use a second-stage strategy.",
            ]
        )
        + "\n",
    )
    return out / "data.yaml"


def prepare_custom(input_dir: Path | str, output: Path | str) -> Path:
    src = Path(input_dir)
    if not src.exists():
        raise FileNotFoundError(f"custom dataset input does not exist: {src}")
    required = ["images/train", "images/val", "labels/train", "labels/val"]
    missing = [item for item in required if not (src / item).exists()]
    if missing:
        raise ValueError(f"custom YOLO dataset missing: {', '.join(missing)}")
    out = Path(output)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    if not (out / "data.yaml").exists():
        write_text(out / "data.yaml", contest_data_yaml(out))
    return out / "data.yaml"

