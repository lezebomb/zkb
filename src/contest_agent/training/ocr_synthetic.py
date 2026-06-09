from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from contest_agent.training.common import ensure_dir, write_text


FIELDS = ["阀位", "压力", "温度", "流量", "电压", "电流"]
UNITS = ["%", "℃", "MPa", "V", "A"]


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def random_text(rng: random.Random) -> str:
    readings = [f"{rng.uniform(0, 150):.1f}", f"{rng.uniform(0, 1):.2f}", str(rng.randint(0, 999))]
    patterns = [
        lambda: rng.choice(readings),
        lambda: rng.choice(readings) + rng.choice(UNITS),
        lambda: rng.choice(FIELDS) + rng.choice(readings) + rng.choice(UNITS),
        lambda: rng.choice(FIELDS) + " " + rng.choice(readings),
    ]
    return rng.choice(patterns)()


def render_sample(text: str, rng: random.Random) -> Image.Image:
    width = rng.randint(180, 360)
    height = rng.randint(54, 110)
    bg = rng.randint(210, 248)
    image = Image.new("RGB", (width, height), (bg, bg, bg))
    draw = ImageDraw.Draw(image)
    font = _font(rng.randint(24, 42))
    bbox = draw.textbbox((0, 0), text, font=font)
    x = max(4, (width - (bbox[2] - bbox[0])) // 2 + rng.randint(-8, 8))
    y = max(4, (height - (bbox[3] - bbox[1])) // 2 + rng.randint(-5, 5))
    draw.text((x, y), text, fill=(rng.randint(0, 55), rng.randint(0, 55), rng.randint(0, 55)), font=font)
    if rng.random() < 0.55:
        image = image.rotate(rng.uniform(-4.0, 4.0), expand=True, fillcolor=(bg, bg, bg))
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.1)))
    if rng.random() < 0.70:
        overlay = Image.new("RGB", image.size, (0, 0, 0))
        pixels = overlay.load()
        for py in range(image.height):
            for px in range(image.width):
                n = rng.randint(-12, 12)
                pixels[px, py] = (max(0, n), max(0, n), max(0, n))
        image = Image.blend(image, overlay, alpha=0.05)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
    return image


def generate_dataset(output: Path | str, count: int, seed: int = 42) -> Path:
    out = ensure_dir(output)
    image_dir = ensure_dir(out / "images")
    rng = random.Random(seed)
    label_lines = []
    for idx in range(1, count + 1):
        text = random_text(rng)
        rel = f"images/{idx:06d}.png"
        render_sample(text, rng).save(out / rel)
        label_lines.append(f"{rel}\t{text}")
    write_text(out / "labels.txt", "\n".join(label_lines) + "\n")
    write_text(
        out / "README.md",
        "Synthetic OCR data for meter-like readings. Useful for smoke tests and fine-tuning preparation; validate before competition use.\n",
    )
    return out

