from __future__ import annotations

import hashlib
import random
import time
from typing import Iterable, TypeVar


T = TypeVar("T")


def stable_seed(seed_text: str) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_random(seed_text: str) -> random.Random:
    return random.Random(stable_seed(seed_text))


def stable_token(seed_text: str, length: int = 8) -> str:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return digest[:length].upper()


def stable_choice(items: Iterable[T], seed_text: str) -> T:
    values = list(items)
    if not values:
        raise ValueError("stable_choice requires a non-empty iterable")
    return stable_random(seed_text).choice(values)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def normalize_task_type(task_type: str) -> str:
    return task_type.strip().lower()


def summarize_text(value: str, max_length: int = 64) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3]}..."
