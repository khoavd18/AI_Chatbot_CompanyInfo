from __future__ import annotations

import re
from typing import Any


_LOW_VALUE_DESCRIPTION_PATTERNS = [
    re.compile(r"được sử dụng trong hệ thống dữ liệu", re.IGNORECASE),
    re.compile(r"trong hệ thống nội dung", re.IGNORECASE),
    re.compile(r"là một nhóm thiết kế", re.IGNORECASE),
]

_PLACEHOLDER_MEDIA_PATTERNS = [
    re.compile(r"^hình minh họa", re.IGNORECASE),
    re.compile(r"-\s*hình minh họa$", re.IGNORECASE),
    re.compile(r"-\s*news$", re.IGNORECASE),
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_for_compare(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def is_same_or_similar(a: str, b: str) -> bool:
    a_norm = normalize_for_compare(a)
    b_norm = normalize_for_compare(b)

    if not a_norm or not b_norm:
        return False

    return a_norm == b_norm or a_norm in b_norm or b_norm in a_norm


def is_low_value_description(text: str, reference: str = "") -> bool:
    text = normalize_text(text)
    if not text:
        return True

    if reference and is_same_or_similar(text, reference):
        return True

    return any(pattern.search(text) for pattern in _LOW_VALUE_DESCRIPTION_PATTERNS)


def is_placeholder_media_text(text: str, reference: str = "") -> bool:
    text = normalize_text(text)
    if not text:
        return True

    if reference and is_same_or_similar(text, reference):
        return True

    return any(pattern.search(text) for pattern in _PLACEHOLDER_MEDIA_PATTERNS)


def make_dedupe_key(*parts: Any) -> tuple[str, ...]:
    return tuple(normalize_for_compare(str(part)) for part in parts)
