import json
import logging
from pathlib import Path
from datetime import datetime

from src.core.setting_loader import load_settings
from src.rag.chunking.helpers.make_metadata import make_metadata
from src.rag.chunking.helpers.split_paragraphs import split_paragraphs
from src.rag.chunking.helpers.text_quality import (
    is_placeholder_media_text,
    is_same_or_similar,
    make_dedupe_key,
)

settings = load_settings()
logger = logging.getLogger("ingestion")


def _safe_strip(value):
    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_number(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return None
    return None


def _join_non_empty(lines):
    return "\n".join([line for line in lines if line and line.strip()])


def chunk_hero_slides():
    file_path = Path(settings["data"]["processed_dir"]) / "heroSlides.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            hero_slides = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load heroSlides.json: {e}")
        return []

    if isinstance(hero_slides, dict):
        hero_slides = [hero_slides]

    if not isinstance(hero_slides, list):
        logger.error(f"Unexpected data format in {file_path}: expected list")
        return []

    if not hero_slides:
        logger.warning(f"No hero slides found in {file_path}")
        return []

    chunks = []
    seen_slides = set()

    CHUNK_PRIORITY = {
        "overview": 6,
        "description": 7,
        "media": 8,
        "video": 9,
    }

    for idx, slide in enumerate(hero_slides):
        if not isinstance(slide, dict):
            logger.warning(f"Skipping invalid hero slide at index {idx}")
            continue

        slide_id = slide.get("id")
        slide_title = _safe_strip(slide.get("title"))
        slide_subtitle = _safe_strip(slide.get("subtitle"))
        slide_description = _safe_strip(slide.get("description"))
        slide_image_url = _normalize_text(slide.get("imageUrl"))
        slide_image_alt = _normalize_text(slide.get("imageAlt"))
        slide_video_url = _normalize_text(slide.get("videoUrl"))
        slide_video_title = _normalize_text(slide.get("videoTitle"))
        slide_page = _normalize_text(slide.get("page"))
        slide_order = _normalize_number(slide.get("order"))
        slide_is_active = slide.get("isActive")

        if not slide_title and not slide_subtitle and not slide_description:
            logger.warning(
                f"Skipping hero slide at index {idx} because title/subtitle/description are all missing"
            )
            continue

        display_title = slide_title or f"Hero slide {idx + 1}"
        slide_key = make_dedupe_key(
            display_title,
            slide_subtitle,
            slide_page,
            slide_image_alt,
            slide_video_title,
        )
        if slide_key in seen_slides:
            logger.info("Skipping duplicated hero slide at index %s", idx)
            continue
        seen_slides.add(slide_key)

        if is_placeholder_media_text(slide_image_alt, display_title):
            slide_image_alt = ""

        base_metadata = {
            "type": "hero_slide",
            "hero_slide_id": slide_id,
            "hero_slide_title": display_title,
            "hero_slide_subtitle": slide_subtitle,
            "hero_slide_page": slide_page,
            "hero_slide_order": slide_order,
            "hero_slide_is_active": slide_is_active,
            "hero_slide_image_url": slide_image_url,
            "hero_slide_video_url": slide_video_url,
            "source": "heroSlides.json",
            "created_at": datetime.utcnow().isoformat(),
            "language": "vi",
        }

        # 1) OVERVIEW CHUNK
        overview_lines = [f"Tiêu đề hero slide: {display_title}"]

        if slide_subtitle:
            overview_lines.append(f"Phụ đề: {slide_subtitle}")

        if slide_page:
            overview_lines.append(f"Trang hiển thị: {slide_page}")

        if slide_order is not None:
            overview_lines.append(f"Thứ tự hiển thị: {slide_order}")

        if slide_is_active is not None:
            overview_lines.append(f"Đang hoạt động: {'Có' if slide_is_active else 'Không'}")

        overview_text = _join_non_empty(overview_lines)
        if overview_text:
            chunks.append({
                "text": overview_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="overview",
                    priority=CHUNK_PRIORITY["overview"],
                )
            })

        # 2) DESCRIPTION CHUNKS
        if slide_description:
            description_parts = split_paragraphs(slide_description)
            for i, part in enumerate(description_parts):
                if not part or not part.strip():
                    continue

                chunks.append({
                    "text": f"Mô tả hero slide {display_title}: {part.strip()}",
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="description",
                        priority=CHUNK_PRIORITY["description"],
                        part_index=i,
                    )
                })

        # 3) MEDIA CHUNK
        if slide_image_alt:
            media_text = _join_non_empty([
                f"Tiêu đề hero slide: {display_title}",
                f"Mô tả hình ảnh: {slide_image_alt}",
            ])
            chunks.append({
                "text": media_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="media",
                    priority=CHUNK_PRIORITY["media"],
                )
            })
        # 4) VIDEO CHUNK
        if slide_video_title or slide_video_url:
            video_lines = [f"Tiêu đề hero slide: {display_title}"]

            if slide_video_title and not is_same_or_similar(slide_video_title, display_title):
                video_lines.append(f"Tiêu đề video: {slide_video_title}")

            if slide_video_url and not slide_video_title:
                video_lines.append("Hero slide này có video minh họa.")

            video_text = _join_non_empty(video_lines)
            if len(video_lines) > 1 and video_text:
                chunks.append({
                    "text": video_text,
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="video",
                        priority=CHUNK_PRIORITY["video"],
                    )
                })

    logger.info(f"Chunked {len(chunks)} hero slide chunks from {len(hero_slides)} hero slides")
    return chunks
