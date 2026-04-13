import json
import logging
from pathlib import Path
from datetime import datetime

from src.core.setting_loader import load_settings
from src.rag.chunking.helpers.make_metadata import make_metadata
from src.rag.chunking.helpers.split_paragraphs import split_paragraphs
from src.rag.chunking.helpers.text_quality import (
    is_low_value_description,
    is_same_or_similar,
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


def _join_non_empty(lines):
    return "\n".join([line for line in lines if line and line.strip()])


def chunk_news_categories():
    file_path = Path(settings["data"]["processed_dir"]) / "newsCategories.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            news_categories = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load newsCategories.json: {e}")
        return []

    if isinstance(news_categories, dict):
        news_categories = [news_categories]

    if not isinstance(news_categories, list):
        logger.error("News categories data is not a list")
        return []

    if not news_categories:
        logger.warning("No news categories found in the file")
        return []

    chunks = []

    CHUNK_PRIORITY = {
        "definition": 2,
        "description": 3,
        "seo": 4,
    }

    for idx, category in enumerate(news_categories):
        if not isinstance(category, dict):
            logger.warning(f"Category at index {idx} is not a dictionary")
            continue

        news_category_id = category.get("id")
        news_category_name = _safe_strip(category.get("name"))
        news_category_slug = _safe_strip(category.get("slug"))
        news_category_description = _safe_strip(category.get("description"))
        seo_title = _normalize_text(category.get("seoTitle"))
        seo_description = _normalize_text(category.get("seoDescription"))

        if not news_category_name and not news_category_slug:
            logger.warning(f"Skipping news category at index {idx} because name/slug is missing")
            continue

        display_name = news_category_name or news_category_slug
        if is_low_value_description(news_category_description, display_name):
            news_category_description = ""

        base_metadata = {
            "type": "news_category",
            "news_category_id": news_category_id,
            "news_category_name": display_name,
            "news_category_slug": news_category_slug,
            "source": "newsCategories.json",
            "created_at": datetime.utcnow().isoformat(),
            "language": "vi",
        }

        # 1) DEFINITION CHUNK
        definition_lines = [f"Tên danh mục tin tức: {display_name}"]

        if news_category_description:
            definition_lines.append(f"Mô tả ngắn: {news_category_description}")
        elif news_category_slug and not is_same_or_similar(news_category_slug, display_name):
            definition_lines.append(f"Slug nhận diện: {news_category_slug}")

        definition_text = _join_non_empty(definition_lines)
        if definition_text:
            chunks.append({
                "text": definition_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="definition",
                    priority=CHUNK_PRIORITY["definition"],
                )
            })

        # 2) DESCRIPTION CHUNKS
        if news_category_description:
            description_parts = split_paragraphs(news_category_description)
            for i, part in enumerate(description_parts):
                if not part or not part.strip():
                    continue

                chunks.append({
                    "text": f"Mô tả danh mục tin tức {display_name}: {part.strip()}",
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="description",
                        priority=CHUNK_PRIORITY["description"],
                        part_index=i,
                    )
                })

        # 3) SEO CHUNK
        seo_lines = [f"Tên danh mục tin tức: {display_name}"]
        meaningful_seo = False

        if seo_title and not is_same_or_similar(seo_title, display_name):
            seo_lines.append(f"SEO title: {seo_title}")
            meaningful_seo = True

        if seo_description and not is_low_value_description(seo_description, display_name):
            seo_lines.append(f"SEO description: {seo_description}")
            meaningful_seo = True

        seo_text = _join_non_empty(seo_lines)
        if meaningful_seo and len(seo_lines) > 1:
            chunks.append({
                "text": seo_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="seo",
                    priority=CHUNK_PRIORITY["seo"],
                )
            })

    logger.info(
        f"Chunked {len(chunks)} news category chunks from {len(news_categories)} news categories"
    )
    return chunks
