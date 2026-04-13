import json
import logging
from pathlib import Path
from datetime import datetime

from src.core.setting_loader import load_settings
from src.rag.chunking.helpers.make_metadata import make_metadata
from src.rag.chunking.helpers.split_paragraphs import split_paragraphs
from src.rag.chunking.helpers.text_quality import (
    is_low_value_description,
    is_placeholder_media_text,
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


def chunk_interior_styles():
    file_path = Path(settings["data"]["processed_dir"]) / "interiorStyles.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            interior_styles = json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load interiorStyles.json: {e}")
        return []

    if isinstance(interior_styles, dict):
        interior_styles = [interior_styles]

    if not isinstance(interior_styles, list):
        logger.error("Interior styles data is not a list")
        return []

    if not interior_styles:
        logger.warning("No interior styles found in the file")
        return []

    chunks = []

    CHUNK_PRIORITY = {
        "definition": 2,
        "description": 3,
        "seo": 4,
        "media": 5,
    }

    for idx, interior_style in enumerate(interior_styles):
        if not isinstance(interior_style, dict):
            logger.warning(f"Interior style at index {idx} is not a dictionary")
            continue

        interior_id = interior_style.get("id")
        interior_slug = _safe_strip(interior_style.get("slug"))
        interior_name = _safe_strip(interior_style.get("name"))
        interior_description = _safe_strip(interior_style.get("description"))

        seo_title = _normalize_text(interior_style.get("seoTitle"))
        seo_description = _normalize_text(interior_style.get("seoDescription"))
        image_alt = _normalize_text(interior_style.get("imageAlt"))
        image_url = _normalize_text(interior_style.get("imageUrl"))

        if not interior_name and not interior_slug:
            logger.warning(f"Skipping interior style at index {idx} because name/slug is missing")
            continue

        display_name = interior_name or interior_slug
        if is_low_value_description(interior_description, display_name):
            interior_description = ""

        if is_placeholder_media_text(image_alt, display_name):
            image_alt = ""

        base_metadata = {
            "type": "interior_style",
            "interior_id": interior_id,
            "interior_name": display_name,
            "interior_slug": interior_slug,
            "interior_image_url": image_url,
            "source": "interiorStyles.json",
            "created_at": datetime.utcnow().isoformat(),
            "language": "vi",
        }

        # 1) DEFINITION CHUNK
        definition_lines = [f"Tên phong cách nội thất: {display_name}"]

        if interior_description:
            definition_lines.append(f"Mô tả ngắn: {interior_description}")
        elif interior_slug and not is_same_or_similar(interior_slug, display_name):
            definition_lines.append(f"Slug nhận diện: {interior_slug}")

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
        if interior_description:
            description_parts = split_paragraphs(interior_description)
            for i, part in enumerate(description_parts):
                if not part or not part.strip():
                    continue

                chunks.append({
                    "text": f"Mô tả chi tiết phong cách nội thất {display_name}: {part.strip()}",
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="description",
                        priority=CHUNK_PRIORITY["description"],
                        part_index=i,
                    )
                })

        # 3) SEO CHUNK
        seo_lines = [f"Tên phong cách nội thất: {display_name}"]
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

        # 4) MEDIA CHUNK
        if image_alt:
            media_text = _join_non_empty([
                f"Tên phong cách nội thất: {display_name}",
                f"Mô tả hình ảnh minh họa: {image_alt}",
            ])
            chunks.append({
                "text": media_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="media",
                    priority=CHUNK_PRIORITY["media"],
                )
            })
    logger.info(f"Chunked {len(chunks)} interior style chunks from {len(interior_styles)} interior styles")
    return chunks
