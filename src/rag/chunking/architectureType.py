import json
import logging
from pathlib import Path
from datetime import datetime

from src.core.setting_loader import load_settings
from src.rag.chunking.helpers.make_metadata import make_metadata
from src.rag.chunking.helpers.split_paragraphs import split_paragraphs

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


def chunk_architecture_types():
    file_path = Path(settings["data"]["processed_dir"]) / "architectureTypes.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            architecture_types = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load architectureTypes.json: {e}")
        return []

    if isinstance(architecture_types, dict):
        architecture_types = [architecture_types]

    if not isinstance(architecture_types, list):
        logger.error("Architecture types data is not a list")
        return []

    if not architecture_types:
        logger.warning("No architecture types found in the data")
        return []

    chunks = []

    CHUNK_PRIORITY = {
        "definition": 2,
        "description": 3,
        "seo": 4,
        "media": 5,
    }

    for idx, architecture_type in enumerate(architecture_types):
        if not isinstance(architecture_type, dict):
            logger.warning(f"Invalid architecture type at index {idx}: expected a dictionary")
            continue

        architecture_id = architecture_type.get("id")
        architecture_slug = _safe_strip(architecture_type.get("slug"))
        architecture_name = _safe_strip(architecture_type.get("name"))
        architecture_description = _safe_strip(architecture_type.get("description"))

        seo_title = _normalize_text(architecture_type.get("seoTitle"))
        seo_description = _normalize_text(architecture_type.get("seoDescription"))
        image_alt = _normalize_text(architecture_type.get("imageAlt"))
        image_url = _normalize_text(architecture_type.get("imageUrl"))

        if not architecture_name and not architecture_slug:
            logger.warning(f"Skipping architecture type at index {idx} because name/slug is missing")
            continue

        display_name = architecture_name or architecture_slug

        base_metadata = {
            "type": "architecture_type",
            "architecture_type_id": architecture_id,
            "architecture_type_name": display_name,
            "architecture_type_slug": architecture_slug,
            "architecture_type_image_url": image_url,
            "source": "architectureTypes.json",
            "created_at": datetime.utcnow().isoformat(),
            "language": "vi",
        }

        # 1) DEFINITION CHUNK
        definition_lines = [f"Tên phong cách kiến trúc: {display_name}"]

        if architecture_description:
            definition_lines.append(f"Mô tả ngắn: {architecture_description}")
        elif architecture_slug:
            definition_lines.append(
                f"Slug nhận diện: {architecture_slug}"
            )
            definition_lines.append(
                "Đây là một phong cách kiến trúc có trong hệ thống dữ liệu của công ty."
            )
        else:
            definition_lines.append(
                "Đây là một phong cách kiến trúc có trong hệ thống dữ liệu của công ty."
            )

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
        if architecture_description:
            description_parts = split_paragraphs(architecture_description)
            for i, part in enumerate(description_parts):
                if not part or not part.strip():
                    continue

                chunks.append({
                    "text": f"Mô tả chi tiết phong cách kiến trúc {display_name}: {part.strip()}",
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="description",
                        priority=CHUNK_PRIORITY["description"],
                        part_index=i,
                    )
                })

        # 3) SEO CHUNK
        seo_lines = [f"Tên phong cách kiến trúc: {display_name}"]

        if seo_title:
            seo_lines.append(f"SEO title: {seo_title}")

        if seo_description:
            seo_lines.append(f"SEO description: {seo_description}")

        seo_text = _join_non_empty(seo_lines)
        if len(seo_lines) > 1:
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
                f"Tên phong cách kiến trúc: {display_name}",
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
        elif image_url:
            chunks.append({
                "text": f"Phong cách kiến trúc {display_name} có hình ảnh minh họa.",
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="media",
                    priority=CHUNK_PRIORITY["media"],
                )
            })

    logger.info(f"Chunked {len(chunks)} architecture type chunks from {len(architecture_types)} architecture types")
    return chunks