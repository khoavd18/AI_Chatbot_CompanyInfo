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


def chunk_project_categories():
    file_path = Path(settings["data"]["processed_dir"]) / "projectCategories.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            project_categories = json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load projectCategories.json: {e}")
        return []

    if isinstance(project_categories, dict):
        project_categories = [project_categories]

    if not isinstance(project_categories, list):
        logger.error("Project categories data is not a list")
        return []

    if not project_categories:
        logger.warning("No project categories found in the file")
        return []

    chunks = []

    CHUNK_PRIORITY = {
        "definition": 2,
        "description": 3,
        "seo": 4,
        "icon": 5,
    }

    for idx, project_category in enumerate(project_categories):
        if not isinstance(project_category, dict):
            logger.warning(f"Project category at index {idx} is not a dictionary")
            continue

        category_id = project_category.get("id")
        category_slug = _safe_strip(project_category.get("slug"))
        category_name = _safe_strip(project_category.get("name"))
        category_description = _safe_strip(project_category.get("description"))
        category_icon = _normalize_text(project_category.get("icon"))
        seo_title = _normalize_text(project_category.get("seoTitle"))
        seo_description = _normalize_text(project_category.get("seoDescription"))

        if not category_name and not category_slug:
            logger.warning(f"Skipping project category at index {idx} because name/slug is missing")
            continue

        display_name = category_name or category_slug

        base_metadata = {
            "type": "project_category",
            "project_category_id": category_id,
            "project_category_name": display_name,
            "project_category_slug": category_slug,
            "project_category_icon": category_icon,
            "source": "projectCategories.json",
            "created_at": datetime.utcnow().isoformat(),
            "language": "vi",
        }

        # 1) DEFINITION CHUNK
        definition_lines = [f"Tên danh mục dự án: {display_name}"]

        if category_description:
            definition_lines.append(f"Mô tả ngắn: {category_description}")
        elif category_slug:
            definition_lines.append(f"Slug nhận diện: {category_slug}")
            definition_lines.append("Đây là một danh mục dùng để phân loại các dự án của công ty.")
        else:
            definition_lines.append("Đây là một danh mục dùng để phân loại các dự án của công ty.")

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
        if category_description:
            description_parts = split_paragraphs(category_description)
            for i, part in enumerate(description_parts):
                if not part or not part.strip():
                    continue

                chunks.append({
                    "text": f"Mô tả danh mục dự án {display_name}: {part.strip()}",
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="description",
                        priority=CHUNK_PRIORITY["description"],
                        part_index=i,
                    )
                })

        # 3) SEO CHUNK
        seo_lines = [f"Tên danh mục dự án: {display_name}"]

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

        # 4) ICON CHUNK
        if category_icon:
            icon_text = _join_non_empty([
                f"Tên danh mục dự án: {display_name}",
                f"Biểu tượng danh mục: {category_icon}",
            ])
            chunks.append({
                "text": icon_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="icon",
                    priority=CHUNK_PRIORITY["icon"],
                )
            })

    logger.info(
        f"Chunked {len(chunks)} project category chunks from {len(project_categories)} project categories"
    )
    return chunks