import json
import logging
from pathlib import Path
from datetime import datetime
import re

from bs4 import BeautifulSoup

from src.core.setting_loader import load_settings
from src.rag.chunking.helpers.make_metadata import make_metadata
from src.rag.chunking.helpers.split_paragraphs import split_paragraphs

settings = load_settings()
logger = logging.getLogger("ingestion")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # bỏ hashtag
    text = re.sub(r'#\S+', '', text)

    # bỏ emoji cơ bản / ký tự trang trí lặp
    text = re.sub(r'[✨🌟🔥💯🎉📌📍]+', ' ', text)

    # giảm lặp từ/cụm từ liên tiếp
    text = re.sub(r'(\b[^\s]+\b)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

    # chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text
def _safe_strip(value):
    if isinstance(value, str):
        return value.strip()
    return ""


def _html_to_text(html: str) -> str:
    html = _safe_strip(html)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


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


def _format_date_for_text(value: str) -> str:
    value = _safe_strip(value)
    if not value:
        return ""
    # Ví dụ: 2024-05-21T00:00:00.000Z -> 2024-05-21
    return value[:10]


def _join_non_empty(lines):
    return "\n".join([line for line in lines if line and line.strip()])


def chunk_projects():
    file_path = Path(settings["data"]["processed_dir"]) / "projects.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            projects = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load projects.json: {e}")
        return []

    if isinstance(projects, dict):
        projects = [projects]

    if not isinstance(projects, list):
        logger.error("Projects data is not a list")
        return []

    if not projects:
        logger.warning("No projects found in the file")
        return []

    chunks = []

    CHUNK_PRIORITY = {
        "overview": 1,
        "full_content": 2,
        "context": 3,
        "specs": 4,
        "seo": 5,
        "media": 6,
    }

    for idx, project in enumerate(projects):
        if not isinstance(project, dict):
            logger.warning(f"Project at index {idx} is not a dict")
            continue

        project_id = project.get("id")
        project_name = _safe_strip(project.get("title"))
        project_slug = _safe_strip(project.get("slug"))
        project_description = _safe_strip(clean_text(project.get("description")))
        project_content_html = _safe_strip(project.get("content"))
        project_content_text = _html_to_text(project_content_html)

        project_investor = _normalize_text(project.get("investor"))
        project_location = _normalize_text(project.get("location"))
        project_thumbnail_url = _normalize_text(project.get("thumbnailUrl"))
        project_thumbnail_alt = _normalize_text(project.get("thumbnailAlt"))

        project_completed_date = _normalize_text(project.get("completedDate"))
        project_published_at = _normalize_text(project.get("publishedAt"))
        project_status = _normalize_text(project.get("status"))

        project_area = _normalize_number(project.get("area"))
        project_view_count = _normalize_number(project.get("viewCount"))
        project_is_featured = project.get("isFeatured")

        project_seo_title = _normalize_text(project.get("seoTitle"))
        project_seo_description = _normalize_text(project.get("seoDescription"))

        category = project.get("category") or {}
        interior = project.get("interiorStyle") or {}
        architecture = project.get("architectureType") or {}

        category_name = _normalize_text(category.get("name"))
        category_slug = _normalize_text(category.get("slug"))

        interior_name = _normalize_text(interior.get("name"))
        interior_slug = _normalize_text(interior.get("slug"))

        architecture_name = _normalize_text(architecture.get("name"))
        architecture_slug = _normalize_text(architecture.get("slug"))

        if not project_name and not project_slug:
            logger.warning(f"Skipping project at index {idx} because title/slug is missing")
            continue

        display_name = project_name or project_slug

        base_metadata = {
            "type": "project",
            "project_id": project_id,
            "project_name": display_name,
            "project_slug": project_slug,
            "project_category_name": category_name,
            "project_category_slug": category_slug,
            "project_interior_name": interior_name,
            "project_interior_slug": interior_slug,
            "project_architecture_name": architecture_name,
            "project_architecture_slug": architecture_slug,
            "project_location": project_location,
            "project_investor": project_investor,
            "project_area": project_area,
            "project_completed_date": project_completed_date,
            "project_published_at": project_published_at,
            "project_status": project_status,
            "project_thumbnail_url": project_thumbnail_url,
            "source": "projects.json",
            "created_at": datetime.utcnow().isoformat(),
            "language": "vi",
        }

        # 1) OVERVIEW CHUNK
        overview_lines = [f"Tên dự án: {display_name}"]

        if project_description:
            overview_lines.append(f"Mô tả ngắn: {project_description}")

        if category_name:
            overview_lines.append(f"Danh mục dự án: {category_name}")

        if interior_name:
            overview_lines.append(f"Phong cách nội thất: {interior_name}")

        if architecture_name:
            overview_lines.append(f"Phong cách kiến trúc: {architecture_name}")

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

        # 2) FULL CONTENT CHUNKS
        # Ưu tiên content, fallback sang description
        detail_source = ""
        if project_content_text:
            detail_source = project_content_text
        elif project_description:
            detail_source = project_description

        if detail_source:
            detail_parts = split_paragraphs(detail_source)
            for i, part in enumerate(detail_parts):
                if not part or not part.strip():
                    continue

                chunks.append({
                    "text": f"Nội dung chi tiết dự án {display_name}: {part.strip()}",
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="full_content",
                        priority=CHUNK_PRIORITY["full_content"],
                        part_index=i,
                    )
                })

        # 3) CONTEXT CHUNK
        context_lines = [f"Tên dự án: {display_name}"]

        if project_location:
            context_lines.append(f"Địa điểm: {project_location}")

        if project_investor:
            context_lines.append(f"Chủ đầu tư: {project_investor}")

        if project_published_at:
            context_lines.append(f"Ngày công bố: {_format_date_for_text(project_published_at)}")

        context_text = _join_non_empty(context_lines)
        if len(context_lines) > 1:
            chunks.append({
                "text": context_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="context",
                    priority=CHUNK_PRIORITY["context"],
                )
            })

        # 4) SPECS CHUNK
        specs_lines = [f"Tên dự án: {display_name}"]

        if project_area is not None:
            specs_lines.append(f"Diện tích: {project_area} m²")

        if project_completed_date:
            specs_lines.append(f"Ngày hoàn thành: {_format_date_for_text(project_completed_date)}")

        if project_status:
            specs_lines.append(f"Trạng thái: {project_status}")

        if project_is_featured is not None:
            specs_lines.append(f"Dự án nổi bật: {'Có' if project_is_featured else 'Không'}")

        if project_view_count is not None:
            specs_lines.append(f"Lượt xem: {project_view_count}")

        specs_text = _join_non_empty(specs_lines)
        if len(specs_lines) > 1:
            chunks.append({
                "text": specs_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="specs",
                    priority=CHUNK_PRIORITY["specs"],
                )
            })

        # 5) SEO CHUNK
        seo_lines = [f"Tên dự án: {display_name}"]

        if project_seo_title:
            seo_lines.append(f"SEO title: {project_seo_title}")

        if project_seo_description:
            seo_lines.append(f"SEO description: {project_seo_description}")

        if project_thumbnail_alt:
            seo_lines.append(f"Mô tả ảnh đại diện: {project_thumbnail_alt}")

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

        # 6) MEDIA CHUNK
        # Chỉ tạo nếu có alt text hoặc ít nhất có ảnh để metadata lưu lại.
        if project_thumbnail_alt:
            media_text = _join_non_empty([
                f"Tên dự án: {display_name}",
                f"Hình ảnh minh họa: {project_thumbnail_alt}",
            ])
            chunks.append({
                "text": media_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="media",
                    priority=CHUNK_PRIORITY["media"],
                )
            })
        elif project_thumbnail_url:
            # Chỉ giữ 1 chunk nhẹ nếu không có alt text nhưng vẫn có ảnh
            chunks.append({
                "text": f"Dự án {display_name} có hình ảnh minh họa.",
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="media",
                    priority=CHUNK_PRIORITY["media"],
                )
            })

    logger.info(f"Chunked {len(chunks)} project chunks from {len(projects)} projects")
    return chunks