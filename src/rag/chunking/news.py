import json
import logging
import re
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup

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


def _html_to_text(html: str) -> str:
    html = _safe_strip(html)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _format_date_for_text(value: str) -> str:
    value = _safe_strip(value)
    if not value:
        return ""
    return value[:10]


def _join_non_empty(lines):
    return "\n".join([line for line in lines if line and line.strip()])


def _limit_text(text: str, max_len: int = 280) -> str:
    text = _normalize_text(text)
    if not text:
        return ""

    if len(text) <= max_len:
        return text

    cut = text[:max_len].rsplit(" ", 1)[0].strip()
    if not cut:
        cut = text[:max_len].strip()
    return cut + "..."


def _normalize_for_compare(text: str) -> str:
    text = _normalize_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _is_same_or_similar(a: str, b: str) -> bool:
    a_norm = _normalize_for_compare(a)
    b_norm = _normalize_for_compare(b)

    if not a_norm or not b_norm:
        return False

    if a_norm == b_norm:
        return True

    return a_norm in b_norm or b_norm in a_norm


def chunk_news():
    file_path = Path(settings["data"]["processed_dir"]) / "news.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            news = json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load news.json: {e}")
        return []

    if isinstance(news, dict):
        news = [news]

    if not isinstance(news, list):
        logger.error("News data is not a list")
        return []

    if not news:
        logger.warning("No news found in the file")
        return []

    chunks = []

    CHUNK_PRIORITY = {
        "overview": 1,
        "full_content": 2,
        "meta": 3,
        "seo": 4,
        "media": 5,
    }

    for idx, news_item in enumerate(news):
        if not isinstance(news_item, dict):
            logger.warning(f"News item at index {idx} is not a dictionary")
            continue

        news_item_id = news_item.get("id")
        news_item_title = _safe_strip(news_item.get("title"))
        news_item_slug = _safe_strip(news_item.get("slug"))
        news_item_excerpt = _safe_strip(news_item.get("excerpt"))
        news_item_content_html = _safe_strip(news_item.get("content"))
        news_item_content_text = _html_to_text(news_item_content_html)

        news_item_author = _normalize_text(news_item.get("author"))
        news_item_status = _normalize_text(news_item.get("status"))
        news_item_published_at = _normalize_text(news_item.get("publishedAt"))
        news_item_reading_time = _normalize_number(news_item.get("readingTime"))
        news_item_view_count = _normalize_number(news_item.get("viewCount"))
        news_item_is_featured = news_item.get("isFeatured")
        news_item_project_id = news_item.get("projectId")

        news_item_seo_title = _normalize_text(news_item.get("seoTitle"))
        news_item_seo_description = _normalize_text(news_item.get("seoDescription"))
        news_item_thumbnail_url = _normalize_text(news_item.get("thumbnailUrl"))
        news_item_thumbnail_alt = _normalize_text(news_item.get("thumbnailAlt"))

        category = news_item.get("category") or {}
        category_name = _normalize_text(category.get("name"))
        category_slug = _normalize_text(category.get("slug"))

        if not news_item_title and not news_item_slug:
            logger.warning(f"Skipping news item at index {idx} because title/slug is missing")
            continue

        display_title = news_item_title or news_item_slug

        base_metadata = {
            "type": "news",
            "news_item_id": news_item_id,
            "news_item_title": display_title,
            "news_item_slug": news_item_slug,
            "news_category_name": category_name,
            "news_category_slug": category_slug,
            "news_author": news_item_author,
            "news_status": news_item_status,
            "news_published_at": news_item_published_at,
            "news_reading_time": news_item_reading_time,
            "news_view_count": news_item_view_count,
            "news_is_featured": news_item_is_featured,
            "news_project_id": news_item_project_id,
            "news_thumbnail_url": news_item_thumbnail_url,
            "source": "news.json",
            "created_at": datetime.utcnow().isoformat(),
            "language": "vi",
        }

        # 1) OVERVIEW CHUNK
        excerpt_short = _limit_text(news_item_excerpt, max_len=280)

        overview_lines = [f"Tiêu đề tin tức: {display_title}"]

        if excerpt_short:
            overview_lines.append(f"Tóm tắt: {excerpt_short}")

        if category_name:
            overview_lines.append(f"Danh mục: {category_name}")

        if news_item_author:
            overview_lines.append(f"Tác giả: {news_item_author}")

        if news_item_published_at:
            overview_lines.append(f"Ngày xuất bản: {_format_date_for_text(news_item_published_at)}")

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
        # Chỉ fallback sang excerpt khi content thật sự không có
        detail_source = news_item_content_text if news_item_content_text else news_item_excerpt

        if detail_source:
            detail_parts = split_paragraphs(detail_source)
            for i, part in enumerate(detail_parts):
                if not part or not part.strip():
                    continue

                chunks.append({
                    "text": f"Nội dung tin tức {display_title}: {part.strip()}",
                    "metadata": make_metadata(
                        base_metadata,
                        chunk_type="full_content",
                        priority=CHUNK_PRIORITY["full_content"],
                        part_index=i,
                    )
                })

        # 3) META CHUNK
        meta_lines = [f"Tiêu đề tin tức: {display_title}"]

        if category_name:
            meta_lines.append(f"Danh mục: {category_name}")

        if news_item_author:
            meta_lines.append(f"Tác giả: {news_item_author}")

        if news_item_status:
            meta_lines.append(f"Trạng thái: {news_item_status}")

        if news_item_reading_time is not None:
            meta_lines.append(f"Thời gian đọc: {news_item_reading_time} phút")

        if news_item_view_count is not None:
            meta_lines.append(f"Lượt xem: {news_item_view_count}")

        if news_item_is_featured is not None:
            meta_lines.append(f"Bài viết nổi bật: {'Có' if news_item_is_featured else 'Không'}")

        if news_item_project_id is not None:
            meta_lines.append(f"Liên kết dự án ID: {news_item_project_id}")

        meta_text = _join_non_empty(meta_lines)
        if len(meta_lines) > 1:
            chunks.append({
                "text": meta_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="meta",
                    priority=CHUNK_PRIORITY["meta"],
                )
            })

        # 4) SEO CHUNK
        seo_lines = [f"Tiêu đề tin tức: {display_title}"]

        has_distinct_seo = False

        if news_item_seo_title and not _is_same_or_similar(news_item_seo_title, display_title):
            seo_lines.append(f"SEO title: {news_item_seo_title}")
            has_distinct_seo = True

        if news_item_seo_description and not _is_same_or_similar(news_item_seo_description, news_item_excerpt):
            seo_lines.append(f"SEO description: {_limit_text(news_item_seo_description, max_len=300)}")
            has_distinct_seo = True

        if has_distinct_seo:
            seo_text = _join_non_empty(seo_lines)
            chunks.append({
                "text": seo_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="seo",
                    priority=CHUNK_PRIORITY["seo"],
                )
            })

        # 5) MEDIA CHUNK
        # Chỉ giữ nếu alt khác đáng kể so với title
        if news_item_thumbnail_alt and not _is_same_or_similar(news_item_thumbnail_alt, display_title):
            media_text = _join_non_empty([
                f"Tiêu đề tin tức: {display_title}",
                f"Hình ảnh minh họa: {news_item_thumbnail_alt}",
            ])
            chunks.append({
                "text": media_text,
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="media",
                    priority=CHUNK_PRIORITY["media"],
                )
            })
        elif news_item_thumbnail_url and not news_item_thumbnail_alt:
            chunks.append({
                "text": f"Tin tức {display_title} có hình ảnh minh họa.",
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="media",
                    priority=CHUNK_PRIORITY["media"],
                )
            })

    logger.info(f"Chunked {len(chunks)} news chunks from {len(news)} news items")
    return chunks