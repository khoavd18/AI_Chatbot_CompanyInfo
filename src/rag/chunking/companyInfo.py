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


def chunk_company_info():
    file_path = Path(settings["data"]["processed_dir"]) / "companyInfo.json"

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            company_info = json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to load companyInfo.json: {e}")
        return []

    if isinstance(company_info, list):
        if not company_info:
            logger.warning("companyInfo.json is empty")
            return []
        company_info = company_info[0]

    if not isinstance(company_info, dict):
        logger.error("Company info data is not a dictionary")
        return []

    company_id = company_info.get("id")
    company_name = _safe_strip(company_info.get("companyName"))
    company_slogan = _safe_strip(company_info.get("companySlogan"))
    company_description = _safe_strip(company_info.get("companyDescription"))

    hotlines = company_info.get("hotlines") or []
    emails = company_info.get("emails") or []
    social_links = company_info.get("socialLinks") or []

    main_address = _normalize_text(company_info.get("mainAddress"))
    working_hours = _normalize_text(company_info.get("workingHours"))
    website = _normalize_text(company_info.get("website"))

    total_projects = _normalize_number(company_info.get("totalProjects"))
    total_employees = _normalize_number(company_info.get("totalEmployees"))
    total_engineers = _normalize_number(company_info.get("totalEngineers"))
    total_architects = _normalize_number(company_info.get("totalArchitects"))

    seo_title = _normalize_text(company_info.get("seoTitle"))
    seo_description = _normalize_text(company_info.get("seoDescription"))
    thumbnail_alt = _normalize_text(company_info.get("thumbnailAlt"))
    thumbnail_url = _normalize_text(company_info.get("thumbnailUrl"))

    if not company_name:
        logger.warning("Skipping company info chunking because companyName is missing")
        return []

    hotlines_clean = [_normalize_text(item) for item in hotlines if _normalize_text(item)]
    emails_clean = [_normalize_text(item) for item in emails if _normalize_text(item)]

    social_clean = []
    for item in social_links:
        if isinstance(item, dict):
            platform = _normalize_text(item.get("platform"))
            url = _normalize_text(item.get("url"))
            if platform and url:
                social_clean.append(f"{platform}: {url}")
            elif platform:
                social_clean.append(platform)
            elif url:
                social_clean.append(url)
        else:
            text = _normalize_text(item)
            if text:
                social_clean.append(text)

    base_metadata = {
        "type": "company_info",
        "company_id": company_id,
        "company_name": company_name,
        "source": "companyInfo.json",
        "created_at": datetime.utcnow().isoformat(),
        "language": "vi",
        "company_website": website,
        "company_main_address": main_address,
    }

    chunks = []

    CHUNK_PRIORITY = {
        "overview": 1,
        "description": 2,
        "contact_info": 1,
        "stats": 3,
        "brand": 4,
    }

    # 1) OVERVIEW CHUNK
    overview_lines = [f"Tên công ty: {company_name}"]

    if company_slogan:
        overview_lines.append(f"Slogan: {company_slogan}")

    if company_description:
        overview_lines.append(f"Giới thiệu ngắn: {company_description}")

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
    if company_description:
        description_parts = split_paragraphs(company_description)
        for i, part in enumerate(description_parts):
            if not part or not part.strip():
                continue

            chunks.append({
                "text": f"Mô tả công ty {company_name}: {part.strip()}",
                "metadata": make_metadata(
                    base_metadata,
                    chunk_type="description",
                    priority=CHUNK_PRIORITY["description"],
                    part_index=i,
                )
            })

    # 3) CONTACT INFO CHUNK
    contact_lines = [f"Tên công ty: {company_name}"]

    if hotlines_clean:
        contact_lines.append(f"Hotline: {', '.join(hotlines_clean)}")

    if emails_clean:
        contact_lines.append(f"Email: {', '.join(emails_clean)}")

    if main_address:
        contact_lines.append(f"Địa chỉ chính: {main_address}")

    if working_hours:
        contact_lines.append(f"Giờ làm việc: {working_hours}")

    if website:
        contact_lines.append(f"Website: {website}")

    contact_text = _join_non_empty(contact_lines)
    if len(contact_lines) > 1:
        chunks.append({
            "text": contact_text,
            "metadata": make_metadata(
                base_metadata,
                chunk_type="contact_info",
                priority=CHUNK_PRIORITY["contact_info"],
            )
        })

    # 4) STATS CHUNK
    stats_lines = [f"Tên công ty: {company_name}"]

    if total_projects is not None:
        stats_lines.append(f"Tổng số dự án: {total_projects}")

    if total_employees is not None:
        stats_lines.append(f"Tổng số nhân sự: {total_employees}")

    if total_engineers is not None:
        stats_lines.append(f"Tổng số kỹ sư: {total_engineers}")

    if total_architects is not None:
        stats_lines.append(f"Tổng số kiến trúc sư: {total_architects}")

    stats_text = _join_non_empty(stats_lines)
    if len(stats_lines) > 1:
        chunks.append({
            "text": stats_text,
            "metadata": make_metadata(
                base_metadata,
                chunk_type="stats",
                priority=CHUNK_PRIORITY["stats"],
            )
        })

    # 5) BRAND / SEO CHUNK
    brand_lines = [f"Tên công ty: {company_name}"]

    if seo_title:
        brand_lines.append(f"SEO title: {seo_title}")

    if seo_description:
        brand_lines.append(f"SEO description: {seo_description}")

    if thumbnail_alt:
        brand_lines.append(f"Mô tả ảnh đại diện: {thumbnail_alt}")

    if social_clean:
        brand_lines.append(f"Kênh mạng xã hội: {' | '.join(social_clean)}")

    brand_text = _join_non_empty(brand_lines)
    if len(brand_lines) > 1:
        chunks.append({
            "text": brand_text,
            "metadata": make_metadata(
                base_metadata,
                chunk_type="brand",
                priority=CHUNK_PRIORITY["brand"],
                company_thumbnail_url=thumbnail_url,
            )
        })

    logger.info(f"Chunked {len(chunks)} company info chunks")
    return chunks