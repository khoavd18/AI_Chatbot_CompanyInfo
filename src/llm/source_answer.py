from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

from src.core.schema import RetrievedDocument


IMAGE_FIELDS = [
    "project_thumbnail_url",
    "news_thumbnail_url",
    "company_thumbnail_url",
    "architecture_type_image_url",
    "interior_image_url",
    "hero_slide_image_url",
]

VIDEO_FIELDS = [
    "hero_slide_video_url",
]


def _strip_accents(text: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", text or "")
        if not unicodedata.combining(character)
    )
    return normalized.replace("đ", "d").replace("Đ", "D")


def _normalize(text: str) -> str:
    plain = _strip_accents(text).lower()
    return re.sub(r"\s+", " ", plain).strip()


def _contains_marker(text: str, marker: str) -> bool:
    normalized_text = _normalize(text)
    normalized_marker = _normalize(marker)
    if not normalized_text or not normalized_marker:
        return False

    if " " in normalized_marker:
        return f" {normalized_marker} " in f" {normalized_text} "

    return normalized_marker in set(normalized_text.split())


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(_contains_marker(text, phrase) for phrase in phrases)


def _extract_location_hint(question: str) -> str:
    normalized_question = _normalize(question)
    patterns = [
        r"\bo\s+(.+)",
        r"\btai\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_question)
        if match:
            hint = match.group(1).strip(" ?.!,:;")
            if hint:
                return hint
    return ""


def _extract_labeled_values(documents: list[RetrievedDocument]) -> dict[str, str]:
    values: dict[str, str] = {}

    for document in documents:
        for raw_line in document.text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            label, value = line.split(":", 1)
            key = _normalize(label)
            cleaned_value = value.strip()
            if key and cleaned_value and key not in values:
                values[key] = cleaned_value

    return values


def _pick_label(values: dict[str, str], *labels: str) -> str | None:
    for label in labels:
        value = values.get(_normalize(label))
        if value:
            return value
    return None


def _unique_names(items: list[str]) -> list[str]:
    seen = set()
    unique_items = []
    for item in items:
        value = (item or "").strip()
        normalized_value = _normalize(value)
        if not value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        unique_items.append(value)
    return unique_items


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item)


def _clip_value(text: str | None, limit: int = 220) -> str | None:
    if not text:
        return None

    normalized_text = re.sub(r"\s+", " ", str(text)).strip()
    if len(normalized_text) <= limit:
        return normalized_text
    return normalized_text[:limit].rstrip(" ,;:.") + "..."


def _strip_prefixes(text: str | None, prefixes: list[str] | None = None) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""

    for prefix in prefixes or []:
        normalized_prefix = re.sub(r"\s+", " ", prefix).strip()
        if normalized_prefix and value.startswith(normalized_prefix):
            return value[len(normalized_prefix) :].strip(" :-")

    return value


def _best_chunk_excerpt(
    documents: list[RetrievedDocument],
    *,
    chunk_types: set[str] | None = None,
    prefixes: list[str] | None = None,
    limit: int = 260,
) -> str | None:
    candidates = [
        document
        for document in documents
        if not chunk_types or (document.metadata or {}).get("chunk_type") in chunk_types
    ]
    if not candidates:
        return None

    best_document = max(candidates, key=_ranking_value)
    return _clip_value(_strip_prefixes(best_document.text, prefixes), limit=limit)


def _ranking_value(document: RetrievedDocument) -> float:
    rerank_score = document.metadata.get("rerank_score")
    if isinstance(rerank_score, (int, float)):
        return float(rerank_score)
    return float(document.score)


def _project_groups(documents: list[RetrievedDocument]) -> dict[str, list[RetrievedDocument]]:
    groups: dict[str, list[RetrievedDocument]] = defaultdict(list)
    for document in documents:
        if document.metadata.get("type") != "project":
            continue
        project_name = (document.metadata.get("project_name") or "").strip()
        if project_name:
            groups[project_name].append(document)
    return groups


def _ranked_project_groups(documents: list[RetrievedDocument]) -> list[tuple[str, list[RetrievedDocument]]]:
    groups = _project_groups(documents)

    return sorted(
        groups.items(),
        key=lambda item: (
            max(_ranking_value(document) for document in item[1]),
            len(item[1]),
        ),
        reverse=True,
    )


def _group_documents_by_name(
    documents: list[RetrievedDocument],
    *,
    source_type: str,
    field_name: str,
) -> dict[str, list[RetrievedDocument]]:
    groups: dict[str, list[RetrievedDocument]] = defaultdict(list)
    for document in documents:
        if document.metadata.get("type") != source_type:
            continue

        display_name = str(document.metadata.get(field_name, "")).strip()
        if display_name:
            groups[display_name].append(document)

    return groups


def _news_groups(documents: list[RetrievedDocument]) -> dict[str, list[RetrievedDocument]]:
    return _group_documents_by_name(
        documents,
        source_type="news",
        field_name="news_item_title",
    )


def _project_category_groups(documents: list[RetrievedDocument]) -> dict[str, list[RetrievedDocument]]:
    return _group_documents_by_name(
        documents,
        source_type="project_category",
        field_name="project_category_name",
    )


def _news_category_groups(documents: list[RetrievedDocument]) -> dict[str, list[RetrievedDocument]]:
    return _group_documents_by_name(
        documents,
        source_type="news_category",
        field_name="news_category_name",
    )


def _question_mentions_projects(question: str) -> bool:
    return _contains_any(question, ["dự án", "công trình"])


def _question_mentions_news(question: str) -> bool:
    return _contains_any(question, ["bài viết", "tin tức", "news"])


def _question_mentions_categories(question: str) -> bool:
    return _contains_any(question, ["danh mục", "chuyên mục", "loại hình"])


def _question_mentions_styles(question: str) -> bool:
    return _contains_any(question, ["phong cách", "style"])


def _is_project_detail_request(question: str) -> bool:
    detail_markers = [
        "thong tin chi tiet",
        "chi tiet du an",
        "gioi thieu du an",
        "du an do",
        "du an nay",
        "cong trinh do",
        "noi ro hon",
        "them thong tin",
    ]
    return _question_mentions_projects(question) and _contains_any(question, detail_markers)


def _is_project_category_request(question: str) -> bool:
    return _question_mentions_projects(question) and _question_mentions_categories(question)


def _is_news_category_request(question: str) -> bool:
    return _question_mentions_news(question) and _question_mentions_categories(question)


def _is_project_list_request(question: str) -> bool:
    if not _question_mentions_projects(question):
        return False

    list_markers = [
        "du an nao",
        "co du an nao",
        "nhung du an",
        "cac du an",
        "mot du an",
        "1 du an",
        "mot cong trinh",
        "1 cong trinh",
        "cho toi mot du an",
        "cho toi 1 du an",
        "du an cua cong ty",
        "du an cua cong ty ban",
        "cong ty co du an nao",
        "mot vai",
        "mot so",
        "goi y",
        "de xuat",
        "noi bat",
        "tieu bieu",
        "tham khao",
        "liet ke",
        "danh sach",
    ]
    return _contains_any(question, list_markers)


def _wants_single_project_suggestion(question: str) -> bool:
    return _contains_any(
        question,
        [
            "mot du an",
            "1 du an",
            "mot cong trinh",
            "1 cong trinh",
            "cho toi mot du an",
            "cho toi 1 du an",
        ],
    )


def _is_news_list_request(question: str) -> bool:
    if not _question_mentions_news(question):
        return False

    list_markers = [
        "bài viết nào",
        "tin tức nào",
        "có bài viết nào",
        "có tin tức nào",
        "những bài viết",
        "các bài viết",
        "một vài bài viết",
        "một số bài viết",
        "gợi ý bài viết",
        "danh sách bài viết",
        "danh sách tin tức",
        "liệt kê bài viết",
    ]
    return _contains_any(question, list_markers)


def _is_category_list_request(question: str) -> bool:
    if not _question_mentions_categories(question):
        return False

    return _contains_any(
        question,
        [
            "những",
            "các",
            "danh sách",
            "liệt kê",
            "có những",
            "gồm những",
            "gồm các",
        ],
    )


def _project_list_item(project_name: str, documents: list[RetrievedDocument]) -> str:
    best_document = max(documents, key=_ranking_value)
    metadata = best_document.metadata

    location = metadata.get("project_location")
    category = metadata.get("project_category_name")
    interior = metadata.get("project_interior_name")
    architecture = metadata.get("project_architecture_name")

    details = []
    if location:
        details.append(str(location).strip())
    if category:
        details.append(str(category).strip())
    elif interior:
        details.append(str(interior).strip())
    elif architecture:
        details.append(str(architecture).strip())

    if details:
        return f"{project_name} ({' | '.join(details[:2])})"
    return project_name


def _project_clarification_prompt(documents: list[RetrievedDocument], intro: str) -> str | None:
    ranked_groups = _ranked_project_groups(documents)
    project_names = [project_name for project_name, _ in ranked_groups[:4]]
    if not project_names:
        return None
    return intro + "\n" + _format_bullets(project_names)


def _source_title(metadata: dict) -> str:
    return (
        metadata.get("project_name")
        or metadata.get("news_item_title")
        or metadata.get("company_name")
        or metadata.get("architecture_type_name")
        or metadata.get("interior_name")
        or metadata.get("project_category_name")
        or metadata.get("news_category_name")
        or metadata.get("hero_slide_title")
        or "nội dung này"
    )


def _extract_after_marker(question: str, marker: str) -> str | None:
    normalized_question = _normalize(question)
    normalized_marker = _normalize(marker)
    if not normalized_question or not normalized_marker:
        return None

    pattern = rf"{re.escape(normalized_marker)}\s+(.+?)\s+khong(?:\s*[?!.,:]*)?$"
    match = re.search(pattern, normalized_question)
    if not match:
        return None

    extracted = match.group(1).strip(" ?!.,:;\"'")
    return extracted or None


def _collect_media_urls(documents: list[RetrievedDocument]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    videos: list[str] = []

    for document in documents:
        metadata = document.metadata or {}
        for key in IMAGE_FIELDS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                images.append(value.strip())
        for key in VIDEO_FIELDS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                videos.append(value.strip())

    return _unique_names(images), _unique_names(videos)


def _dominant_project_name(documents: list[RetrievedDocument]) -> str | None:
    names = [
        document.metadata.get("project_name", "")
        for document in documents
        if document.metadata.get("type") == "project"
    ]
    ranked_names = [name for name in names if name]
    if not ranked_names:
        return None
    return Counter(ranked_names).most_common(1)[0][0]


def _select_media_documents(question: str, documents: list[RetrievedDocument]) -> tuple[str | None, list[RetrievedDocument]]:
    project_groups = _project_groups(documents)
    normalized_question = _normalize(question)

    for project_name, group_documents in project_groups.items():
        if _normalize(project_name) in normalized_question:
            return project_name, group_documents

    dominant_project = _dominant_project_name(documents)
    if dominant_project:
        return dominant_project, project_groups.get(dominant_project, documents)

    best_document = max(documents, key=_ranking_value)
    subject_title = _source_title(best_document.metadata or {})
    subject_type = best_document.metadata.get("type")

    related_documents = []
    for document in documents:
        metadata = document.metadata or {}
        if subject_type and metadata.get("type") != subject_type:
            continue
        if _source_title(metadata) == subject_title:
            related_documents.append(document)

    return subject_title, related_documents or documents


def _compose_media_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    if not _contains_any(question, ["hình ảnh", "ảnh", "image", "thumbnail", "video", "clip", "media"]):
        return None

    subject_title, selected_documents = _select_media_documents(question, documents)
    image_urls, video_urls = _collect_media_urls(selected_documents or documents)

    if not image_urls and not video_urls:
        if subject_title:
            return f"Mình chưa thấy URL hình ảnh hoặc video cho {subject_title} trong dữ liệu hiện tại."
        return "Mình chưa thấy URL hình ảnh hoặc video liên quan trong dữ liệu hiện tại."

    label = subject_title or "nội dung này"
    lines = [f"Mình đã tìm được media cho {label} và hiển thị bên dưới phần trả lời."]

    if image_urls:
        lines.append(f"- Hình ảnh: {len(image_urls[:5])} mục")

    if video_urls:
        lines.append(f"- Video: {len(video_urls[:3])} mục")

    return "\n".join(lines)


def _compose_subjective_project_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    if not _contains_any(
        question,
        [
            "tâm huyết nhất",
            "ấn tượng nhất",
            "đẹp nhất",
            "tốt nhất",
            "nổi bật nhất",
            "best",
        ],
    ):
        return None

    project_groups = _project_groups(documents)
    if not project_groups:
        return None

    ranked_project_names = _unique_names(
        [
            document.metadata.get("project_name", "")
            for document in documents
            if document.metadata.get("type") == "project"
        ]
    )

    lines = [
        'Dữ liệu hiện tại chưa có tiêu chí xếp hạng để xác định đâu là dự án "tâm huyết nhất".',
    ]

    if ranked_project_names:
        lines.append(f"Nếu cần chọn một dự án để xem tiếp, có thể bắt đầu với: {ranked_project_names[0]}.")
        lines.append("Một vài dự án đang xuất hiện nổi bật trong kết quả hiện tại:")
        lines.extend(f"- {name}" for name in ranked_project_names[:5])

    return "\n".join(lines)


def _category_summary(
    *,
    display_name: str,
    documents: list[RetrievedDocument],
    intro: str,
    definition_prefixes: list[str],
) -> str:
    labeled_values = _extract_labeled_values(documents)
    description = _pick_label(labeled_values, "Mô tả ngắn")
    if not description:
        description = _best_chunk_excerpt(
            documents,
            chunk_types={"description", "definition"},
            prefixes=definition_prefixes,
            limit=240,
        )

    slug = _pick_label(labeled_values, "Slug nhận diện")
    lines = [intro.format(name=display_name)]
    if description:
        lines.append(f"- Mô tả: {description}")
    elif slug:
        lines.append(f"- Slug: {slug}")
    return "\n".join(lines)


def _compose_project_category_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    groups = _project_category_groups(documents)
    if not groups:
        if _is_project_category_request(question):
            return "Mình chưa thấy danh mục dự án phù hợp trong kết quả hiện tại."
        return None

    normalized_question = _normalize(question)
    for category_name, category_documents in groups.items():
        if _normalize(category_name) in normalized_question:
            return _category_summary(
                display_name=category_name,
                documents=category_documents,
                intro="Công ty hiện có danh mục dự án {name}.",
                definition_prefixes=[
                    f"Tên danh mục dự án: {category_name}",
                    f"Mô tả danh mục dự án {category_name}:",
                ],
            )

    if _is_project_category_request(question):
        category_names = _unique_names(list(groups.keys()))
        if _is_category_list_request(question) or len(category_names) > 1:
            return "Hệ thống hiện có các danh mục dự án sau:\n" + _format_bullets(category_names[:8])

        only_name = category_names[0]
        return _category_summary(
            display_name=only_name,
            documents=groups[only_name],
            intro="Công ty hiện có danh mục dự án {name}.",
            definition_prefixes=[
                f"Tên danh mục dự án: {only_name}",
                f"Mô tả danh mục dự án {only_name}:",
            ],
        )

    return None


def _compose_news_category_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    groups = _news_category_groups(documents)
    if not groups:
        if _is_news_category_request(question):
            return "Mình chưa thấy danh mục tin tức phù hợp trong kết quả hiện tại."
        return None

    normalized_question = _normalize(question)
    for category_name, category_documents in groups.items():
        if _normalize(category_name) in normalized_question:
            return _category_summary(
                display_name=category_name,
                documents=category_documents,
                intro="Hệ thống hiện có danh mục tin tức {name}.",
                definition_prefixes=[
                    f"Tên danh mục tin tức: {category_name}",
                    f"Mô tả danh mục tin tức {category_name}:",
                ],
            )

    if _is_news_category_request(question):
        category_names = _unique_names(list(groups.keys()))
        if _is_category_list_request(question) or len(category_names) > 1:
            return "Hệ thống hiện có các danh mục tin tức sau:\n" + _format_bullets(category_names[:8])

        only_name = category_names[0]
        return _category_summary(
            display_name=only_name,
            documents=groups[only_name],
            intro="Hệ thống hiện có danh mục tin tức {name}.",
            definition_prefixes=[
                f"Tên danh mục tin tức: {only_name}",
                f"Mô tả danh mục tin tức {only_name}:",
            ],
        )

    return None


def _company_strength_bullets(overview: str | None) -> list[str]:
    if not overview:
        return []

    bullets: list[str] = []
    strength_rules = [
        (
            ["kien truc", "noi that", "xay dung"],
            "Thiết kế kiến trúc, nội thất và giải pháp xây dựng đồng bộ.",
        ),
        (
            ["cong nang", "lay con nguoi lam trung tam", "hanh vi su dung"],
            "Ưu tiên công năng sử dụng và nhu cầu thực tế của người dùng.",
        ),
        (
            ["kha thi thi cong", "thi cong", "kiem soat chat luong"],
            "Đề cao tính khả thi thi công, kiểm soát chất lượng và bám sát cam kết.",
        ),
        (
            ["ngan sach", "chi phi"],
            "Tối ưu chi phí đầu tư theo ngân sách và bài toán vận hành.",
        ),
        (
            ["van hanh dai han", "thich ung theo thoi gian"],
            "Phát triển giải pháp có thể thích ứng theo thời gian và mục tiêu vận hành dài hạn.",
        ),
        (
            ["ben vung", "chu dau tu", "cong dong"],
            "Hướng tới giá trị bền vững cho chủ đầu tư, người sử dụng và cộng đồng.",
        ),
    ]

    for markers, bullet in strength_rules:
        if _contains_any(overview, markers):
            bullets.append(bullet)

    return _unique_names(bullets)


def _contains_count_marker(question: str) -> bool:
    return _contains_any(question, ["bao nhieu", "co may", "tong so", "so luong"])


def _is_employee_count_question(question: str) -> bool:
    return (
        _contains_any(question, ["bao nhieu nguoi"])
        or (_contains_count_marker(question) and _contains_any(question, ["nhan vien", "nhan su", "nhan luc"]))
    )


def _is_engineer_count_question(question: str) -> bool:
    return _contains_count_marker(question) and _contains_any(question, ["ky su"])


def _is_architect_count_question(question: str) -> bool:
    return _contains_count_marker(question) and _contains_any(question, ["kien truc su"])


def _is_company_overview_question(question: str) -> bool:
    return _contains_any(
        question,
        [
            "gioi thieu",
            "thong tin tong quan",
            "tong quan",
            "ve cong ty",
            "thong tin cong ty",
            "gioi thieu cong ty",
            "thong tin ve cong ty",
            "profile cong ty",
            "ho so cong ty",
        ],
    )


def _is_company_address_question(question: str) -> bool:
    return _contains_any(
        question,
        [
            "dia chi",
            "dia diem",
            "o dau",
            "o cho nao",
            "o duong nao",
            "nam o dau",
            "tru so o dau",
            "tru so chinh o dau",
            "van phong o dau",
            "van phong chinh o dau",
            "van phong o cho nao",
            "tru so o cho nao",
        ],
    )


def _is_company_hotline_question(question: str) -> bool:
    return _contains_any(
        question,
        [
            "hotline",
            "so dien thoai",
            "dien thoai",
            "so hotline",
            "so lien he",
            "sdt",
        ],
    )


def _is_company_email_question(question: str) -> bool:
    return _contains_any(question, ["email", "mail", "thu dien tu"])


def _is_company_working_hours_question(question: str) -> bool:
    return _contains_any(
        question,
        [
            "gio lam viec",
            "thoi gian lam viec",
            "lam viec may gio",
            "lam tu may gio",
            "mo cua luc nao",
            "dong cua luc nao",
        ],
    )


def _is_company_contact_question(question: str) -> bool:
    return _contains_any(question, ["lien he", "thong tin lien he"]) and not any(
        (
            _is_company_address_question(question),
            _is_company_hotline_question(question),
            _is_company_email_question(question),
            _is_company_working_hours_question(question),
        )
    )


def _compose_company_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    company_documents = [document for document in documents if document.metadata.get("type") == "company_info"]
    if not company_documents:
        return None

    labeled_values = _extract_labeled_values(company_documents)
    metadata = max(company_documents, key=_ranking_value).metadata

    company_name = metadata.get("company_name") or _pick_label(labeled_values, "Ten cong ty")
    slogan = _pick_label(labeled_values, "Slogan")
    overview_raw = _pick_label(labeled_values, "Gioi thieu ngan")
    overview = _clip_value(overview_raw, limit=260)
    address = metadata.get("company_main_address") or _pick_label(labeled_values, "Dia chi chinh", "Dia chi")
    hotline = _pick_label(labeled_values, "Hotline")
    email = _pick_label(labeled_values, "Email")
    website = metadata.get("company_website") or _pick_label(labeled_values, "Website")
    working_hours = _pick_label(labeled_values, "Gio lam viec")

    total_projects = _pick_label(labeled_values, "Tong so du an")
    total_employees = _pick_label(labeled_values, "Tong so nhan su")
    total_engineers = _pick_label(labeled_values, "Tong so ky su")
    total_architects = _pick_label(labeled_values, "Tong so kien truc su")

    if _is_company_hotline_question(question):
        return f"Hotline của {company_name} là: {hotline}" if company_name and hotline else None

    if _is_company_email_question(question):
        return f"Email liên hệ của {company_name} là: {email}" if company_name and email else None

    if _is_company_address_question(question):
        return f"Địa chỉ của {company_name} là: {address}" if company_name and address else None

    if _contains_any(question, ["website"]):
        return f"Website của {company_name} là: {website}" if company_name and website else None

    if _contains_any(question, ["ten cong ty", "cong ty ten gi", "ten doanh nghiep"]):
        return f"Ten cong ty la: {company_name}" if company_name else None

    if _contains_any(question, ["slogan"]):
        return f"Slogan cua {company_name} la: {slogan}" if company_name and slogan else None

    if _is_company_working_hours_question(question):
        return f"Giờ làm việc của {company_name} là: {working_hours}" if company_name and working_hours else None

    if _contains_any(question, ["tong so du an", "bao nhieu du an"]):
        return f"{company_name} hiện có tổng số dự án là: {total_projects}" if company_name and total_projects else None

    if _is_employee_count_question(question):
        if company_name and total_employees:
            return f"{company_name} hiện có tổng số nhân sự là: {total_employees}"
        if company_name:
            return f"Mình chưa thấy số lượng nhân sự của {company_name} trong dữ liệu hiện tại."
        return "Mình chưa thấy số lượng nhân sự của công ty trong dữ liệu hiện tại."

    if _is_engineer_count_question(question):
        if company_name and total_engineers:
            return f"{company_name} hiện có tổng số kỹ sư là: {total_engineers}"
        if company_name:
            return f"Mình chưa thấy số lượng kỹ sư của {company_name} trong dữ liệu hiện tại."
        return "Mình chưa thấy số lượng kỹ sư trong dữ liệu hiện tại."

    if _is_architect_count_question(question):
        if company_name and total_architects:
            return f"{company_name} hiện có tổng số kiến trúc sư là: {total_architects}"
        if company_name:
            return f"Mình chưa thấy số lượng kiến trúc sư của {company_name} trong dữ liệu hiện tại."
        return "Mình chưa thấy số lượng kiến trúc sư trong dữ liệu hiện tại."

    if _contains_any(
        question,
        [
            "the manh",
            "linh vuc",
            "dich vu",
            "chuyen ve",
            "nang luc",
            "so truong",
            "manh o",
            "lam gi",
        ],
    ):
        strength_bullets = _company_strength_bullets(overview_raw or overview)
        if strength_bullets:
            intro = f"{company_name} hiện mạnh ở các mảng sau:" if company_name else "Hiện công ty mạnh ở các mảng sau:"
            return intro + "\n" + _format_bullets(strength_bullets[:4])

        if overview:
            prefix = f"{company_name} hiện tập trung vào:" if company_name else "Hiện công ty tập trung vào:"
            return f"{prefix} {overview}"

    if _is_company_contact_question(question):
        lines = []
        if company_name:
            lines.append(f"Thong tin lien he cua {company_name}:")
        if address:
            lines.append(f"- Dia chi: {address}")
        if hotline:
            lines.append(f"- Hotline: {hotline}")
        if email:
            lines.append(f"- Email: {email}")
        if website:
            lines.append(f"- Website: {website}")
        if working_hours:
            lines.append(f"- Gio lam viec: {working_hours}")
        return "\n".join(lines) if len(lines) >= 2 else None

    if not _is_company_overview_question(question):
        return None

    lines = []
    if company_name:
        lines.append(f"Thông tin về {company_name}:")
    if slogan:
        lines.append(f"- Slogan: {slogan}")
    if overview:
        lines.append(f"- Giới thiệu ngắn: {overview}")
    if address:
        lines.append(f"- Địa chỉ: {address}")
    if hotline:
        lines.append(f"- Hotline: {hotline}")
    if email:
        lines.append(f"- Email: {email}")
    if website:
        lines.append(f"- Website: {website}")
    if working_hours:
        lines.append(f"- Giờ làm việc: {working_hours}")

    if _is_company_overview_question(question):
        stats = _unique_names(
            [
                f"Tổng số dự án: {total_projects}" if total_projects else "",
                f"Tổng số nhân sự: {total_employees}" if total_employees else "",
                f"Tổng số kỹ sư: {total_engineers}" if total_engineers else "",
                f"Tổng số kiến trúc sư: {total_architects}" if total_architects else "",
            ]
        )
        lines.extend(f"- {item}" for item in stats[:2])

    return "\n".join(lines) if lines else None


def _style_summary(
    *,
    style_type: str,
    style_name: str,
    style_documents: list[RetrievedDocument],
    all_documents: list[RetrievedDocument],
) -> str:
    labeled_values = _extract_labeled_values(style_documents)
    if style_type == "interior":
        description = _pick_label(labeled_values, "Mô tả ngắn") or _best_chunk_excerpt(
            style_documents,
            chunk_types={"description", "definition"},
            prefixes=[
                f"Tên phong cách nội thất: {style_name}",
                f"Mô tả chi tiết phong cách nội thất {style_name}:",
            ],
            limit=260,
        )
    else:
        description = _pick_label(labeled_values, "Mô tả ngắn") or _best_chunk_excerpt(
            style_documents,
            chunk_types={"description", "definition"},
            prefixes=[
                f"Tên phong cách kiến trúc: {style_name}",
                f"Mô tả chi tiết phong cách kiến trúc {style_name}:",
            ],
            limit=260,
        )

    related_projects = []
    for document in all_documents:
        if document.metadata.get("type") != "project":
            continue
        if style_type == "interior" and document.metadata.get("project_interior_name") == style_name:
            related_projects.append(document.metadata.get("project_name", ""))
        if style_type == "architecture" and document.metadata.get("project_architecture_name") == style_name:
            related_projects.append(document.metadata.get("project_name", ""))

    project_names = _unique_names(related_projects)
    lines = [
        f"{style_name} là một phong cách {'nội thất' if style_type == 'interior' else 'kiến trúc'} có trong hệ thống.",
    ]
    if description:
        lines.append(f"- Mô tả: {description}")
    if project_names:
        lines.append("Một số dự án liên quan:")
        lines.extend(f"- {name}" for name in project_names[:5])
    return "\n".join(lines)


def _news_summary(title: str, documents: list[RetrievedDocument], question: str) -> str:
    labeled_values = _extract_labeled_values(documents)
    metadata = max(documents, key=_ranking_value).metadata

    summary = _pick_label(labeled_values, "Tóm tắt")
    if not summary:
        summary = _best_chunk_excerpt(
            documents,
            chunk_types={"full_content", "overview"},
            prefixes=[
                f"Nội dung tin tức {title}:",
                f"Tiêu đề tin tức: {title}",
            ],
            limit=260,
        )

    category = metadata.get("news_category_name") or _pick_label(labeled_values, "Danh mục")
    published = metadata.get("news_published_at") or _pick_label(labeled_values, "Ngày xuất bản")
    if published:
        published = str(published)[:10]

    mention_target = _extract_after_marker(question, "có nhắc đến")
    if mention_target:
        haystack = " ".join(document.text for document in documents)
        if _normalize(mention_target) in _normalize(haystack):
            if summary:
                return f"Có, bài viết {title} có nhắc đến {mention_target}. Nội dung liên quan: {summary}"
            return f"Có, bài viết {title} có nhắc đến {mention_target}."
        return f"Mình chưa thấy {mention_target} trong phần nội dung đang có của bài viết {title}."

    if _contains_any(question, ["nói về điều gì", "nói về gì", "chủ đề gì", "về gì"]):
        if summary:
            return f"Bài viết {title} nói về: {summary}"

    lines = [f"Thông tin về bài viết {title}:"]
    if summary:
        lines.append(f"- Tóm tắt: {summary}")
    if category:
        lines.append(f"- Danh mục: {category}")
    if published:
        lines.append(f"- Ngày xuất bản: {published}")
    return "\n".join(lines)


def _compose_style_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    normalized_question = _normalize(question)

    style_documents = documents
    interior_names = _unique_names(
        [
            document.metadata.get("interior_name", "")
            for document in style_documents
            if document.metadata.get("type") == "interior_style"
        ]
    )
    architecture_names = _unique_names(
        [
            document.metadata.get("architecture_type_name", "")
            for document in style_documents
            if document.metadata.get("type") == "architecture_type"
        ]
    )

    if not interior_names and not architecture_names:
        return None

    if any(
        key in normalized_question
        for key in ["nhung phong cach noi that", "phong cach noi that nao", "co nhung phong cach noi that"]
    ):
        return "Công ty hiện có các phong cách nội thất sau:\n" + _format_bullets(interior_names[:8])

    if any(
        key in normalized_question
        for key in ["nhung phong cach kien truc", "phong cach kien truc nao", "co nhung phong cach kien truc"]
    ):
        return "Công ty hiện có các phong cách kiến trúc sau:\n" + _format_bullets(architecture_names[:8])

    interior_groups = _group_documents_by_name(
        style_documents,
        source_type="interior_style",
        field_name="interior_name",
    )
    architecture_groups = _group_documents_by_name(
        style_documents,
        source_type="architecture_type",
        field_name="architecture_type_name",
    )

    for style_name, grouped_documents in interior_groups.items():
        if _normalize(style_name) in normalized_question:
            return _style_summary(
                style_type="interior",
                style_name=style_name,
                style_documents=grouped_documents,
                all_documents=documents,
            )

    for style_name, grouped_documents in architecture_groups.items():
        if _normalize(style_name) in normalized_question:
            return _style_summary(
                style_type="architecture",
                style_name=style_name,
                style_documents=grouped_documents,
                all_documents=documents,
            )

    if _question_mentions_styles(question):
        if len(interior_groups) == 1:
            only_name = next(iter(interior_groups))
            return _style_summary(
                style_type="interior",
                style_name=only_name,
                style_documents=interior_groups[only_name],
                all_documents=documents,
            )

        if len(architecture_groups) == 1:
            only_name = next(iter(architecture_groups))
            return _style_summary(
                style_type="architecture",
                style_name=only_name,
                style_documents=architecture_groups[only_name],
                all_documents=documents,
            )

        groups = []
        if interior_names:
            groups.append("Phong cách nội thất:\n" + _format_bullets(interior_names[:6]))
        if architecture_names:
            groups.append("Phong cách kiến trúc:\n" + _format_bullets(architecture_names[:6]))
        return "\n\n".join(groups) if groups else None

    return None


def _compose_news_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    groups = _news_groups(documents)
    if not groups:
        if _question_mentions_news(question):
            return "Mình chưa thấy bài viết phù hợp trong kết quả hiện tại."
        return None

    normalized_question = _normalize(question)
    for title, news_documents in groups.items():
        if _normalize(title) in normalized_question:
            return _news_summary(title, news_documents, question)

    if len(groups) == 1 and _question_mentions_news(question):
        only_title = next(iter(groups))
        return _news_summary(only_title, groups[only_title], question)

    if _is_news_list_request(question):
        items = []
        for title, news_documents in sorted(
            groups.items(),
            key=lambda item: max(_ranking_value(document) for document in item[1]),
            reverse=True,
        ):
            metadata = max(news_documents, key=_ranking_value).metadata
            category = metadata.get("news_category_name")
            published = metadata.get("news_published_at")
            details = []
            if category:
                details.append(str(category).strip())
            if published:
                details.append(str(published)[:10])
            if details:
                items.append(f"{title} ({' | '.join(details[:2])})")
            else:
                items.append(title)

        unique_items = _unique_names(items)
        if unique_items:
            return "Mình tìm thấy một vài bài viết phù hợp trong kết quả hiện tại:\n" + _format_bullets(unique_items[:5])

    return None


def _project_summary(project_name: str, documents: list[RetrievedDocument]) -> str:
    best_document = max(documents, key=_ranking_value)
    metadata = best_document.metadata
    labeled_values = _extract_labeled_values(documents)

    description = _clip_value(_pick_label(labeled_values, "Mô tả ngắn"), limit=220)
    location = metadata.get("project_location") or _pick_label(labeled_values, "Địa điểm")
    investor = metadata.get("project_investor") or _pick_label(labeled_values, "Chủ đầu tư")
    area = metadata.get("project_area")
    completed_date = metadata.get("project_completed_date") or _pick_label(labeled_values, "Ngày hoàn thành")
    status = metadata.get("project_status") or _pick_label(labeled_values, "Trạng thái")
    category = metadata.get("project_category_name") or _pick_label(labeled_values, "Danh mục dự án")
    interior = metadata.get("project_interior_name") or _pick_label(labeled_values, "Phong cách nội thất")
    architecture = metadata.get("project_architecture_name") or _pick_label(labeled_values, "Phong cách kiến trúc")

    lines = [f"Thông tin về dự án {project_name}:"]
    if description:
        lines.append(f"- Mô tả ngắn: {description}")
    if location:
        lines.append(f"- Địa điểm: {location}")
    if investor:
        lines.append(f"- Chủ đầu tư: {investor}")
    if category:
        lines.append(f"- Danh mục: {category}")
    if interior:
        lines.append(f"- Phong cách nội thất: {interior}")
    if architecture:
        lines.append(f"- Phong cách kiến trúc: {architecture}")
    if area not in (None, ""):
        lines.append(f"- Diện tích: {area} m²")
    if completed_date:
        lines.append(f"- Ngày hoàn thành: {str(completed_date)[:10]}")
    if status:
        lines.append(f"- Trạng thái: {status}")
    return "\n".join(lines)


def _compose_project_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    normalized_question = _normalize(question)
    groups = _project_groups(documents)
    if not groups:
        if _question_mentions_projects(question):
            return "Mình chưa thấy dự án phù hợp trong kết quả hiện tại. Bạn có thể nêu rõ tên dự án, loại công trình hoặc địa điểm để mình tìm chính xác hơn."
        return None

    for project_name, project_documents in groups.items():
        if _normalize(project_name) in normalized_question:
            return _project_summary(project_name, project_documents)

    if _is_project_detail_request(question):
        if len(groups) == 1:
            only_project_name = next(iter(groups))
            return _project_summary(only_project_name, groups[only_project_name])

        clarification = _project_clarification_prompt(
            documents,
            "Mình đang thấy nhiều dự án trong kết quả hiện tại. Bạn muốn xem chi tiết dự án nào?",
        )
        if clarification:
            return clarification

    if len(groups) == 1 and any(key in normalized_question for key in ["du an", "cong trinh"]):
        only_project_name = next(iter(groups))
        return _project_summary(only_project_name, groups[only_project_name])

    if _is_project_list_request(question):
        location_hint = _extract_location_hint(question)
        items = []
        for project_name, project_documents in _ranked_project_groups(documents):
            metadata = max(project_documents, key=_ranking_value).metadata
            location = metadata.get("project_location")
            if location_hint and location and location_hint not in _normalize(location):
                continue
            items.append(_project_list_item(project_name, project_documents))

        items = _unique_names(items)
        if items:
            if _wants_single_project_suggestion(question):
                return (
                    "Mình gợi ý dự án này trong kết quả hiện tại:\n"
                    f"- {items[0]}\n"
                    "Nếu bạn muốn, mình có thể nói rõ hơn về địa điểm, diện tích hoặc chủ đầu tư của dự án này."
                )

            intro = "Mình tìm thấy một vài dự án nổi bật trong kết quả hiện tại:"
            if location_hint:
                intro = f"Mình tìm thấy một vài dự án phù hợp với khu vực {location_hint}:"
            return intro + "\n" + _format_bullets(items[:5])

        if location_hint:
            return f"Mình chưa thấy dự án phù hợp với khu vực {location_hint} trong kết quả hiện tại."

    return None


def compose_grounded_answer(question: str, documents: list[RetrievedDocument]) -> str | None:
    for builder in (
        _compose_media_answer,
        _compose_subjective_project_answer,
        _compose_project_category_answer,
        _compose_news_category_answer,
        _compose_style_answer,
        _compose_news_answer,
        _compose_project_answer,
        _compose_company_answer,
    ):
        answer = builder(question, documents)
        if answer:
            return answer
    return None
