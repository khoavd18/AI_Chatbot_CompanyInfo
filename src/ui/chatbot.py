from __future__ import annotations

import html
import os

import requests
import streamlit as st

from src.core.setting_loader import ensure_env_loaded

ensure_env_loaded()

DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
HISTORY_LIMIT = 6
QUICK_PROMPTS = [
    ("Thông tin công ty", "Giới thiệu ngắn về công ty"),
    ("Dự án nổi bật", "Cho tôi một vài dự án nổi bật"),
    ("Hình ảnh dự án", "Cho tôi hình ảnh của một dự án nổi bật"),
    ("Phong cách", "Công ty có những phong cách nội thất nào?"),
]
SOURCE_TYPE_LABELS = {
    "project": "Dự án",
    "company_info": "Công ty",
    "news": "Tin tức",
    "architecture_type": "Phong cách kiến trúc",
    "interior_style": "Phong cách nội thất",
    "hero_slide": "Hero slide",
    "project_category": "Danh mục dự án",
    "news_category": "Danh mục tin tức",
}

st.set_page_config(
    page_title="NVDK Architects Chatbot",
    page_icon="🤖",
    layout="wide",
)


def _repair_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    current = text
    for _ in range(3):
        changed = False
        for codec in ("cp1252", "latin1"):
            try:
                candidate = current.encode(codec).decode("utf-8")
            except Exception:
                continue
            if candidate != current:
                current = candidate
                changed = True
                break
        if not changed:
            break
    return current


def _sanitize_sources(sources: list[dict]) -> list[dict]:
    sanitized = []
    for source in sources or []:
        item = dict(source)
        for key in ("title", "snippet", "source_type", "chunk_type", "doc_id", "image_url", "video_url"):
            value = item.get(key)
            if isinstance(value, str):
                item[key] = _repair_text(value)
        sanitized.append(item)
    return sanitized


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    sanitized = []
    for message in messages or []:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = _repair_text(content)
        item["sources"] = _sanitize_sources(item.get("sources", []))
        item["show_media_preview"] = bool(item.get("show_media_preview", False))
        sanitized.append(item)
    return sanitized


def _safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f1e8;
            --surface: rgba(255, 251, 245, 0.78);
            --surface-strong: rgba(255, 251, 245, 0.92);
            --line: rgba(131, 104, 73, 0.18);
            --text: #1e1915;
            --muted: #6f665d;
            --teal: #1f6b66;
            --clay: #b86d3b;
            --sand: #efe2d1;
            --shadow: 0 18px 40px rgba(49, 34, 20, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(33, 119, 109, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(184, 109, 59, 0.12), transparent 26%),
                linear-gradient(180deg, #f9f5ee 0%, #f2eadf 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1080px;
            padding-top: 1.35rem;
            padding-bottom: 3rem;
        }

        body, [class*="css"], .stMarkdown, .stTextInput input, .stChatInput input, .stSelectbox, .stButton button {
            font-family: Aptos, "Segoe UI", sans-serif;
        }

        code, pre, .stCode {
            font-family: Consolas, "IBM Plex Mono", monospace;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 249, 241, 0.96), rgba(242, 232, 218, 0.96));
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(255, 252, 247, 0.94), rgba(244, 234, 219, 0.90)),
                linear-gradient(135deg, rgba(31, 107, 102, 0.08), rgba(184, 109, 59, 0.08));
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 1.4rem 1.5rem 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -40px -60px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(31, 107, 102, 0.18), transparent 68%);
            pointer-events: none;
        }

        .hero-kicker {
            display: inline-block;
            font-size: 0.76rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--teal);
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-size: 2rem;
            line-height: 1.08;
            font-weight: 700;
            color: var(--text);
            max-width: 720px;
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            max-width: 760px;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.65;
            margin-bottom: 1rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .hero-chip, .metric-pill, .meta-pill, .latency-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.42rem 0.72rem;
            font-size: 0.79rem;
            line-height: 1;
            border: 1px solid rgba(111, 102, 93, 0.16);
            background: rgba(255, 255, 255, 0.70);
            color: #403832;
        }

        .hero-chip strong, .metric-pill strong {
            font-weight: 700;
            color: var(--text);
        }

        .sidebar-card {
            background: rgba(255, 251, 245, 0.82);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: 0 10px 24px rgba(49, 34, 20, 0.08);
            padding: 1rem 1rem 0.9rem 1rem;
            margin-bottom: 1rem;
        }

        .sidebar-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 0.96rem;
            color: var(--text);
        }

        .sidebar-card p {
            margin: 0;
            font-size: 0.86rem;
            line-height: 1.55;
            color: var(--muted);
        }

        .section-label {
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--clay);
            font-weight: 700;
            margin: 0.6rem 0 0.65rem 0;
        }

        .conversation-label {
            margin-top: 0.25rem;
        }

        div[data-testid="stChatMessage"] {
            background: var(--surface);
            border: 1px solid rgba(131, 104, 73, 0.14);
            border-radius: 24px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 14px 28px rgba(49, 34, 20, 0.07);
            backdrop-filter: blur(8px);
            margin-bottom: 0.95rem;
        }

        div[data-testid="stChatMessage"] p {
            font-size: 0.98rem;
            line-height: 1.7;
        }

        .latency-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.35rem 0 0.2rem 0;
        }

        .latency-pill {
            background: rgba(255, 255, 255, 0.84);
        }

        .latency-pill.primary {
            background: rgba(31, 107, 102, 0.12);
            color: #154e4a;
            border-color: rgba(31, 107, 102, 0.14);
        }

        .latency-pill.soft {
            background: rgba(184, 109, 59, 0.10);
            color: #844a24;
            border-color: rgba(184, 109, 59, 0.14);
        }

        .source-card {
            background: var(--surface-strong);
            border: 1px solid rgba(131, 104, 73, 0.14);
            border-radius: 22px;
            box-shadow: 0 14px 30px rgba(49, 34, 20, 0.07);
            padding: 0.95rem 1rem 0.9rem 1rem;
            margin: 0.75rem 0 1rem 0;
        }

        .source-heading {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 0.65rem;
        }

        .source-index {
            width: 1.9rem;
            height: 1.9rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(31, 107, 102, 0.12);
            color: #154e4a;
            font-weight: 700;
            font-size: 0.83rem;
            flex: 0 0 auto;
        }

        .source-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text);
            line-height: 1.35;
        }

        .source-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-bottom: 0.65rem;
        }

        .meta-pill {
            background: rgba(255, 255, 255, 0.82);
        }

        .meta-pill.type {
            background: rgba(31, 107, 102, 0.12);
            color: #154e4a;
            border-color: rgba(31, 107, 102, 0.14);
        }

        .meta-pill.score {
            background: rgba(184, 109, 59, 0.10);
            color: #844a24;
            border-color: rgba(184, 109, 59, 0.14);
        }

        .source-snippet {
            color: var(--muted);
            line-height: 1.68;
            font-size: 0.93rem;
        }

        .media-link {
            margin-top: 0.45rem;
            font-size: 0.85rem;
        }

        .media-link a {
            color: var(--teal);
            text-decoration: none;
            font-weight: 600;
        }

        .media-link a:hover {
            text-decoration: underline;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(131, 104, 73, 0.18);
            background: rgba(255, 252, 247, 0.82);
            color: var(--text);
            font-weight: 600;
            box-shadow: none;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            border-color: rgba(31, 107, 102, 0.28);
            color: #154e4a;
            background: rgba(255, 255, 255, 0.96);
        }

        .stTextInput > div > div > input,
        div[data-testid="stChatInput"] textarea,
        div[data-testid="stChatInput"] input {
            border-radius: 18px !important;
            border: 1px solid rgba(131, 104, 73, 0.18) !important;
            background: rgba(255, 252, 247, 0.92) !important;
        }

        .stExpander {
            background: rgba(255, 251, 245, 0.58);
            border: 1px solid rgba(131, 104, 73, 0.12);
            border-radius: 18px;
        }

        @media (max-width: 768px) {
            .hero-title {
                font-size: 1.55rem;
            }

            .block-container {
                padding-top: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: str | None) -> str:
    return html.escape(_repair_text(value or ""))


def _format_source_type(source_type: str | None) -> str:
    return SOURCE_TYPE_LABELS.get(source_type or "", _repair_text(source_type or "Nguồn"))


def _format_score_pills(scores: dict) -> str:
    pills = []
    for key in ("hybrid", "dense", "bm25", "rerank"):
        value = scores.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        pills.append(f'<span class="meta-pill score">{key}: {numeric:.3f}</span>')
    return "".join(pills)


def _render_source_image(image_url: str, caption: str):
    try:
        st.image(image_url, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image_url, caption=caption, use_column_width=True)


def _render_debug_table(debug_rows):
    try:
        st.dataframe(debug_rows, use_container_width=True)
    except TypeError:
        st.dataframe(debug_rows)


def _collect_media_from_sources(sources: list[dict]) -> tuple[list[str], list[str]]:
    image_urls: list[str] = []
    video_urls: list[str] = []
    seen_images = set()
    seen_videos = set()

    for source in _sanitize_sources(sources):
        image_url = source.get("image_url")
        if image_url and image_url not in seen_images:
            seen_images.add(image_url)
            image_urls.append(image_url)

        video_url = source.get("video_url")
        if video_url and video_url not in seen_videos:
            seen_videos.add(video_url)
            video_urls.append(video_url)

    return image_urls, video_urls


def render_media_preview(sources: list[dict]):
    image_urls, video_urls = _collect_media_from_sources(sources)
    if not image_urls and not video_urls:
        return

    st.caption("Media preview")

    for image_url in image_urls[:3]:
        _render_source_image(image_url, caption="Ảnh liên quan")

    for video_url in video_urls[:2]:
        try:
            st.video(video_url)
        except Exception:
            st.markdown(
                f'<div class="media-link"><a href="{html.escape(video_url)}" target="_blank">Mở video</a></div>',
                unsafe_allow_html=True,
            )


def call_health(api_base: str):
    response = requests.get(f"{api_base.rstrip('/')}/health", timeout=15)
    response.raise_for_status()
    return response.json()


def build_history_payload(messages: list[dict], limit: int = HISTORY_LIMIT) -> list[dict]:
    payload = []
    for message in messages[-limit:]:
        content = (message.get("content") or "").strip()
        role = message.get("role")
        if role not in {"user", "assistant"} or not content:
            continue

        history_item = {
            "role": role,
            "content": content,
            "sources": [],
        }

        for source in message.get("sources", [])[:3]:
            title = (source.get("title") or "").strip()
            if not title:
                continue
            history_item["sources"].append(
                {
                    "title": title,
                    "source_type": source.get("source_type"),
                    "image_url": source.get("image_url"),
                    "video_url": source.get("video_url"),
                }
            )

        payload.append(history_item)

    return payload


def call_chat(api_base: str, query: str, show_debug: bool, history: list[dict], session_id: str | None = None):
    payload = {
        "query": query,
        "debug": show_debug,
        "history": history,
    }
    if session_id:
        payload["session_id"] = session_id

    response = requests.post(
        f"{api_base.rstrip('/')}/api/chat",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def render_latency(latency: dict, request_id: str | None):
    if not latency:
        return

    pills = []
    if latency.get("total_ms") is not None:
        pills.append(f'<span class="latency-pill primary">total {float(latency["total_ms"]):.1f}ms</span>')
    if latency.get("retrieval_ms") is not None:
        pills.append(f'<span class="latency-pill">retrieval {float(latency["retrieval_ms"]):.1f}ms</span>')
    if latency.get("rerank_ms") is not None:
        pills.append(f'<span class="latency-pill">rerank {float(latency["rerank_ms"]):.1f}ms</span>')
    if latency.get("answer_ms") is not None:
        pills.append(f'<span class="latency-pill">answer {float(latency["answer_ms"]):.1f}ms</span>')
    if request_id:
        pills.append(f'<span class="latency-pill soft">request {html.escape(str(request_id)[:8])}</span>')

    if pills:
        st.markdown(f'<div class="latency-row">{"".join(pills)}</div>', unsafe_allow_html=True)


def render_sources(sources):
    if not sources:
        st.write("Không có source.")
        return

    for item in _sanitize_sources(sources):
        title = item.get("title") or "Không rõ tiêu đề"
        snippet = item.get("snippet") or ""
        source_type = _format_source_type(item.get("source_type"))
        chunk_type = _repair_text(item.get("chunk_type") or "unknown")
        doc_id = _repair_text(item.get("doc_id") or "")
        scores = item.get("scores", {}) or {}
        meta_pills = [
            f'<span class="meta-pill type">{_escape(source_type)}</span>',
            f'<span class="meta-pill">chunk: {_escape(chunk_type)}</span>',
        ]
        if doc_id:
            meta_pills.append(f'<span class="meta-pill">doc: {_escape(doc_id)}</span>')

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-heading">
                    <div class="source-index">{html.escape(str(item.get('index', '?')))}</div>
                    <div class="source-title">{_escape(title)}</div>
                </div>
                <div class="source-meta">
                    {''.join(meta_pills)}
                    {_format_score_pills(scores)}
                </div>
                <div class="source-snippet">{_escape(snippet)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        image_url = item.get("image_url")
        if image_url:
            _render_source_image(image_url, caption="Hình từ source")
            st.markdown(
                f'<div class="media-link"><a href="{html.escape(image_url)}" target="_blank">Mở ảnh gốc</a></div>',
                unsafe_allow_html=True,
            )

        video_url = item.get("video_url")
        if video_url:
            st.markdown(
                f'<div class="media-link"><a href="{html.escape(video_url)}" target="_blank">Mở video</a></div>',
                unsafe_allow_html=True,
            )


def render_debug(debug_rows):
    if not debug_rows:
        st.write("Backend hiện không trả dữ liệu debug.")
        return

    _render_debug_table(debug_rows)


def render_hero(messages: list[dict]):
    assistant_turns = max(sum(1 for message in messages if message.get("role") == "assistant") - 1, 0)
    source_count = sum(len(message.get("sources", [])) for message in messages if message.get("role") == "assistant")
    last_latency = {}
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("latency"):
            last_latency = message.get("latency", {})
            break

    hero_chips = [
        f'<span class="hero-chip"><strong>{assistant_turns}</strong> lượt trả lời</span>',
        f'<span class="hero-chip"><strong>{source_count}</strong> nguồn đã dùng</span>',
        f'<span class="hero-chip"><strong>{int(float(last_latency.get("total_ms", 0))) if last_latency else 0}ms</strong> phản hồi gần nhất</span>',
    ]

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">NVDK Architects • Grounded RAG Assistant</div>
            <div class="hero-title">Tra cứu dự án, phong cách và hình ảnh theo cách dễ đọc hơn.</div>
            <div class="hero-subtitle">
                Giao diện này ưu tiên ba việc: câu trả lời ngắn gọn, nguồn rõ ràng và media hiển thị trực tiếp.
                Bạn có thể hỏi tiếp nối như “dự án đó”, “chi tiết hơn”, hoặc “hình ảnh đâu?”.
            </div>
            <div class="chip-row">
                {''.join(hero_chips)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_prompts():
    st.markdown('<div class="section-label">Gợi ý hỏi nhanh</div>', unsafe_allow_html=True)
    for row_start in range(0, len(QUICK_PROMPTS), 2):
        columns = st.columns(2)
        for offset, (label, prompt) in enumerate(QUICK_PROMPTS[row_start : row_start + 2]):
            index = row_start + offset
            if columns[offset].button(label, key=f"quick_prompt_{index}"):
                st.session_state.pending_query = prompt
                _safe_rerun()


def render_sidebar(messages: list[dict]):
    assistant_turns = max(sum(1 for message in messages if message.get("role") == "assistant") - 1, 0)
    user_turns = sum(1 for message in messages if message.get("role") == "user")

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <h4>NVDK Architects Chatbot</h4>
                <p>Thiết kế chuẩn công năng, thi công chuẩn cam kết. Bản UI này ưu tiên đọc nhanh và đối chiếu source dễ hơn.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        left.markdown(f'<div class="metric-pill"><strong>{user_turns}</strong> câu hỏi</div>', unsafe_allow_html=True)
        right.markdown(f'<div class="metric-pill"><strong>{assistant_turns}</strong> phản hồi</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Kết nối</div>', unsafe_allow_html=True)
        api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE)
        show_debug = st.checkbox("Hiện debug retrieval", value=True)

        button_col_1, button_col_2 = st.columns(2)
        if button_col_1.button("Check health"):
            try:
                st.session_state.health_payload = call_health(api_base)
                st.session_state.health_error = None
            except Exception as exc:
                st.session_state.health_error = str(exc)
                st.session_state.health_payload = None

        if button_col_2.button("Xóa chat"):
            st.session_state.messages = []
            st.session_state.chat_session_id = None
            st.session_state.pending_query = None
            _safe_rerun()

        health_payload = st.session_state.get("health_payload")
        health_error = st.session_state.get("health_error")
        if health_payload:
            st.success("API đang hoạt động")
            with st.expander("Xem health payload", expanded=False):
                st.json(health_payload)
        elif health_error:
            st.error(f"Không kết nối được API: {health_error}")

        st.markdown('<div class="section-label">Mẹo dùng</div>', unsafe_allow_html=True)
        st.markdown(
            "- Hỏi theo chuỗi: dự án nổi bật -> chi tiết hơn -> hình ảnh đâu\n"
            "- Bật debug khi muốn xem retrieval và rerank\n"
            "- Nếu dữ liệu mới vừa ingest, hãy refresh UI rồi hỏi lại"
        )

    return api_base, show_debug


def process_query(user_query: str, api_base: str, show_debug: bool):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
            "sources": [],
            "show_media_preview": False,
            "debug": [],
            "latency": {},
            "request_id": None,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            history_payload = build_history_payload(st.session_state.messages[:-1])

            with st.spinner("Đang hỏi backend RAG..."):
                result = call_chat(
                    api_base=api_base,
                    query=user_query,
                    show_debug=show_debug,
                    history=history_payload,
                    session_id=st.session_state.chat_session_id,
                )

            answer = _repair_text(result.get("answer", "Không có câu trả lời."))
            sources = _sanitize_sources(result.get("sources", []))
            debug_rows = result.get("debug", [])
            returned_session_id = result.get("session_id")
            request_id = result.get("request_id")
            latency = result.get("latency", {})
            show_media_preview = bool(result.get("show_media_preview", False))

            if returned_session_id:
                st.session_state.chat_session_id = returned_session_id

            st.markdown(answer)
            if show_media_preview:
                render_media_preview(sources)
            render_latency(latency, request_id)

            with st.expander("Nguồn đã dùng", expanded=False):
                render_sources(sources)

            if show_debug:
                with st.expander("Debug retrieval", expanded=False):
                    render_debug(debug_rows)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "show_media_preview": show_media_preview,
                    "debug": debug_rows,
                    "latency": latency,
                    "request_id": request_id,
                }
            )

        except requests.HTTPError as exc:
            error_text = _repair_text(f"Lỗi HTTP từ backend: {exc}")
            try:
                error_payload = exc.response.json()
                error_text += f"\n\nChi tiết: {error_payload}"
            except Exception:
                pass

            st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                    "show_media_preview": False,
                    "debug": [],
                    "latency": {},
                    "request_id": None,
                }
            )

        except Exception as exc:
            error_text = _repair_text(f"Lỗi khi gọi backend: {exc}")
            st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                    "show_media_preview": False,
                    "debug": [],
                    "latency": {},
                    "request_id": None,
                }
            )


_inject_styles()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn, hãy hỏi tôi về dữ liệu trong hệ thống RAG của bạn.",
            "sources": [],
            "show_media_preview": False,
            "debug": [],
            "latency": {},
            "request_id": None,
        }
    ]
else:
    st.session_state.messages = _sanitize_messages(st.session_state.messages)

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "health_payload" not in st.session_state:
    st.session_state.health_payload = None

if "health_error" not in st.session_state:
    st.session_state.health_error = None

api_base, show_debug = render_sidebar(st.session_state.messages)
render_hero(st.session_state.messages)
render_quick_prompts()
st.markdown('<div class="section-label conversation-label">Cuộc hội thoại</div>', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(_repair_text(message["content"]))

        if message["role"] == "assistant":
            if message.get("show_media_preview", False):
                render_media_preview(message.get("sources", []))
            render_latency(message.get("latency", {}), message.get("request_id"))
            with st.expander("Nguồn đã dùng", expanded=False):
                render_sources(message.get("sources", []))

            if show_debug:
                with st.expander("Debug retrieval", expanded=False):
                    render_debug(message.get("debug", []))

chat_input_value = st.chat_input("Nhập câu hỏi của bạn...")
user_query = chat_input_value or st.session_state.get("pending_query")

if user_query:
    st.session_state.pending_query = None
    process_query(user_query=user_query, api_base=api_base, show_debug=show_debug)
