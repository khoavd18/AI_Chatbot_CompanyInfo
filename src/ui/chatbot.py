from __future__ import annotations

import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Internal Support Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Internal Support Copilot")
st.caption("UI riêng cho local/offline RAG chatbot")


def call_health(api_base: str):
    response = requests.get(f"{api_base.rstrip('/')}/health", timeout=15)
    response.raise_for_status()
    return response.json()


def call_chat(api_base: str, query: str, session_id: str | None = None):
    payload = {
        "query": query,
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


def render_sources(sources):
    if not sources:
        st.write("Không có source.")
        return

    for idx, item in enumerate(sources, start=1):
        # New API shape: {text, metadata, score}
        if isinstance(item, dict) and "metadata" in item:
            metadata = item.get("metadata", {}) or {}
            title = (
                metadata.get("title")
                or metadata.get("name")
                or metadata.get("project_title")
                or metadata.get("news_title")
                or f"Document {idx}"
            )
            source = metadata.get("source") or metadata.get("type") or "unknown"
            doc_id = metadata.get("chunk_id") or metadata.get("id") or ""
            dense_score = item.get("score", "")
            preview = item.get("text", "")

            st.markdown(f"**[{idx}] {title}**")
            st.caption(
                f"source={source} | doc_id={doc_id} | score={dense_score}"
            )
            if preview:
                st.write(preview)
            continue

        # Old API shape: {index, title, source, doc_id, rerank_score}
        if isinstance(item, dict):
            st.markdown(f"**[{item.get('index', idx)}] {item.get('title', 'Không có tiêu đề')}**")
            st.caption(
                f"source={item.get('source', '')} | "
                f"doc_id={item.get('doc_id', '')} | "
                f"rerank_score={item.get('rerank_score', '')}"
            )
            continue

        st.write(item)


def render_debug(debug_rows):
    if not debug_rows:
        st.write("Backend hiện không trả dữ liệu debug.")
        return

    st.dataframe(debug_rows, use_container_width=True)


with st.sidebar:
    st.subheader("Cấu hình kết nối")

    api_base = st.text_input(
        "API Base URL",
        value=DEFAULT_API_BASE,
    )

    show_debug = st.checkbox(
        "Hiện vùng debug retrieval (nếu backend có trả)",
        value=True,
    )

    st.subheader("Kiểm tra API")
    if st.button("Check /health"):
        try:
            health = call_health(api_base)
            st.success("API đang hoạt động")
            st.json(health)
        except Exception as exc:
            st.error(f"Không kết nối được API: {exc}")

    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = []
        st.session_state.chat_session_id = None
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn, hãy hỏi tôi về dữ liệu trong hệ thống RAG của bạn.",
            "sources": [],
            "debug": [],
        }
    ]

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            with st.expander("Nguồn đã dùng", expanded=False):
                render_sources(message.get("sources", []))

            if show_debug:
                with st.expander("Debug retrieval", expanded=False):
                    render_debug(message.get("debug", []))


user_query = st.chat_input("Nhập câu hỏi của bạn...")

if user_query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
            "sources": [],
            "debug": [],
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang hỏi backend RAG..."):
                result = call_chat(
                    api_base=api_base,
                    query=user_query,
                    session_id=st.session_state.chat_session_id,
                )

            answer = result.get("answer", "Không có câu trả lời.")
            sources = result.get("sources", [])
            debug_rows = result.get("debug", [])
            returned_session_id = result.get("session_id")

            if returned_session_id:
                st.session_state.chat_session_id = returned_session_id

            st.markdown(answer)

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
                    "debug": debug_rows,
                }
            )

        except requests.HTTPError as exc:
            error_text = f"Lỗi HTTP từ backend: {exc}"
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
                    "debug": [],
                }
            )

        except Exception as exc:
            error_text = f"Lỗi khi gọi backend: {exc}"
            st.error(error_text)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                    "debug": [],
                }
            )