import logging
from typing import List

from src.core.schema import RetrievedDocument

logger = logging.getLogger("context_builder")


class ContextBuilder:
    def __init__(self, max_documents: int = 5, max_context_length: int = 3200, separator: str = "\n\n---\n\n"):
        self.max_documents = max_documents
        self.max_context_length = max_context_length
        self.separator = separator

    def _get_title(self, doc: RetrievedDocument) -> str:
        metadata = doc.metadata or {}
        return (
            metadata.get("project_name")
            or metadata.get("news_item_title")
            or metadata.get("company_name")
            or metadata.get("architecture_type_name")
            or metadata.get("interior_name")
            or metadata.get("project_category_name")
            or metadata.get("news_category_name")
            or metadata.get("hero_slide_title")
            or "Không rõ tiêu đề"
        )

    def _format_document(self, index: int, doc: RetrievedDocument) -> str:
        metadata = doc.metadata or {}
        source_type = metadata.get("type", "unknown")
        chunk_type = metadata.get("chunk_type", "unknown")
        title = self._get_title(doc)

        lines = [
            f"[{index}] Nguồn: {source_type}",
            f"Loại chunk: {chunk_type}",
            f"Tiêu đề: {title}",
            f"Nội dung: {doc.text.strip()}",
        ]
        return "\n".join(lines)

    def build(self, documents: List[RetrievedDocument]) -> str:
        if not documents:
            logger.warning("No documents provided to context builder.")
            return ""

        context_parts = []
        current_length = 0

        for idx, document in enumerate(documents[: self.max_documents], start=1):
            if not document.text or not document.text.strip():
                continue

            formatted = self._format_document(idx, document)

            if current_length + len(formatted) > self.max_context_length:
                remaining = self.max_context_length - current_length
                if remaining <= 0:
                    break
                formatted = formatted[:remaining]

            context_parts.append(formatted)
            current_length += len(formatted)

        context = self.separator.join(context_parts)

        logger.info(
            f"Built context with {len(context_parts)} documents, total length {len(context)} characters."
        )
        return context