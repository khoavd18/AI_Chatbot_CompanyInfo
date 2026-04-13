from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.routes.chat import _grounded_only_message, _llm_fallback_policy
from src.core.schema import RetrievedDocument


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _normalize(text: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", text or "")
        if not unicodedata.combining(character)
    )
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    return " ".join(normalized.lower().split())


def _contains_all(text: str, keywords: list[str]) -> bool:
    normalized_text = _normalize(text)
    return all(_normalize(keyword) in normalized_text for keyword in keywords)


def main() -> int:
    project_docs = [
        RetrievedDocument(id="p1", score=0.9, text="project", metadata={"type": "project", "project_name": "Nhà vườn Bình Tân"}),
        RetrievedDocument(
            id="p2",
            score=0.8,
            text="category",
            metadata={"type": "project_category", "project_category_name": "Nhà phố"},
        ),
    ]
    news_docs = [
        RetrievedDocument(id="n1", score=0.9, text="news", metadata={"type": "news", "news_item_title": "Bài viết A"}),
        RetrievedDocument(
            id="n2",
            score=0.8,
            text="category",
            metadata={"type": "news_category", "news_category_name": "Tin công ty"},
        ),
    ]
    style_docs = [
        RetrievedDocument(
            id="s1",
            score=0.9,
            text="style",
            metadata={"type": "interior_style", "interior_name": "Japandi style"},
        )
    ]
    company_docs = [
        RetrievedDocument(
            id="c1",
            score=0.9,
            text="company",
            metadata={"type": "company_info", "company_name": "Nguyen Vo Dang Khoa Architects"},
        )
    ]

    cases = [
        {
            "id": "company-blocked",
            "question": "dia chi cong ty o dau",
            "documents": company_docs,
            "expected_policy": (False, "company", "intent_blocked"),
            "expected_message_keywords": ["thong tin cong ty/fact", "khong dung llm fallback"],
        },
        {
            "id": "project-allowed",
            "question": "du an cua cong ty la gi",
            "documents": project_docs,
            "expected_policy": (True, "project", "allowed"),
            "expected_message_keywords": [],
        },
        {
            "id": "news-allowed",
            "question": "co bai viet nao moi khong",
            "documents": news_docs,
            "expected_policy": (True, "news", "allowed"),
            "expected_message_keywords": [],
        },
        {
            "id": "style-needs-more-evidence",
            "question": "phong cach japandi la gi",
            "documents": style_docs,
            "expected_policy": (False, "style", "insufficient_evidence"),
            "expected_message_keywords": ["chua du evidence", "ten phong cach"],
        },
    ]

    failures = []
    for index, case in enumerate(cases, start=1):
        policy = _llm_fallback_policy(case["question"], case["documents"])
        message = _grounded_only_message(policy[1], policy[2])
        passed = policy == case["expected_policy"] and (
            not case["expected_message_keywords"] or _contains_all(message, case["expected_message_keywords"])
        )

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {index:02d}/{len(cases)} {case['id']}")
        if not passed:
            failures.append(
                {
                    "id": case["id"],
                    "policy": policy,
                    "expected_policy": case["expected_policy"],
                    "message": message,
                    "expected_message_keywords": case["expected_message_keywords"],
                }
            )

    print(f"Ket qua: {len(cases) - len(failures)}/{len(cases)} case pass.")

    if failures:
        print("\nChi tiet fail:")
        for failure in failures:
            print(f"- {failure['id']}: policy={failure['policy']} expected={failure['expected_policy']}")
            print(f"  message={failure['message']!r}")
            print(f"  expected_message_keywords={failure['expected_message_keywords']}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
