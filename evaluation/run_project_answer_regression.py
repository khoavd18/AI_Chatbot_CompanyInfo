from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_answer_regression_cases import PROJECT_ANSWER_REGRESSION_CASES

from src.core.schema import RetrievedDocument
from src.llm.source_answer import _compose_project_answer


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


def _contains_none(text: str, keywords: list[str]) -> bool:
    normalized_text = _normalize(text)
    return all(_normalize(keyword) not in normalized_text for keyword in keywords)


def _build_project_documents() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            id="project-1",
            score=0.96,
            text="\n".join(
                [
                    "Tên dự án: Nhà vườn Bình Tân",
                    "Mô tả ngắn: Nhà vườn hiện đại cho gia đình trẻ.",
                    "Địa điểm: Bình Tân, TP.HCM",
                    "Chủ đầu tư: Anh Minh",
                ]
            ),
            metadata={
                "type": "project",
                "project_name": "Nhà vườn Bình Tân",
                "project_location": "Bình Tân, TP.HCM",
                "project_category_name": "Nhà phố",
                "chunk_type": "overview",
            },
        ),
        RetrievedDocument(
            id="project-2",
            score=0.92,
            text="\n".join(
                [
                    "Tên dự án: Resort Hải Vân 2",
                    "Mô tả ngắn: Khu nghỉ dưỡng ven biển.",
                    "Địa điểm: Hải Vân, Đà Nẵng",
                    "Chủ đầu tư: Công ty du lịch ABC",
                ]
            ),
            metadata={
                "type": "project",
                "project_name": "Resort Hải Vân 2",
                "project_location": "Hải Vân, Đà Nẵng",
                "project_category_name": "Khách sạn & Resort",
                "chunk_type": "overview",
            },
        ),
        RetrievedDocument(
            id="project-3",
            score=0.89,
            text="\n".join(
                [
                    "Tên dự án: Nhà phố hiện đại 5x20m",
                    "Mô tả ngắn: Nhà phố 3 tầng hiện đại.",
                    "Địa điểm: Bình Tân, TP.HCM",
                    "Chủ đầu tư: Chị Lan",
                ]
            ),
            metadata={
                "type": "project",
                "project_name": "Nhà phố hiện đại 5x20m",
                "project_location": "Bình Tân, TP.HCM",
                "project_category_name": "Nhà phố",
                "chunk_type": "overview",
            },
        ),
    ]


def main() -> int:
    documents = _build_project_documents()
    failures = []

    for index, case in enumerate(PROJECT_ANSWER_REGRESSION_CASES, start=1):
        answer = _compose_project_answer(case["question"], documents) or ""
        passed = bool(answer) and _contains_all(answer, case["expected_keywords"]) and _contains_none(
            answer, case["forbidden_keywords"]
        )

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {index:02d}/{len(PROJECT_ANSWER_REGRESSION_CASES)} {case['id']}")

        if not passed:
            failures.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "answer": answer,
                    "expected_keywords": case["expected_keywords"],
                    "forbidden_keywords": case["forbidden_keywords"],
                }
            )

    print(
        f"Ket qua: {len(PROJECT_ANSWER_REGRESSION_CASES) - len(failures)}/{len(PROJECT_ANSWER_REGRESSION_CASES)} case pass."
    )

    if failures:
        print("\nChi tiet fail:")
        for failure in failures:
            print(f"- {failure['id']}: {failure['question']}")
            print(f"  answer={failure['answer']!r}")
            print(f"  expected={failure['expected_keywords']}")
            print(f"  forbidden={failure['forbidden_keywords']}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
