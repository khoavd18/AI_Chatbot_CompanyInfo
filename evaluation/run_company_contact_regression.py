from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from company_contact_cases import COMPANY_CONTACT_CASES

from src.core.schema import RetrievedDocument
from src.llm.source_answer import _compose_company_answer


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


def _build_company_document() -> RetrievedDocument:
    return RetrievedDocument(
        id="company-contact-regression",
        score=1.0,
        text="\n".join(
            [
                "Ten cong ty: Nguyen Vo Dang Khoa Architects",
                "Slogan: Thiet ke chuan cong nang - thi cong chuan cam ket",
                "Gioi thieu ngan: Don vi thiet ke va thi cong nha pho, biet thu va cong trinh dich vu.",
                "Hotline: 0909.268.416",
                "Email: info@nvdkarchitects.vn",
                "Gio lam viec: Thu 2 - Thu 7, 8:00 - 18:00",
            ]
        ),
        metadata={
            "type": "company_info",
            "company_name": "Nguyen Vo Dang Khoa Architects",
            "company_main_address": "98/12 Nguyen Xi, Phuong 26, Quan Binh Thanh, TP.HCM",
            "company_website": "https://nguyen-vo-dang-khoa-architects.vercel.app",
        },
    )


def main() -> int:
    document = _build_company_document()
    failures = []

    for index, case in enumerate(COMPANY_CONTACT_CASES, start=1):
        answer = _compose_company_answer(case["question"], [document]) or ""
        passed = bool(answer) and _contains_all(answer, case["expected_keywords"]) and _contains_none(
            answer, case["forbidden_keywords"]
        )

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {index:02d}/{len(COMPANY_CONTACT_CASES)} {case['id']}")

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

    print(f"Ket qua: {len(COMPANY_CONTACT_CASES) - len(failures)}/{len(COMPANY_CONTACT_CASES)} case pass.")

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
