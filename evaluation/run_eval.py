from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

from eval_cases import EVAL_CASES


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_all(text: str, keywords: list[str]) -> bool:
    normalized_text = _normalize(text)
    return all(_normalize(keyword) in normalized_text for keyword in keywords)


def _match_any_partial(candidates: list[str], expected_values: list[str]) -> bool:
    if not expected_values:
        return True

    normalized_candidates = [_normalize(candidate) for candidate in candidates]
    for expected in expected_values:
        normalized_expected = _normalize(expected)
        if any(normalized_expected in candidate for candidate in normalized_candidates):
            return True
    return False


def _validate_cases() -> list[str]:
    errors = []
    ids = set()

    if not 30 <= len(EVAL_CASES) <= 50:
        errors.append(f"Dataset phải có 30-50 câu, hiện tại là {len(EVAL_CASES)}.")

    for index, case in enumerate(EVAL_CASES, start=1):
        case_id = case.get("id")
        question = case.get("question")
        if not case_id:
            errors.append(f"Case #{index} thiếu id.")
            continue
        if case_id in ids:
            errors.append(f"Trùng id: {case_id}")
        ids.add(case_id)
        if not question:
            errors.append(f"Case {case_id} thiếu question.")

    return errors


def _call_api(api_base: str, question: str, debug: bool) -> dict:
    response = requests.post(
        f"{api_base.rstrip('/')}/api/chat",
        json={"query": question, "debug": debug},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def _evaluate_case(case: dict, result: dict) -> dict:
    answer = result.get("answer", "")
    sources = result.get("sources", []) or []
    source_titles = [source.get("title", "") for source in sources]
    source_types = [source.get("source_type", "") for source in sources]

    answer_ok = _contains_all(answer, case.get("expected_answer_keywords", []))
    source_type_ok = _match_any_partial(source_types, case.get("expected_source_types", []))
    source_title_ok = _match_any_partial(source_titles, case.get("expected_source_titles", []))

    passed = answer_ok and source_type_ok and source_title_ok
    return {
        "id": case["id"],
        "question": case["question"],
        "passed": passed,
        "answer_ok": answer_ok,
        "source_type_ok": source_type_ok,
        "source_title_ok": source_title_ok,
        "latency_ms": (result.get("latency") or {}).get("total_ms"),
        "request_id": result.get("request_id"),
        "answer_preview": answer[:200],
        "source_titles": source_titles,
        "source_types": source_types,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation against the chat API.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000", help="Base URL của FastAPI backend")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số case để chạy")
    parser.add_argument("--debug", action="store_true", help="Yêu cầu backend trả debug rows")
    parser.add_argument("--validate-only", action="store_true", help="Chỉ validate dataset, không gọi API")
    parser.add_argument("--output", help="Ghi report JSON ra file")
    args = parser.parse_args()

    validation_errors = _validate_cases()
    if validation_errors:
        for error in validation_errors:
            print(f"[INVALID] {error}")
        return 1

    print(f"Dataset hợp lệ với {len(EVAL_CASES)} câu.")
    if args.validate_only:
        return 0

    cases = EVAL_CASES[: args.limit] if args.limit and args.limit > 0 else EVAL_CASES
    report = []

    for index, case in enumerate(cases, start=1):
        try:
            result = _call_api(args.api_base, case["question"], args.debug)
            evaluation = _evaluate_case(case, result)
            report.append(evaluation)
            status = "PASS" if evaluation["passed"] else "FAIL"
            latency = evaluation.get("latency_ms")
            latency_text = f" latency={latency:.1f}ms" if isinstance(latency, (int, float)) else ""
            print(f"[{status}] {index:02d}/{len(cases)} {case['id']}{latency_text}")
        except Exception as exc:
            print(f"[ERROR] {index:02d}/{len(cases)} {case['id']}: {exc}")
            report.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "passed": False,
                    "error": str(exc),
                }
            )

    passed = sum(1 for item in report if item.get("passed"))
    print(f"Kết quả: {passed}/{len(report)} case pass.")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã ghi report vào {output_path}")

    return 0 if passed == len(report) else 1


if __name__ == "__main__":
    sys.exit(main())
