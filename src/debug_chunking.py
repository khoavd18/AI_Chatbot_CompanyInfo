from collections import Counter, defaultdict
from statistics import mean

from src.rag.chunking.architectureType import chunk_architecture_types
from src.rag.chunking.companyInfo import chunk_company_info
from src.rag.chunking.interiorStyles import chunk_interior_styles
from src.rag.chunking.newsCategories import chunk_news_categories
from src.rag.chunking.news import chunk_news
from src.rag.chunking.projectCategories import chunk_project_categories
from src.rag.chunking.projects import chunk_projects
from src.rag.chunking.heroSlides import chunk_hero_slides


CHUNKERS = [
    ("architectureTypes", chunk_architecture_types),
    ("companyInfo", chunk_company_info),
    ("interiorStyles", chunk_interior_styles),
    ("newsCategories", chunk_news_categories),
    ("news", chunk_news),
    ("projectCategories", chunk_project_categories),
    ("projects", chunk_projects),
    ("heroSlides", chunk_hero_slides),
]


REQUIRED_METADATA_BY_TYPE = {
    "architectureTypes": ["type", "source", "chunk_id", "chunk_type", "architecture_type_name"],
    "companyInfo": ["type", "source", "chunk_id", "chunk_type", "company_name"],
    "interiorStyles": ["type", "source", "chunk_id", "chunk_type", "interior_name"],
    "newsCategories": ["type", "source", "chunk_id", "chunk_type", "news_category_name"],
    "news": ["type", "source", "chunk_id", "chunk_type", "news_item_title"],
    "projectCategories": ["type", "source", "chunk_id", "chunk_type", "project_category_name"],
    "projects": ["type", "source", "chunk_id", "chunk_type", "project_name"],
    "heroSlides": ["type", "source", "chunk_id", "chunk_type", "hero_slide_title"],
}


def _safe_len(text):
    return len(text.strip()) if isinstance(text, str) else 0


def _truncate(text, max_len=180):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + ("..." if len(text) > max_len else "")


def analyze_source(source_name, chunks):
    print("=" * 100)
    print(f"SOURCE: {source_name}")
    print(f"TOTAL CHUNKS: {len(chunks)}")

    if not chunks:
        print("No chunks found.\n")
        return

    type_counter = Counter()
    text_lengths = []
    missing_metadata_counter = Counter()
    too_short = []
    too_long = []

    examples_by_type = defaultdict(list)

    required_fields = REQUIRED_METADATA_BY_TYPE.get(source_name, ["type", "source", "chunk_id", "chunk_type"])

    for chunk in chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {}) or {}

        chunk_type = metadata.get("chunk_type", "UNKNOWN")
        type_counter[chunk_type] += 1

        length = _safe_len(text)
        text_lengths.append(length)

        for field in required_fields:
            value = metadata.get(field)
            if value is None or value == "":
                missing_metadata_counter[field] += 1

        if length < 40:
            too_short.append(chunk)

        if length > 800:
            too_long.append(chunk)

        if len(examples_by_type[chunk_type]) < 2:
            examples_by_type[chunk_type].append(chunk)

    print("\nChunk type distribution:")
    for chunk_type, count in sorted(type_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {chunk_type}: {count}")

    print("\nText length stats:")
    print(f"  - min: {min(text_lengths)}")
    print(f"  - max: {max(text_lengths)}")
    print(f"  - avg: {mean(text_lengths):.2f}")

    print("\nMissing metadata summary:")
    if not missing_metadata_counter:
        print("  - No missing required metadata fields")
    else:
        for field, count in missing_metadata_counter.items():
            print(f"  - {field}: missing in {count} chunks")

    print(f"\nToo short chunks (<40 chars): {len(too_short)}")
    for i, chunk in enumerate(too_short[:3], start=1):
        md = chunk.get('metadata', {})
        print(f"  [{i}] type={md.get('chunk_type')} | text={_truncate(chunk.get('text', ''))}")

    print(f"\nToo long chunks (>800 chars): {len(too_long)}")
    for i, chunk in enumerate(too_long[:3], start=1):
        md = chunk.get('metadata', {})
        print(f"  [{i}] type={md.get('chunk_type')} | len={_safe_len(chunk.get('text', ''))}")

    print("\nExamples by chunk type:")
    for chunk_type, examples in sorted(examples_by_type.items()):
        print(f"  * {chunk_type}")
        for idx, chunk in enumerate(examples, start=1):
            print(f"    [{idx}] {_truncate(chunk.get('text', ''))}")

    print()


def main():
    all_total = 0

    for source_name, chunk_func in CHUNKERS:
        try:
            chunks = chunk_func() or []
            all_total += len(chunks)
            analyze_source(source_name, chunks)
        except Exception as e:
            print("=" * 100)
            print(f"SOURCE: {source_name}")
            print(f"FAILED: {e}\n")

    print("=" * 100)
    print(f"GRAND TOTAL CHUNKS: {all_total}")
    print("=" * 100)


if __name__ == "__main__":
    main()