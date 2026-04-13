import argparse
import logging

from src.core.setting_loader import ensure_env_loaded
from src.core.logging_setup import setup_logging
from src.rag.chunking.architectureType import chunk_architecture_types
from src.rag.chunking.companyInfo import chunk_company_info
from src.rag.chunking.heroSlides import chunk_hero_slides
from src.rag.chunking.interiorStyles import chunk_interior_styles
from src.rag.chunking.news import chunk_news
from src.rag.chunking.newsCategories import chunk_news_categories
from src.rag.chunking.projectCategories import chunk_project_categories
from src.rag.chunking.projects import chunk_projects
from src.rag.vectorstore.upsert import upsert_chunks

ensure_env_loaded()
setup_logging()
logger = logging.getLogger("ingestion")


def _collect_chunks(name, chunk_func):
    try:
        chunks = chunk_func() or []
        logger.info(f"{name}: collected {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.exception(f"{name}: failed during chunking - {e}")
        return []


def run_ingestion_pipeline(*, recreate_collection: bool = False):
    all_chunks = []

    chunk_sources = [
        ("architectureTypes", chunk_architecture_types),
        ("companyInfo", chunk_company_info),
        ("interiorStyles", chunk_interior_styles),
        ("newsCategories", chunk_news_categories),
        ("news", chunk_news),
        ("projectCategories", chunk_project_categories),
        ("projects", chunk_projects),
        ("heroSlides", chunk_hero_slides),
    ]

    for source_name, chunk_func in chunk_sources:
        source_chunks = _collect_chunks(source_name, chunk_func)
        all_chunks.extend(source_chunks)

    if not all_chunks:
        logger.warning("No chunks to upsert.")
        return

    logger.info(f"Total chunks collected before upsert: {len(all_chunks)}")
    logger.info(
        "Pipeline upsert mode: %s",
        "full rebuild with collection recreation" if recreate_collection else "safe upsert without collection recreation",
    )

    upsert_chunks(all_chunks, recreate_collection=recreate_collection)
    logger.info(f"Upserted {len(all_chunks)} chunks into the vector store.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or update the Qdrant hybrid index.")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Recreate the Qdrant collection before upsert. Use this for a full rebuild only.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_ingestion_pipeline(recreate_collection=args.recreate_collection)
