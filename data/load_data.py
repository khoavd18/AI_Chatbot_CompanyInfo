import argparse
import json
import logging
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.setting_loader import PROJECT_ROOT, load_settings


DEFAULT_RAW_FILENAME = "database_export_nguyen_vo_dang_khoa_completed.json"
settings = load_settings()
logger = logging.getLogger("ingestion")


def _resolve_input_path(cli_input: str | None = None) -> Path:
    raw_dir = Path(settings["data"]["raw_dir"])
    env_input = os.getenv("RAW_DATA_FILE")
    selected_input = cli_input or env_input

    if selected_input:
        candidate = Path(selected_input).expanduser()
    else:
        candidate = raw_dir / DEFAULT_RAW_FILENAME

    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    return candidate.resolve(strict=False)


def load_data(raw_input: str | None = None) -> int:
    raw_path = _resolve_input_path(raw_input)
    logger.info("Using raw data file: %s", raw_path)

    if not raw_path.exists():
        logger.error("Raw data file not found: %s", raw_path)
        logger.error(
            "Provide a valid file with --input <path>, set RAW_DATA_FILE in .env/environment, or use the default file."
        )
        return 1

    with raw_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    if not data:
        logger.error("No raw data found in file: %s", raw_path)
        return 1

    tables = data.get("tables", {})

    if not tables:
        logger.warning("No tables found in raw data file: %s", raw_path)
        return 0

    processed_dir = Path(settings["data"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # table_name: ten bang, table_data: du lieu trong bang
    for table_name, table_data in tables.items():
        if not table_data:
            logger.warning("No data for table %s", table_name)
            continue

        # tao duong dan file json moi cho tung bang
        output_path = processed_dir / f"{table_name}.json"

        # mo file de ghi du lieu
        with output_path.open("w", encoding="utf-8") as outfile:
            # ensure_ascii=False de giu nguyen tieng viet, indent=4 de format lai file de doc hon
            json.dump(table_data, outfile, ensure_ascii=False, indent=4)

        logger.info("Data for table %s written to %s", table_name, output_path)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Split raw export JSON into processed table files.")
    parser.add_argument(
        "--input",
        help="Path to raw export JSON. Priority: --input > RAW_DATA_FILE > default raw file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    return load_data(args.input)


if __name__ == "__main__":
    raise SystemExit(main())
