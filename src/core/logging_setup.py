import logging.config
from pathlib import Path

import yaml


def setup_logging():
    src_dir = Path(__file__).resolve().parents[1]          # .../src
    project_root = src_dir.parent                           # project root
    config_path = src_dir / "config" / "logging.yaml"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # đổi filename sang absolute path để không phụ thuộc cwd
    if "handlers" in config and "file" in config["handlers"]:
        config["handlers"]["file"]["filename"] = str(logs_dir / "application.log")

    logging.config.dictConfig(config)