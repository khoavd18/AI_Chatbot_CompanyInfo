import logging.config
import os
from pathlib import Path

import yaml

from src.core.setting_loader import ensure_env_loaded


def setup_logging():
    ensure_env_loaded()
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

    level_override = os.getenv("LOG_LEVEL")
    if level_override:
        normalized_level = level_override.upper()
        config["root"]["level"] = normalized_level
        for handler in config.get("handlers", {}).values():
            handler["level"] = normalized_level
        for logger_config in config.get("loggers", {}).values():
            logger_config["level"] = normalized_level

    logging.config.dictConfig(config)
