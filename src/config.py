import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_ENV_VAR = "ROOT"
ROOT_DIR = Path(os.getenv(ROOT_ENV_VAR, Path(__file__).resolve().parents[1])).expanduser().resolve()

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
TABLES_DIR = ROOT_DIR / "reports" / "tables"


def external_paper_dir(paper_key: str) -> Path:
    return EXTERNAL_DATA_DIR / paper_key


def processed_paper_dir(paper_key: str) -> Path:
    return PROCESSED_DATA_DIR / paper_key


def external_paper_path(paper_key: str, filename: str) -> Path:
    return external_paper_dir(paper_key) / filename


def processed_paper_path(paper_key: str, filename: str) -> Path:
    return processed_paper_dir(paper_key) / filename
