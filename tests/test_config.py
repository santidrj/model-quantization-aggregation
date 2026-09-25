from pathlib import Path

from src.config import (
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
    external_paper_dir,
    external_paper_path,
    processed_paper_dir,
    processed_paper_path,
)
from src.data.papers.entities import Papers


def test_root_and_data_directories_exist():
    assert ROOT_DIR.exists()
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()
    assert INTERIM_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert EXTERNAL_DATA_DIR.exists()
    assert FIGURES_DIR.exists()


def test_paper_path_helpers_resolve_expected_locations():
    paper_key = Papers.DUBHIR.value.KEY

    assert external_paper_dir(paper_key) == EXTERNAL_DATA_DIR / paper_key
    assert processed_paper_dir(paper_key) == PROCESSED_DATA_DIR / paper_key
    assert external_paper_path(paper_key, "paper-data.csv") == EXTERNAL_DATA_DIR / paper_key / "paper-data.csv"
    assert (
        processed_paper_path(paper_key, "improvement_metrics.parquet")
        == PROCESSED_DATA_DIR / paper_key / "improvement_metrics.parquet"
    )


def test_root_dir_is_a_normalized_path():
    assert Path(ROOT_DIR).resolve() == ROOT_DIR
