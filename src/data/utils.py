import polars as pl

from src.config import RAW_DATA_DIR


def read_scopus_quantization_papers() -> pl.DataFrame:
    """
    Read the Scopus data on model quantization.

    Returns
    -------
    polars.DataFrame
        The Scopus data on model quantization
    """
    return pl.read_csv(RAW_DATA_DIR / "scopus-model-quantization.csv", encoding="utf8").with_columns(
        pl.col("Author full names").str.replace_all(r"\s\(\d+\)", ""),
    )
