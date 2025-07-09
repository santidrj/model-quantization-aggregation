import polars as pl
import tiktoken

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def read_potentially_relevant() -> pl.DataFrame:
    """
    Read the potentially relevant studies from the interim data directory.

    Returns
    -------
    polars.DataFrame
        The potentially relevant studies.
    """
    df = pl.read_csv(
        INTERIM_DATA_DIR / "potentially_relevant.csv",
        schema_overrides={
            "Selected": pl.Boolean,
            "To discuss": pl.Boolean,
            "Category": pl.String,
            "Study Type": pl.String,
            "Title": pl.String,
            "Abstract": pl.String,
            "Authors": pl.String,
            "Year": pl.UInt16,
            "Document Type": pl.String,
            "DOI": pl.String,
            "Source title": pl.String,
            "Open Access": pl.String,
        },
    )

    # Split the Category column into multiple columns
    df = (
        df.with_columns(pl.col("Category").str.split(";").alias("Category"))
        .explode("Category")
        .with_columns(pl.col("Category").str.strip_chars().alias("Category"))
    )

    return df


def read_accepted_studies() -> pl.DataFrame:
    """
    Read the accepted studies from the interim data directory.

    Returns
    -------
    polars.DataFrame
        The accepted studies.
    """
    df = pl.read_excel(
        INTERIM_DATA_DIR / "deployment_studies_w.xlsx",
        schema_overrides={
            "Title": pl.String,
            "DOI": pl.String,
            "Selected": pl.String,
            "Cause of rejection": pl.String,
            "To discuss": pl.String,
            "Main category": pl.String,
            "Subcategory": pl.String,
            "Context": pl.String,
            "Model type": pl.String,
            "Hardware": pl.String,
            "Metrics": pl.String,
            "Energy profiling tool": pl.String,
            "Empirical measurement": pl.String,
            "Notes": pl.String,
            "Decision 1": pl.String,
            "Decision 2": pl.String,
            "Decision 3": pl.String,
            "Decision 4": pl.String,
            "Decision 5": pl.String,
            "Decision 6": pl.String,
            "Decision 7": pl.String,
        },
    )

    return df.filter(pl.col("Selected") == "y")


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


def count_tokens_for_openai_model(model: str, promt: str) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(promt))
