import polars as pl

from src.config import PROCESSED_DATA_DIR
from src.data.papers.entities import Paper


def read_paper_metadata(paper: Paper) -> pl.DataFrame:
    """
    Read the metadata for a given paper.

    Parameters
    ----------
    paper : Paper
        The to read the metadata for.

    Returns
    -------
    polars.DataFrame
        The metadata for the paper.
    """
    return pl.read_json(
        PROCESSED_DATA_DIR / paper.KEY / "metadata.json",
        schema=pl.Schema(
            {
                "id": pl.String,
                "key": pl.String,
                "title": pl.String,
                "study_type": pl.String,
                "data_quality": pl.String,
                "energy_measurement": pl.Struct(
                    {
                        "measurement_method": pl.List(pl.String),
                        "software_tools": pl.List(pl.String),
                        "repetitions": pl.UInt8,
                    }
                ),
                "quantization_schema": pl.Struct(
                    {
                        "baseline_precision": pl.String,
                        "target_precision": pl.List(pl.String),
                        "quantization_targets": pl.List(pl.String),
                        "quantization_method": pl.List(pl.String),
                        "frameworks": pl.List(pl.String),
                        "formats": pl.List(pl.String),
                    }
                ),
                "hardware": pl.List(
                    pl.Struct(
                        {
                            "device": pl.Struct(
                                {
                                    "model": pl.String,
                                    "board": pl.String,
                                    "CPU": pl.String,
                                    "GPU": pl.String,
                                    "RAM": pl.String,
                                    "Flash": pl.String,
                                }
                            )
                        }
                    ),
                ),
                "models": pl.List(pl.String),
                "datasets": pl.List(pl.String),
            }
        ),
    ).with_columns(pl.lit(paper.YEAR).alias("year"))
