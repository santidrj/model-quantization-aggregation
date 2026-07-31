import polars as pl

from src.config import processed_paper_path
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
        processed_paper_path(paper.KEY, "metadata.json"),
        schema=pl.Schema(
            {
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
                        "baseline_precision_configuration": pl.String,
                        "precision_configurations": pl.List(pl.String),
                        "quantization_method": pl.List(pl.String),
                        "quantization_techniques": pl.List(pl.String),
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
                                    "SRAM": pl.String,
                                    "SDRAM": pl.String,
                                    "Storage": pl.String,
                                }
                            )
                        }
                    ),
                ),
                "models": pl.List(pl.String),
                "datasets": pl.List(pl.String),
            }
        ),
    ).with_columns(pl.lit(paper.YEAR).alias("year"), pl.lit(paper.ID).alias("id"), pl.lit(paper.KEY).alias("key"))
