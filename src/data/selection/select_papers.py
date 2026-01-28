from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import os

from google import genai
import polars as pl
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR
from src.data.selection.llm import (
    GEMINI_CONFIG,
    GEMINI_MODEL,
    QUERY_CONTEXT,
    create_paper_context_message,
    gemini_query,
)


def make_queries(client: genai.Client, papers: pl.DataFrame) -> Generator[list[dict], None, None]:
    """
    Make queries to the Gemini model for each paper in the DataFrame.

    Parameters
    ----------
    client : genai.Client
        The Gemini client to use for the queries.
    papers : pl.DataFrame
        The DataFrame containing the papers to query.

    Yields
    ------
    Generator[list[dict], None, None]
        A generator that yields the results of the queries.
    """
    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor())
        pbar = stack.enter_context(tqdm(total=len(papers)))
        queries = (f"{QUERY_CONTEXT}\n\n{create_paper_context_message(paper)}" for paper in papers.to_dicts())
        futures = [executor.submit(gemini_query, client, query) for query in queries]
        for future in as_completed(futures):
            pbar.update(1)
            yield future.result()


def main():
    print("Loading data...")
    papers = pl.read_csv(INTERIM_DATA_DIR / "model-quantization-papers.csv", encoding="utf8")
    sample_papers = pl.read_excel(INTERIM_DATA_DIR / "model-quantization-papers-200-sample.xlsx")

    # Remove the papers used for the promt refinement and select the relevant data
    papers = papers.join(sample_papers, on="Title", how="anti")
    relevant_data = papers.select(["Title", "Abstract", "Author Keywords"])

    # Load the Gemini model
    print("Loading Gemini model...")

    client = genai.Client()

    # Query the model for one paper at a time
    print("Querying Gemini model...")
    results = {}
    for result in make_queries(client, relevant_data):
        results = results | result

    scores_df = pl.from_dict(results).transpose(
        include_header=True,
        header_name="Title",
        column_names=["IC1", "IC2", "IC3", "IC4", "IC5", "IC6"],
    )

    # Save the results
    scores_df.write_parquet(INTERIM_DATA_DIR / f"{GEMINI_MODEL}-scores.parquet")


if __name__ == "__main__":
    main()
