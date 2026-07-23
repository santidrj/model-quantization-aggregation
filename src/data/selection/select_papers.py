from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from google import genai
import polars as pl
from tqdm import tqdm

from src.config import INTERIM_DATA_DIR
from src.data.selection.llm import (
    GEMINI_MODEL,
    QUERY_CONTEXT,
    REQUIRED_QUERY_COLUMNS,
    create_paper_context_message,
    gemini_query,
)

PAPERS_FILENAME = "model-quantization-papers.csv"
SAMPLE_PAPERS_FILENAME = "model-quantization-papers-200-sample.xlsx"
SCORES_FILENAME = f"{GEMINI_MODEL}-scores.parquet"


def load_selection_papers(
    papers_path: Path = INTERIM_DATA_DIR / PAPERS_FILENAME,
    sample_papers_path: Path = INTERIM_DATA_DIR / SAMPLE_PAPERS_FILENAME,
) -> pl.DataFrame:
    """
    Load papers that still need LLM scoring.
    """
    papers = pl.read_csv(papers_path, encoding="utf8")
    sample_papers = pl.read_excel(sample_papers_path)
    return papers.join(sample_papers, on="Title", how="anti").select(list(REQUIRED_QUERY_COLUMNS))


def build_single_paper_queries(papers: pl.DataFrame) -> Generator[str, None, None]:
    """
    Build one prompt per paper using the shared query context.
    """
    for paper in papers.to_dicts():
        yield f"{QUERY_CONTEXT}\n\n{create_paper_context_message(paper)}"


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
        queries = build_single_paper_queries(papers)
        futures = [executor.submit(gemini_query, client, query) for query in queries]
        for future in as_completed(futures):
            pbar.update(1)
            yield future.result()


def collect_query_results(client: genai.Client, papers: pl.DataFrame) -> dict:
    """
    Merge per-paper query results into one title-to-scores mapping.
    """
    results: dict = {}
    for result in make_queries(client, papers):
        results |= result
    return results


def scores_to_frame(results: dict) -> pl.DataFrame:
    """
    Convert raw LLM responses into the stable parquet output schema.
    """
    return pl.from_dict(results).transpose(
        include_header=True,
        header_name="Title",
        column_names=["IC1", "IC2", "IC3", "IC4", "IC5", "IC6"],
    )


def write_scores(scores_df: pl.DataFrame, output_path: Path = INTERIM_DATA_DIR / SCORES_FILENAME) -> None:
    scores_df.write_parquet(output_path)


def main():
    print("Loading data...")
    relevant_data = load_selection_papers()

    # Load the Gemini model
    print("Loading Gemini model...")

    client = genai.Client()

    # Query the model for one paper at a time
    print("Querying Gemini model...")
    scores_df = scores_to_frame(collect_query_results(client, relevant_data))

    # Save the results
    write_scores(scores_df)


if __name__ == "__main__":
    main()
