import polars as pl
import requests
import xmltodict

from src.config import INTERIM_DATA_DIR
from src.data.utils import read_scopus_quantization_papers

ARXIV_API = "http://export.arxiv.org/api/query"


def download_arxiv_papers(query: str, max_results: int) -> list[dict[str, str]]:
    """Download papers from arXiv based on the query."""
    # Prepare the query parameters
    params = {
        "search_query": query,
        "max_results": max_results,
    }
    # Make the request
    response = requests.get(ARXIV_API, params=params)

    # Parse the response
    parsed_response = xmltodict.parse(response.text)["feed"]
    print(f"Found {parsed_response['opensearch:totalResults']['#text']} papers on arXiv.")
    return parsed_response["entry"]


def papers_dict_to_polars_df(papers: list[dict[str, str]]) -> pl.DataFrame:
    """Convert a list of dictionaries to a Polars DataFrame."""

    tmp_list = []
    for paper in papers:
        authors = paper["author"]
        if not isinstance(authors, list):
            authors = authors["name"]
        else:
            authors = "; ".join([author["name"] for author in authors])

        tmp_dict = {
            "Title": paper["title"],
            "Abstract": paper["summary"],
            "Author full names": authors,
            "Published": paper["published"],
            "Link": paper["link"][0]["@href"],
        }
        tmp_list.append(tmp_dict)
    df = (
        pl.from_dicts(tmp_list)
        .with_columns(
            (pl.col("Published").str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ").dt.year()).alias("Year"),
            pl.lit("Pre-print").alias("Document Type"),
            pl.lit("arXiv").alias("Source"),
        )
        .drop("Published")
        .sort("Year", descending=False)
    )

    return df


def clean_titles(papers: pl.DataFrame) -> pl.DataFrame:
    """Clean the titles of the papers."""
    return papers.with_columns(
        pl.col("Title")
        .str.normalize("NFKC")
        .str.replace_all(r"\r\n|\n", " ")
        .str.replace_all(r"\s{2,}", " ")
        .str.replace_all(r"[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]", "'")
        .str.replace_all(r"[\u201c-\u201e\u2033\u275d\u275e\u301d\u301e]", '"'),
        pl.col("Abstract")
        .str.normalize("NFKC")
        .str.replace_all(r"\r\n|\n", " ")
        .str.replace_all(r"\s{2,}", " ")
        .str.replace_all(r"[\u02BB\u02BC\u066C\u2018-\u201A\u275B\u275C]", "'")
        .str.replace_all(r"[\u201c-\u201e\u2033\u275d\u275e\u301d\u301e]", '"'),
    )


if __name__ == "__main__":
    search_query = '(ti:("machine learning" OR ML OR "deep learning" OR DL OR "large language model?" OR "LLM?" OR "neural network?" OR "?NN?" OR "f?undational model?" OR agent) AND (quantization OR quantize OR quantized) AND ("energy consumption" OR "energy efficien*" OR "sustain*" OR "carbon footprint" OR "carbon emission") ANDNOT ("FL" OR "federated learning")) OR (abs:("machine learning" OR ML OR "deep learning" OR DL OR "large language model?" OR "LLM?" OR "neural network?" OR "?NN?" OR "f?undational model?" OR agent) AND (quantization OR quantize OR quantized) AND ("energy consumption" OR "energy efficien*" OR "sustain*" OR "carbon footprint" OR "carbon emission") ANDNOT ("FL" OR "federated learning")) AND submittedDate:[202201010000 TO 202502040000]'  # noqa: E501
    max_results = 1000
    arxiv_papers = download_arxiv_papers(search_query, max_results)

    arxiv_papers = papers_dict_to_polars_df(arxiv_papers)

    scopus_papers = read_scopus_quantization_papers().sort("Year", descending=False)

    print(f"Joining {scopus_papers.height} Scopus papers with {arxiv_papers.height} arXiv papers.")
    all_papers = pl.concat([scopus_papers, arxiv_papers], how="diagonal_relaxed")

    # Find pre-prints with Conference or Journal versions
    papers = (
        clean_titles(all_papers)
        .with_columns(pl.col("Title").str.to_lowercase().alias("Temp Title"))
        .unique("Temp Title", keep="first", maintain_order=True)
        .drop("Temp Title")
    )
    print(f"Found {all_papers.height - papers.height} papers with both pre-print and Conference/Journal versions.")

    # Write the data to a CSV file
    save_path = INTERIM_DATA_DIR / "model-quantization-papers.csv"
    print(f"Writing {papers.height} papers to {save_path}.")
    papers.write_csv(save_path)
