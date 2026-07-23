import polars as pl
import pytest

from src.data.selection.llm import (
    QUERY_CONTEXT,
    assign_inclusion,
    build_batched_query,
    build_query,
    combine_llm_scores,
    get_excluded_papers,
    get_included_papers,
    get_manual_review_papers,
    read_llm_output,
    simplify_inclusion_results,
)
from tests.helpers import FIXTURES_DIR

EXPECTED_LLM_OUTPUT_ROWS = 2


@pytest.fixture
def paper_scores():
    return pl.DataFrame(
        {
            "Title": ["Excluded Paper", "Included Paper", "Manual Review Paper"],
            "IC1": [3, 5, 4],
            "IC2": [4, 5, 4],
            "IC3": [5, 5, 4],
            "IC4": [5, 6, 4],
            "IC5": [5, 7, 4],
            "IC6": [5, 6, 4],
        }
    )

@pytest.fixture
def query_input():
    return pl.DataFrame(
        {
            "Title": ["Paper A", "Paper B"],
            "Abstract": ["Abstract A", "Abstract B"],
            "Author Keywords": ["keyword-a", "keyword-b"],
        }
    )

def test_inclusion_classification_branches(paper_scores):
    excluded = get_excluded_papers(paper_scores)
    included = get_included_papers(paper_scores)
    manual = get_manual_review_papers(paper_scores, excluded, included)
    assigned = assign_inclusion(paper_scores, conservative=True)

    assert excluded.get_column("Title").to_list() == ["Excluded Paper"]
    assert included.get_column("Title").to_list() == ["Included Paper"]
    assert manual.get_column("Title").to_list() == ["Manual Review Paper"]
    assert assigned.sort("Title").select(["Title", "Included"]).rows() == [
        ("Excluded Paper", "n"),
        ("Included Paper", "y"),
        ("Manual Review Paper", "y"),
    ]

def test_build_query_and_batching(query_input):
    query = build_query(query_input)
    batches = list(build_batched_query(query_input, batch_size=1))

    assert query.startswith(QUERY_CONTEXT)
    assert "Title: Paper A" in query
    assert "Keywords: keyword-b" in query
    assert len(batches) == EXPECTED_LLM_OUTPUT_ROWS
    assert all(batch.startswith(QUERY_CONTEXT) for batch in batches)

def test_build_query_requires_expected_columns(query_input):
    with pytest.raises(ValueError):
        build_query(query_input.drop("Author Keywords"))

def test_combine_and_simplify_scores():
    left = pl.DataFrame({"Title": ["Paper A"], "IC1": [7], "IC2": [6], "IC3": [5], "IC4": [4], "IC5": [3], "IC6": [2]})
    right = pl.DataFrame({"Title": ["Paper A"], "IC1": [5], "IC2": [4], "IC3": [3], "IC4": [2], "IC5": [1], "IC6": [7]})

    combined = combine_llm_scores([left, right])
    assert combined.row(0) == ("Paper A", 6, 5, 4, 3, 2, 4)

    simplified = simplify_inclusion_results(
        pl.DataFrame(
            {
                "Included": ["m", "n", "y"],
                "Manually Included": ["y", "n", "y"],
            }
        )
    )
    assert simplified.get_column("Included").to_list() == [True, False, True]
    assert simplified.get_column("Manually Included").to_list() == [True, False, True]

def test_read_llm_output_fixture():
    fixture_path = FIXTURES_DIR / "llm_output.json"
    frame = read_llm_output(fixture_path)

    assert frame.columns == ["Title", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6"]
    assert frame.height == EXPECTED_LLM_OUTPUT_ROWS
    assert frame.filter(pl.col("Title") == "Paper One").row(0) == ("Paper One", 7, 6, 5, 4, 3, 2)
