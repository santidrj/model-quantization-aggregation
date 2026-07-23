import polars as pl

from src.data.selection.select_papers import (
    build_single_paper_queries,
    collect_query_results,
    load_selection_papers,
    scores_to_frame,
)

EXPECTED_QUERY_COUNT = 2


def test_load_selection_papers_filters_prompt_refinement_sample(tmp_path):
    papers_path = tmp_path / "papers.csv"
    sample_path = tmp_path / "sample.xlsx"

    pl.DataFrame(
        {
            "Title": ["Keep Me", "Drop Me"],
            "Abstract": ["A", "B"],
            "Author Keywords": ["k1", "k2"],
            "Extra": [1, 2],
        }
    ).write_csv(papers_path)
    pl.DataFrame({"Title": ["Drop Me"]}).write_excel(sample_path)

    result = load_selection_papers(papers_path=papers_path, sample_papers_path=sample_path)

    assert result.columns == ["Title", "Abstract", "Author Keywords"]
    assert result.rows() == [("Keep Me", "A", "k1")]


def test_build_single_paper_queries_includes_context_per_paper():
    papers = pl.DataFrame(
        {
            "Title": ["Paper A", "Paper B"],
            "Abstract": ["Abstract A", "Abstract B"],
            "Author Keywords": ["kw-a", "kw-b"],
        }
    )

    queries = list(build_single_paper_queries(papers))

    assert len(queries) == EXPECTED_QUERY_COUNT
    assert "Title: Paper A" in queries[0]
    assert "Keywords: kw-b" in queries[1]


def test_collect_query_results_merges_all_results(monkeypatch):
    papers = pl.DataFrame(
        {
            "Title": ["Paper A", "Paper B"],
            "Abstract": ["Abstract A", "Abstract B"],
            "Author Keywords": ["kw-a", "kw-b"],
        }
    )

    def fake_make_queries(_client, _papers):
        yield {"Paper A": [1, 2, 3, 4, 5, 6]}
        yield {"Paper B": [7, 6, 5, 4, 3, 2]}

    monkeypatch.setattr("src.data.selection.select_papers.make_queries", fake_make_queries)

    assert collect_query_results(object(), papers) == {
        "Paper A": [1, 2, 3, 4, 5, 6],
        "Paper B": [7, 6, 5, 4, 3, 2],
    }


def test_scores_to_frame_uses_stable_output_schema():
    frame = scores_to_frame({"Paper A": [1, 2, 3, 4, 5, 6]})

    assert frame.columns == ["Title", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6"]
    assert frame.rows() == [("Paper A", 1, 2, 3, 4, 5, 6)]
