from __future__ import annotations

import pytest
import requests

from src.data.download import _opensearch_text_value, download_arxiv_papers


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("208", "208"),
        ({"#text": "208"}, "208"),
        ({"#text": "208", "@xmlns": "http://example.test"}, "208"),
        (None, "0"),
    ],
)
def test_opensearch_text_value_normalizes_bare_and_dict_shapes(value: object, expected: str) -> None:
    assert _opensearch_text_value(value) == expected  # type: ignore[arg-type]


def test_opensearch_text_value_rejects_unexpected_shapes() -> None:
    with pytest.raises(TypeError, match="Unexpected OpenSearch field shape"):
        _opensearch_text_value({"count": "208"})  # type: ignore[arg-type]


def test_download_arxiv_papers_handles_bare_total_results_and_entry_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = """<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://www.w3.org/2005/Atom">
          <opensearch:totalResults>2</opensearch:totalResults>
          <entry><title>Paper A</title></entry>
          <entry><title>Paper B</title></entry>
        </feed>
        """

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())

    entries = download_arxiv_papers("quantization", max_results=10)

    assert [entry["title"] for entry in entries] == ["Paper A", "Paper B"]


def test_download_arxiv_papers_handles_single_entry_and_dict_total_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        text = """<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://www.w3.org/2005/Atom">
          <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:totalResults>
          <entry><title>Only Paper</title></entry>
        </feed>
        """

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())

    entries = download_arxiv_papers("quantization", max_results=10)

    assert [entry["title"] for entry in entries] == ["Only Paper"]


def test_download_arxiv_papers_returns_empty_list_when_no_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = """<?xml version='1.0' encoding='UTF-8'?>
        <feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://www.w3.org/2005/Atom">
          <opensearch:totalResults>0</opensearch:totalResults>
        </feed>
        """

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: FakeResponse())

    assert download_arxiv_papers("quantization", max_results=10) == []
