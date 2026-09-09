# arXiv OpenSearch totalResults shape breaks papers download

Status: resolved

## Problem
`uv run mq papers download` raised `TypeError: string indices must be integers, not 'str'` because live arXiv Atom feeds return bare-string `opensearch:totalResults` while the code expected `{#text: ...}`.

## Answer
Normalize OpenSearch values in `src/data/download.py` and cover bare-string / dict / empty-entry cases in `tests/test_download.py`.
