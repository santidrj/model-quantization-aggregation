# Zenodo 504 during Gonzalez archive download

Status: resolved

## Problem
`uv run mq papers ensure-external-data --download-missing` failed with HTTP 504 while fetching the ~853 MB Gonzalez Zenodo zip. Manual `curl --retry` succeeded.

## Answer
Retry transient HTTP/network errors in `_default_download_archive` (408/429/5xx) with backoff; do not retry client errors.
