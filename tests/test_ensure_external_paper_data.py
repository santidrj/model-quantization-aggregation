from pathlib import Path
from zipfile import ZipFile

import pytest

from src.data.ensure_external_paper_data import (
    ArchiveMember,
    RemoteArchiveSource,
    ensure_external_paper_data,
)


def test_missing_default_artifact_without_source_raises_pointing_at_readme(tmp_path: Path):
    (tmp_path / "README.md").write_text("Obtain paper-data.csv manually.\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"paper-data\.csv") as exc_info:
        ensure_external_paper_data(tmp_path, source=None)

    assert "README.md" in str(exc_info.value)


def test_present_default_artifact_without_source_is_noop(tmp_path: Path):
    (tmp_path / "paper-data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    ensure_external_paper_data(tmp_path, source=None)

    assert (tmp_path / "paper-data.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_source_with_all_required_files_present_does_not_download(tmp_path: Path):
    (tmp_path / "A100.xlsx").write_bytes(b"existing")
    source = RemoteArchiveSource(
        archive_url="https://example.test/archive.zip",
        members=(ArchiveMember(archive_path="pkg/data/A100.xlsx", local_filename="A100.xlsx"),),
    )

    def download_archive(url: str, destination: Path) -> None:
        raise AssertionError(f"download should not be called for {url} -> {destination}")

    ensure_external_paper_data(tmp_path, source=source, download_archive=download_archive)

    assert (tmp_path / "A100.xlsx").read_bytes() == b"existing"


def test_missing_required_members_are_extracted_from_downloaded_archive(tmp_path: Path):
    archive_root = tmp_path / "remote"
    archive_root.mkdir()
    archive_path = archive_root / "archive.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("pkg/data/A100.xlsx", b"workbook-bytes")
        archive.writestr("pkg/data/other.txt", b"ignore-me")

    dest = tmp_path / "external"
    dest.mkdir()
    source = RemoteArchiveSource(
        archive_url="https://example.test/archive.zip",
        members=(ArchiveMember(archive_path="pkg/data/A100.xlsx", local_filename="A100.xlsx"),),
    )

    def download_archive(url: str, destination: Path) -> None:
        assert url == source.archive_url
        destination.write_bytes(archive_path.read_bytes())

    ensure_external_paper_data(dest, source=source, download_archive=download_archive)

    assert (dest / "A100.xlsx").read_bytes() == b"workbook-bytes"
    assert not (dest / "other.txt").exists()
    assert list(dest.glob("*.zip")) == []


def test_partial_presence_extracts_only_missing_and_leaves_existing(tmp_path: Path):
    archive_root = tmp_path / "remote"
    archive_root.mkdir()
    archive_path = archive_root / "archive.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("pkg/keep.xlsx", b"new-keep")
        archive.writestr("pkg/missing.csv", b"new-missing")

    dest = tmp_path / "external"
    dest.mkdir()
    (dest / "keep.xlsx").write_bytes(b"old-keep")

    source = RemoteArchiveSource(
        archive_url="https://example.test/archive.zip",
        members=(
            ArchiveMember(archive_path="pkg/keep.xlsx", local_filename="keep.xlsx"),
            ArchiveMember(archive_path="pkg/missing.csv", local_filename="missing.csv"),
        ),
    )

    def download_archive(url: str, destination: Path) -> None:
        destination.write_bytes(archive_path.read_bytes())

    ensure_external_paper_data(dest, source=source, download_archive=download_archive)

    assert (dest / "keep.xlsx").read_bytes() == b"old-keep"
    assert (dest / "missing.csv").read_bytes() == b"new-missing"


def test_download_failure_raises_runtime_error_naming_expected_files(tmp_path: Path):
    dest = tmp_path / "external"
    dest.mkdir()
    source = RemoteArchiveSource(
        archive_url="https://example.test/archive.zip",
        members=(ArchiveMember(archive_path="pkg/data.csv", local_filename="data.csv"),),
    )

    def download_archive(url: str, destination: Path) -> None:
        raise ConnectionError("network down")

    with pytest.raises(RuntimeError, match=r"data\.csv") as exc_info:
        ensure_external_paper_data(dest, source=source, download_archive=download_archive)

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert not (dest / "data.csv").exists()
