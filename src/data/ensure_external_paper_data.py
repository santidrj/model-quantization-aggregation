from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import requests

DEFAULT_EXTERNAL_ARTIFACT = "paper-data.csv"


@dataclass(frozen=True)
class ArchiveMember:
    archive_path: str
    local_filename: str


@dataclass(frozen=True)
class RemoteArchiveSource:
    archive_url: str
    members: tuple[ArchiveMember, ...]


def _default_download_archive(url: str, destination: Path) -> None:
    response = requests.get(url, stream=True, timeout=(60, 3600))
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def ensure_external_paper_data(
    dest_dir: Path,
    source: RemoteArchiveSource | None = None,
    *,
    download_archive: Callable[[str, Path], None] | None = None,
) -> None:
    """Ensure required external paper data files exist under ``dest_dir``."""
    dest_dir = Path(dest_dir)
    if source is None:
        artifact = dest_dir / DEFAULT_EXTERNAL_ARTIFACT
        if not artifact.is_file():
            readme = dest_dir / "README.md"
            raise FileNotFoundError(
                f"Missing {DEFAULT_EXTERNAL_ARTIFACT} in {dest_dir}. "
                f"See {readme} for instructions on obtaining the data."
            )
        return

    missing = [member for member in source.members if not (dest_dir / member.local_filename).is_file()]
    if not missing:
        return

    downloader = download_archive or _default_download_archive
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="external-paper-archive-") as tmp_dir:
        archive_file = Path(tmp_dir) / "archive.zip"
        try:
            downloader(source.archive_url, archive_file)
            with ZipFile(archive_file) as archive:
                for member in missing:
                    partial = dest_dir / f".{member.local_filename}.partial"
                    try:
                        with archive.open(member.archive_path) as src, partial.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        partial.replace(dest_dir / member.local_filename)
                    finally:
                        if partial.exists():
                            partial.unlink()
        except Exception as exc:
            expected = ", ".join(member.local_filename for member in source.members)
            raise RuntimeError(
                f"Failed to fetch external paper data for {dest_dir}. Expected files: {expected}."
            ) from exc
