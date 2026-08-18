import importlib
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_tracked_python_cache_artifacts_in_git_index():
    tracked_files = subprocess.check_output(["git", "ls-files", "src"], cwd=REPO_ROOT, text=True).splitlines()
    tracked_cache_artifacts = [
        tracked_file for tracked_file in tracked_files if "__pycache__" in tracked_file or tracked_file.endswith(".pyc")
    ]

    assert tracked_cache_artifacts == []


def test_import_smoke_for_core_modules():
    module_names = [
        "src.config",
        "src.data.papers.entities",
        "src.data.papers.knowledge_extraction",
        "src.data.selection.llm",
        "src.forestplot.utils",
        "src.run_evidence_extraction",
        "src.dempster_shafer",
        "src.belief_assignment",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name) is not None
