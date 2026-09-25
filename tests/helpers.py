import json
from pathlib import Path

import polars as pl

from tests.conftest import REPO_ROOT

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def assert_frame_equal_with_tolerance(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    sort_by: list[str] | None = None,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> None:
    if sort_by:
        left = left.sort(sort_by)
        right = right.sort(sort_by)

    assert left.columns == right.columns
    assert left.shape == right.shape

    for column_name, dtype in zip(left.columns, left.dtypes, strict=True):
        if dtype.is_float():
            left_values = left.get_column(column_name).fill_nan(None).to_list()
            right_values = right.get_column(column_name).fill_nan(None).to_list()
            for left_value, right_value in zip(left_values, right_values, strict=True):
                if left_value is None or right_value is None:
                    assert left_value == right_value
                else:
                    assert abs(left_value - right_value) <= max(abs_tol, abs(right_value) * rel_tol)
        else:
            left_values = left.get_column(column_name).to_list()
            right_values = right.get_column(column_name).to_list()
            assert left_values == right_values
