"""Forest-plot parquets must carry the same B' as effects_by_precision.json."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.config import PROCESSED_DATA_DIR

_EFFECT_KEY = {
    "Accuracy": "accuracy",
    "F1 Score": "f1_score",
    "mAP": "map",
    "Storage Size": "storage_size",
    "GPU Utilization": "gpu_utilization",
    "GPU Power Draw": "gpu_power_draw",
    "GPU Energy Consumption": "gpu_energy_consumption",
    "GPU Memory Utilization": "gpu_memory_utilization",
    "RAM Usage": "ram_usage",
    "Inference Power Draw": "inference_power_draw",
    "Inference Energy Consumption": "inference_energy_consumption",
    "Inference Latency": "inference_latency",
}


def _paper_dirs() -> list[Path]:
    return sorted(
        path
        for path in PROCESSED_DATA_DIR.iterdir()
        if path.is_dir()
        and (path / "effects_by_precision.json").exists()
        and (path / "improvement_statistics_by_precision.parquet").exists()
    )


@pytest.mark.parametrize("paper_dir", _paper_dirs(), ids=lambda path: path.name)
def test_precision_parquet_beliefs_match_effects_json(paper_dir: Path) -> None:
    effects = json.loads((paper_dir / "effects_by_precision.json").read_text(encoding="utf-8"))
    stats = pl.read_parquet(paper_dir / "improvement_statistics_by_precision.parquet")

    expected: dict[tuple[str, str, str], float] = {}
    for row in effects:
        method = row["quantization_method"]
        precision = row["precision_configuration"]
        for key, payload in row.items():
            if isinstance(payload, dict) and "belief" in payload:
                expected[(method, precision, key)] = float(payload["belief"])

    mismatches: list[str] = []
    for row in stats.iter_rows(named=True):
        configuration = row["configuration"]
        method = configuration["quantization_method"]
        precision = configuration["precision_configuration"]
        effect_key = _EFFECT_KEY.get(row["effect"])
        if effect_key is None:
            continue
        key = (method, precision, effect_key)
        if key not in expected:
            continue
        parquet_belief = float(row["belief"])
        json_belief = expected[key]
        if abs(parquet_belief - json_belief) > 5e-4:
            mismatches.append(
                f"{method}/{precision}/{row['effect']}: parquet={parquet_belief} json={json_belief}"
            )

    assert not mismatches, "Stale forest-plot beliefs:\n" + "\n".join(mismatches)


def test_alizadeh_accuracy_beliefs_match_worked_example() -> None:
    """Manuscript discount example uses B'=0.644 for S14 w-int4 accuracy."""
    paper_dir = PROCESSED_DATA_DIR / "alizadehLanguageModelsSoftware2025"
    effects = json.loads((paper_dir / "effects_by_precision.json").read_text(encoding="utf-8"))
    stats = pl.read_parquet(paper_dir / "improvement_statistics_by_precision.parquet")

    json_beliefs = [row["accuracy"]["belief"] for row in effects]
    parquet_beliefs = (
        stats.filter(pl.col("effect") == "Accuracy").sort("configuration")["belief"].to_list()
    )

    assert json_beliefs == [0.644, 0.644]
    assert parquet_beliefs == [0.644, 0.644]
