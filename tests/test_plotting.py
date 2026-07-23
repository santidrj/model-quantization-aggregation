import json

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

from src.forestplot.utils import draw_forestplot
from tests.helpers import FIXTURES_DIR


def test_forest_plot_matches_snapshot():
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "id": "S1",
                "evidence_id": 2,
                "effect": "Accuracy",
                "yticklabel": "Quantized A",
                "mean": 8.0,
                "lower_ci": 5.0,
                "upper_ci": 11.0,
            },
            {
                "index": 1,
                "id": "S1",
                "evidence_id": 1,
                "effect": "Accuracy",
                "yticklabel": "Aggregated Accuracy",
                "mean": 7.0,
                "lower_ci": 6.0,
                "upper_ci": 8.0,
            },
            {
                "index": 2,
                "id": "S1",
                "evidence_id": 0,
                "effect": "Accuracy",
                "yticklabel": "Accuracy Belief",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
            {
                "index": 3,
                "id": "S2",
                "evidence_id": 2,
                "effect": "Inference Latency",
                "yticklabel": "Quantized B",
                "mean": -12.0,
                "lower_ci": -15.0,
                "upper_ci": -9.0,
            },
            {
                "index": 4,
                "id": "S2",
                "evidence_id": 1,
                "effect": "Inference Latency",
                "yticklabel": "Aggregated Inference Latency",
                "mean": -10.0,
                "lower_ci": -11.0,
                "upper_ci": -9.0,
            },
            {
                "index": 5,
                "id": "S2",
                "evidence_id": 0,
                "effect": "Inference Latency",
                "yticklabel": "Inference Latency Belief",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    draw_forestplot(frame, axis, main_effects=["Accuracy", "Inference Latency"])

    snapshot = json.loads((FIXTURES_DIR / "forestplot_snapshot.json").read_text())
    actual = {
        "xlim": [round(value, 3) for value in axis.get_xlim()],
        "xticks": [round(value, 3) for value in axis.get_xticks()],
        "yticklabels": [tick.get_text() for tick in axis.get_yticklabels()],
        "ylim": [round(value, 3) for value in axis.get_ylim()],
    }

    assert actual == snapshot
    plt.close(figure)
