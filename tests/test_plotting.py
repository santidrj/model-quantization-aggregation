import json
import math

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd

from src.forestplot.utils import draw_forestplot
from tests.helpers import FIXTURES_DIR


def _overflow_arrow_count(axis) -> int:
    return sum(1 for text in axis.texts if text.get_text() == "")


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


def test_forest_plot_clips_offscale_values_to_fixed_display_range():
    point_dimensions = 2
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "id": "S1",
                "evidence_id": 2,
                "effect": "Accuracy",
                "yticklabel": "Quantized A",
                "mean": -130.0,
                "lower_ci": -160.0,
                "upper_ci": -110.0,
            },
            {
                "index": 1,
                "id": "S1",
                "evidence_id": 1,
                "effect": "Accuracy",
                "yticklabel": "Aggregated Accuracy",
                "mean": 92.0,
                "lower_ci": 80.0,
                "upper_ci": 120.0,
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
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    draw_forestplot(frame, axis, main_effects=["Accuracy"], xlim=100)

    marker_offsets = [
        tuple(offset)
        for collection in axis.collections
        if hasattr(collection, "get_offsets")
        for offset in collection.get_offsets()
        if len(offset) == point_dimensions
    ]

    expected_overflow_arrows = 3
    assert tuple(round(value, 3) for value in axis.get_xlim()) == (-100.0, 100.0)
    assert (-100.0, 0.0) in [(round(x, 3), round(y, 3)) for x, y in marker_offsets]
    assert any(text.get_text() == "-130" for text in axis.texts)
    assert _overflow_arrow_count(axis) == expected_overflow_arrows

    plt.close(figure)


def test_forest_plot_skips_arrows_for_headers_and_missing_cis():
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "id": "S1",
                "evidence_id": 2,
                "effect": "Accuracy",
                "yticklabel": "Quantized A",
                "mean": 8.0,
                "lower_ci": math.nan,
                "upper_ci": math.nan,
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
                "yticklabel": "Accuracy",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
            {
                "index": 3,
                "id": "S1",
                "evidence_id": -1,
                "effect": "Accuracy",
                "yticklabel": "Effect #samples Belief",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    draw_forestplot(frame, axis, main_effects=["Accuracy"], contains_header=True, xlim=100)

    assert _overflow_arrow_count(axis) == 0

    plt.close(figure)


def test_forest_plot_keeps_map_variants_in_separate_blocks():
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "id": "S1",
                "evidence_id": 2,
                "effect": "mAP",
                "yticklabel": "Quantized mAP",
                "mean": 1.0,
                "lower_ci": 0.0,
                "upper_ci": 2.0,
            },
            {
                "index": 1,
                "id": "S1",
                "evidence_id": 1,
                "effect": "mAP",
                "yticklabel": "Aggregated mAP",
                "mean": 1.0,
                "lower_ci": 0.0,
                "upper_ci": 2.0,
            },
            {
                "index": 2,
                "id": "S1",
                "evidence_id": 0,
                "effect": "mAP",
                "yticklabel": "mAP",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
            {
                "index": 3,
                "id": "S2",
                "evidence_id": 2,
                "effect": "mAP@0.5",
                "yticklabel": "Quantized mAP@0.5",
                "mean": 3.0,
                "lower_ci": 2.0,
                "upper_ci": 4.0,
            },
            {
                "index": 4,
                "id": "S2",
                "evidence_id": 1,
                "effect": "mAP@0.5",
                "yticklabel": "Aggregated mAP@0.5",
                "mean": 3.0,
                "lower_ci": 2.0,
                "upper_ci": 4.0,
            },
            {
                "index": 5,
                "id": "S2",
                "evidence_id": 0,
                "effect": "mAP@0.5",
                "yticklabel": "mAP@0.5",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
            {
                "index": 6,
                "id": "S3",
                "evidence_id": 2,
                "effect": "mAP@0.5:0.95",
                "yticklabel": "Quantized mAP@0.5:0.95",
                "mean": 5.0,
                "lower_ci": 4.0,
                "upper_ci": 6.0,
            },
            {
                "index": 7,
                "id": "S3",
                "evidence_id": 1,
                "effect": "mAP@0.5:0.95",
                "yticklabel": "Aggregated mAP@0.5:0.95",
                "mean": 5.0,
                "lower_ci": 4.0,
                "upper_ci": 6.0,
            },
            {
                "index": 8,
                "id": "S3",
                "evidence_id": 0,
                "effect": "mAP@0.5:0.95",
                "yticklabel": "mAP@0.5:0.95",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    draw_forestplot(
        frame,
        axis,
        main_effects=["mAP", "mAP@0.5", "mAP@0.5:0.95"],
        xlim=100,
    )

    yticklabels = [tick.get_text() for tick in axis.get_yticklabels()]
    expected_aggregated_rows = 3
    assert yticklabels.count("mAP") == 1
    assert yticklabels.count("mAP@0.5") == 1
    assert yticklabels.count("mAP@0.5:0.95") == 1
    assert sum("Aggregated" in label for label in yticklabels) == expected_aggregated_rows

    plt.close(figure)


def test_forest_plot_places_overflow_label_away_from_arrow():
    display_limit = 100
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "id": "S1",
                "evidence_id": 2,
                "effect": "Accuracy",
                "yticklabel": "Quantized A",
                "mean": -130.0,
                "lower_ci": -160.0,
                "upper_ci": -110.0,
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
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    draw_forestplot(frame, axis, main_effects=["Accuracy"], xlim=display_limit)

    overflow_labels = [text for text in axis.texts if text.get_text() == "-130"]
    assert len(overflow_labels) == 1
    label = overflow_labels[0]
    assert label.get_position()[0] > -display_limit
    assert label.get_position()[1] > 0.0

    plt.close(figure)


def test_forest_plot_orders_studies_and_string_evidence_ids_top_to_bottom():
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "id": "S2",
                "evidence_id": "e10",
                "effect": "Accuracy",
                "yticklabel": "S2 e10",
                "mean": 4.0,
                "lower_ci": 3.0,
                "upper_ci": 5.0,
            },
            {
                "index": 1,
                "id": "S1",
                "evidence_id": "e2",
                "effect": "Accuracy",
                "yticklabel": "S1 e2",
                "mean": 3.0,
                "lower_ci": 2.0,
                "upper_ci": 4.0,
            },
            {
                "index": 2,
                "id": "S1",
                "evidence_id": "e10",
                "effect": "Accuracy",
                "yticklabel": "S1 e10",
                "mean": 2.0,
                "lower_ci": 1.0,
                "upper_ci": 3.0,
            },
            {
                "index": 3,
                "id": "S2",
                "evidence_id": "e1",
                "effect": "Accuracy",
                "yticklabel": "S2 e1",
                "mean": 5.0,
                "lower_ci": 4.0,
                "upper_ci": 6.0,
            },
            {
                "index": 4,
                "id": "S1",
                "evidence_id": "e1",
                "effect": "Accuracy",
                "yticklabel": "S1 e1",
                "mean": 1.0,
                "lower_ci": 0.0,
                "upper_ci": 2.0,
            },
            {
                "index": 5,
                "id": "Aggregated",
                "evidence_id": "e0",
                "effect": "Accuracy",
                "yticklabel": "Aggregated Accuracy",
                "mean": 3.5,
                "lower_ci": 3.0,
                "upper_ci": 4.0,
            },
            {
                "index": 6,
                "id": "S2",
                "evidence_id": "e0",
                "effect": "Accuracy",
                "yticklabel": "Accuracy Belief",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    draw_forestplot(frame, axis, main_effects=["Accuracy"], xlim=100)

    top_to_bottom = [tick.get_text() for tick in reversed(axis.get_yticklabels())]
    assert top_to_bottom == ["Accuracy Belief", "S1 e1", "S1 e2", "S1 e10", "S2 e1", "S2 e10", "Aggregated Accuracy"]

    plt.close(figure)
