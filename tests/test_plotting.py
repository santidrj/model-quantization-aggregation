import json
import math

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import polars as pl

from src.forestplot.utils import (
    EVIDENCE_MODEL_CI_LINEWIDTH,
    SSM_INTENSITY_RANGE_LINEWIDTH,
    draw_forestplot,
    format_forestplot_annote_value,
    generate_forestplot_data,
    split_forestplot_frames,
)
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
    assert label.get_position()[1] < 0.0

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


def test_format_forestplot_annote_value_accepts_numbers_and_strings():
    assert format_forestplot_annote_value(3) == "3"
    assert format_forestplot_annote_value(3.0) == "3"
    assert format_forestplot_annote_value("12") == "12"
    assert format_forestplot_annote_value("") == ""
    assert format_forestplot_annote_value(None) == ""
    assert format_forestplot_annote_value(float("nan")) == ""
    assert format_forestplot_annote_value("nan") == ""


def test_split_forestplot_frames_keeps_header_on_correctness():
    frame = pd.DataFrame(
        [
            {"effect": "Accuracy", "yticklabel": "S1 e1  3  0.20", "mean": 8.0},
            {"effect": "Accuracy", "yticklabel": "Accuracy", "mean": None},
            {"effect": "Storage Size", "yticklabel": "S2 e1  4  0.50", "mean": 40.0},
            {"effect": None, "yticklabel": "Effect  n_eff  Belief", "mean": None},
        ]
    )

    correctness_df, efficiency_df = split_forestplot_frames(frame)

    assert "Effect  n_eff  Belief" in correctness_df["yticklabel"].to_list()
    assert "Effect  n_eff  Belief" in efficiency_df["yticklabel"].to_list()
    assert "S1 e1  3  0.20" in correctness_df["yticklabel"].to_list()
    assert "S2 e1  4  0.50" in efficiency_df["yticklabel"].to_list()


def test_forest_plot_shows_main_header_at_top():
    frame = pd.DataFrame(
        [
            {
                "index": 0,
                "id": "S1",
                "evidence_id": "e1",
                "effect": "Accuracy",
                "yticklabel": "S1 e1  3  0.20",
                "mean": 8.0,
                "lower_ci": 5.0,
                "upper_ci": 11.0,
            },
            {
                "index": 1,
                "id": "Aggregated",
                "evidence_id": "e0",
                "effect": "Accuracy",
                "yticklabel": "Aggregated  0.50",
                "mean": 7.0,
                "lower_ci": 6.0,
                "upper_ci": 8.0,
            },
            {
                "index": 2,
                "id": "S1",
                "evidence_id": "e0",
                "effect": "Accuracy",
                "yticklabel": "Accuracy",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
            {
                "index": 3,
                "id": None,
                "evidence_id": None,
                "effect": None,
                "yticklabel": "Effect  n_eff  Belief",
                "mean": None,
                "lower_ci": None,
                "upper_ci": None,
            },
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    draw_forestplot(frame, axis, main_effects=["Accuracy"], contains_header=True, xlim=100)

    top_to_bottom = [tick.get_text() for tick in reversed(axis.get_yticklabels())]
    assert top_to_bottom[0] == "Effect  n_eff  Belief"
    assert "Accuracy" in top_to_bottom

    plt.close(figure)


def test_generate_forestplot_data_keeps_header_and_blanks_aggregated_n_eff():
    data = pl.DataFrame(
        {
            "id": ["S1", "Aggregated"],
            "evidence_label": ["S1 e1", "Aggregated"],
            "evidence_id": ["e1", "e0"],
            "effect": ["Accuracy", "Accuracy"],
            "mean": [8.0, 7.0],
            "lower_ci": [5.0, 6.0],
            "upper_ci": [11.0, 8.0],
            "n_eff": [3, None],
            "belief": [0.2, 0.5],
        }
    )
    plot_sorting = {"S1": 1, "Aggregated": 0}

    correctness_df, _efficiency_df = generate_forestplot_data(data, ["Accuracy"], plot_sorting)

    header_labels = correctness_df.loc[correctness_df["mean"].isna(), "yticklabel"].astype(str)
    assert any("Belief" in label for label in header_labels)
    assert any(label.strip().startswith("Effect") for label in header_labels)

    aggregated_labels = correctness_df.loc[
        correctness_df["yticklabel"].astype(str).str.contains("Aggregated"), "yticklabel"
    ].astype(str)
    assert not aggregated_labels.str.contains("nan", case=False).any()


def _horizontal_linewidths_at_y(axis, y: float) -> list[float]:
    widths = []
    for collection in axis.collections:
        segments = getattr(collection, "get_segments", lambda: [])()
        linewidths = collection.get_linewidths()
        for index, segment in enumerate(segments):
            if len(segment) != 2:  # noqa: PLR2004
                continue
            (x0, y0), (x1, y1) = segment
            if y0 == y1 == y and x0 != x1:
                width = linewidths[0] if len(linewidths) == 1 else linewidths[index]
                widths.append(float(width))
    return widths


def _has_vertical_caps_at_y(axis, y: float) -> bool:
    for collection in axis.collections:
        segments = getattr(collection, "get_segments", lambda: [])()
        for segment in segments:
            if len(segment) != 2:  # noqa: PLR2004
                continue
            (x0, y0), (x1, y1) = segment
            if x0 == x1 and min(y0, y1) < y < max(y0, y1):
                return True
    return False


def test_forest_plot_styles_aggregated_ranges_unlike_confidence_intervals():
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
        ]
    )

    figure, axis = plt.subplots(figsize=(8, 4))
    draw_forestplot(frame, axis, main_effects=["Accuracy"], xlim=100)

    labels = [tick.get_text() for tick in axis.get_yticklabels()]
    positions = list(axis.get_yticks())
    study_y = positions[labels.index("Quantized A")]
    aggregated_y = positions[labels.index("Aggregated Accuracy")]

    assert EVIDENCE_MODEL_CI_LINEWIDTH in _horizontal_linewidths_at_y(axis, study_y)
    assert SSM_INTENSITY_RANGE_LINEWIDTH in _horizontal_linewidths_at_y(axis, aggregated_y)
    assert _has_vertical_caps_at_y(axis, study_y)
    assert not _has_vertical_caps_at_y(axis, aggregated_y)

    plt.close(figure)
