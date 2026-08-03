from collections.abc import Sequence
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import polars as pl

from src.data.papers.study_id import study_id_numeric_rank
from src.effect_intensity import CorrectnessIntensity, CorrectnessMetrics, EffectIntensity


def draw_ci(  # noqa: PLR0913
    data: pl.DataFrame,  # noqa: PLR0913
    estimate: str,
    y_tick_label: str,
    lower_ci_col: str,
    higher_ci_col: str,
    ax: Axes,
    **kwargs: Any,
) -> Axes:
    """
    Draws a confidence interval on the given axis.

    Parameters
    ----------
    data : pl.DataFrame
        The data frame containing the estimates and confidence intervals.
    estimate : str
        The column name for the estimate.
    y_tick_label : str
        The label for the y-tick.
    lower_ci_col : str
        The column name for the lower confidence interval.
    higher_ci_col : str
        The column name for the higher confidence interval.
    ax : Axes
        The axis to draw on.

    Returns
    -------
    Axes
        The axis with the confidence interval drawn.
    """
    ecolor = kwargs.get("ecolor", "black")

    lower_ci = data.select(lower_ci_col).to_numpy().flatten()
    upper_ci = data.select(higher_ci_col).to_numpy().flatten()
    y_tick_label_values = data.select(y_tick_label).to_numpy().flatten()

    for lower, upper, y_value in zip(lower_ci, upper_ci, y_tick_label_values, strict=False):
        _draw_clipped_ci(ax, lower, upper, y_value, ecolor, linewidth=1.4)

    return ax


def draw_markers(data: pl.DataFrame, estimate: str, y_tick_label: str, ax: Axes, **kwargs: Any) -> Axes:
    """
    Draws markers on the given axis.

    Parameters
    ----------
    data : pl.DataFrame
        The data frame containing the estimates.
    estimate : str
        The column name for the estimate.
    y_tick_label : str
        The label for the y-tick.
    ax : Axes
        The axis to draw on.
    color : str
        The color of the markers.

    Returns
    -------
    Axes
        The axis with the markers drawn.
    """
    marker = kwargs.get("marker", "s")
    markersize = kwargs.get("markersize", 40)
    markercolor = kwargs.get("markercolor", "darkslategray")
    markeralpha = kwargs.get("markeralpha", 0.8)
    ax.scatter(
        y=y_tick_label,
        x=estimate,
        data=data,
        marker=marker,
        s=markersize,
        color=markercolor,
        alpha=markeralpha,
    )
    return ax


def format_xticks(  # noqa: PLR0913
    data: pl.DataFrame,  # noqa: PLR0913
    estimate: str,
    lower_ci_col: str,
    upper_ci_col: str,
    ax: Axes,
    xlim: tuple | list | None = None,
    **kwargs: Any,
) -> Axes:
    """
    Formats the x-ticks on the given axis.

    Parameters
    ----------
    data : pl.DataFrame
        The data frame containing the estimates and confidence intervals.
    estimate : str
        The column name for the estimate.
    lower_ci_col : str
        The column name for the lower confidence interval.
    upper_ci_col : str
        The column name for the higher confidence interval.
    ax : Axes
        The axis to format.

    Returns
    -------
    Axes
        The axis with formatted x-ticks.
    """
    nticks = kwargs.get("nticks", 5)
    xtick_size = kwargs.get("xtick_size", 10)
    xticklabels = kwargs.get("xticklabels")

    x_min = data.select(lower_ci_col).min().item(0, 0)
    x_max = data.select(upper_ci_col).max().item(0, 0)

    ax.set_xlim(x_min - 0.8, x_max + 0.8)

    ax.xaxis.set_major_locator(plt.MaxNLocator(nticks))
    ax.tick_params(axis="x", labelsize=xtick_size)

    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])

    return ax


def draw_ref_xline(
    ax: Axes,
    y_max: float,
    annoteheaders: Sequence[str] | None | None,
    right_annoteheaders: Sequence[str] | None | None,
    **kwargs: Any,
) -> Axes:
    """
    Draw the vertical reference xline at zero. Unless defaults are overridden in kwargs.

    Parameters
    ----------
    ax : Axes
        The axis to draw the line on.
    y_max : float
        The maximum y-value for the line.
    annoteheaders : list of str
        The headers for the annotations.
    right_annoteheaders : list of str
        The headers for the right annotations.
    kwargs : dict
        Additional arguments to customize the line.

    Returns
    -------
            Matplotlib Axes object.
    """
    xline = kwargs.get("xline", 0)
    if xline is not None:
        xlinestyle = kwargs.get("xlinestyle", "-")
        xlinecolor = kwargs.get("xlinecolor", ".2")
        xlinewidth = kwargs.get("xlinewidth", 1)
        ax.vlines(
            x=xline,
            ymin=-0.8,
            ymax=y_max,
            linestyle=xlinestyle,
            color=xlinecolor,
            linewidth=xlinewidth,
        )
    return ax


def right_flush_yticklabels(data: pl.DataFrame, yticklabel: str, flush: bool, ax: Axes, **kwargs: Any) -> float:
    """Flushes the formatted ytickers to the left. Also returns the amount of max padding in the window width.

    Parameters
    ----------
    data : pl.DataFrame
        The data frame containing the y-tick labels.
    yticklabel : str
        The column name for the y-tick labels.
    flush : bool
        If True, flush the y-tick labels to the left.
    ax : Axes
        The axis to format.
    kwargs : dict
        Additional arguments to customize the labels.

    Returns
    -------
            Window wdith of figure (float)
    """
    fontfamily = kwargs.get("fontfamily", "monospace")
    fontsize = kwargs.get("fontsize", 12)

    fig = plt.gcf()

    y_tick_label = data.select(yticklabel).to_numpy().flatten()
    if flush:
        ax.set_yticklabels(y_tick_label, fontfamily=fontfamily, fontsize=fontsize, ha="left")
    else:
        ax.set_yticklabels(y_tick_label, fontfamily=fontfamily, fontsize=fontsize, ha="right")

    yax = ax.get_yaxis()

    try:
        pad = max(T.label.get_window_extent(renderer=fig.canvas.get_renderer()).width for T in yax.majorTicks)
    except AttributeError:
        pad = max(T.label1.get_window_extent(renderer=fig.canvas.get_renderer()).width for T in yax.majorTicks)
    if flush:
        yax.set_tick_params(pad=pad)

    return pad


def draw_ylabel1(ylabel: str, pad: float, ax: Axes, **kwargs: Any) -> Axes:
    """
    Draw ylabel title for the left-hand side y-axis.

    Parameters
    ----------
    ylabel (str)
            Title of the left-hand side y-axis.
    pad (float)
            Window wdith of figure
    ax (Matplotlib Axes)
            Axes to operate on.

    Returns
    -------
            Matplotlib Axes object.
    """
    fontsize = kwargs.get("fontsize", 12)
    ax.set_ylabel("")
    if ylabel is not None:
        # Retrieve settings from kwargs
        ylabel1_size = kwargs.get("ylabel1_size", 1 + fontsize)
        ylabel1_fontweight = kwargs.get("ylabel1_fontweight", "bold")
        ylabel_loc = kwargs.get("ylabel_loc", "top")
        ylabel_angle = kwargs.get("ylabel_angle", "horizontal")
        ax.set_ylabel(
            ylabel,
            loc=ylabel_loc,
            labelpad=-pad,
            rotation=ylabel_angle,
            size=ylabel1_size,
            fontweight=ylabel1_fontweight,
        )
    return ax


def draw_text(x: float, y: float, text: str, ax: Axes, rotation=0) -> Axes:
    """
    Draws text on the given axis.

    Parameters
    ----------
    x : float
        The x-coordinate of the text.
    y : float
        The y-coordinate of the text.
    text : str
        The text to draw.
    ax : Axes
        The axis to draw on.

    Returns
    -------
    Axes
        The axis with the text drawn.
    """
    ax.text(
        x=x,
        y=y,
        s=text,
        fontfamily="monospace",
        fontsize=11,
        color="black",
        ha="center",
        va="center" if rotation == 0 else "bottom",
        rotation=rotation,
    )
    return ax


def _clip_value(value: float, x_min: float, x_max: float) -> float:
    return min(max(value, x_min), x_max)


def _is_finite(value: float) -> bool:
    return value is not None and np.isfinite(value)


def _draw_overflow_arrow(  # noqa: PLR0913
    ax: Axes, x: float, y: float, direction: str, color: str, linewidth: float
) -> None:
    x_min, x_max = ax.get_xlim()
    arrow_length = max((x_max - x_min) * 0.04, 2)
    start_x = min(x + arrow_length, x_max) if direction == "left" else max(x - arrow_length, x_min)

    ax.annotate(
        "",
        xy=(x, y),
        xytext=(start_x, y),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": linewidth},
        annotation_clip=False,
        zorder=1,
    )


def _draw_ci_cap(ax: Axes, x: float, y: float, color: str, linewidth: float) -> None:
    cap_half_height = 0.14
    ax.vlines(
        x=x,
        ymin=y - cap_half_height,
        ymax=y + cap_half_height,
        color=color,
        linewidth=linewidth,
        zorder=0,
    )


def _draw_clipped_ci(  # noqa: PLR0913
    ax: Axes, lower: float, upper: float, y: float, color: str, linewidth: float
) -> None:
    if not (_is_finite(lower) and _is_finite(upper)):
        return

    x_min, x_max = ax.get_xlim()
    clipped_lower = _clip_value(lower, x_min, x_max)
    clipped_upper = _clip_value(upper, x_min, x_max)

    ax.hlines(y=y, xmin=clipped_lower, xmax=clipped_upper, color=color, linewidth=linewidth, zorder=0)

    if lower < x_min:
        _draw_overflow_arrow(ax, x_min, y, "left", color, linewidth)
    else:
        _draw_ci_cap(ax, clipped_lower, y, color, linewidth)

    if upper > x_max:
        _draw_overflow_arrow(ax, x_max, y, "right", color, linewidth)
    else:
        _draw_ci_cap(ax, clipped_upper, y, color, linewidth)


def _format_overflow_label(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _annotate_overflow_estimate(ax: Axes, value: float, y: float, direction: str, color: str) -> None:
    x_min, x_max = ax.get_xlim()
    arrow_length = max((x_max - x_min) * 0.04, 2)
    label = _format_overflow_label(value)
    label_offset = 0.35

    if direction == "left":
        _draw_overflow_arrow(ax, x_min, y, "left", color, linewidth=1.4)
        ax.text(
            x_min + arrow_length + 2,
            y + label_offset,
            label,
            fontsize=8,
            color=color,
            ha="left",
            va="bottom",
            clip_on=False,
            zorder=3,
        )
        return

    _draw_overflow_arrow(ax, x_max, y, "right", color, linewidth=1.4)
    ax.text(
        x_max - arrow_length - 2,
        y + label_offset,
        label,
        fontsize=8,
        color=color,
        ha="right",
        va="bottom",
        clip_on=False,
        zorder=3,
    )


def _plot_marker_row(ax: Axes, row: pd.Series, marker: str, color: str, size: float) -> None:
    mean_value = row["mean"]
    if not _is_finite(mean_value):
        return

    x_min, x_max = ax.get_xlim()
    y_value = row["index"]
    clipped_mean = _clip_value(mean_value, x_min, x_max)

    ax.scatter([clipped_mean], [y_value], marker=marker, color=color, s=size, zorder=2)

    if mean_value < x_min:
        _annotate_overflow_estimate(ax, mean_value, y_value, "left", color)
    elif mean_value > x_max:
        _annotate_overflow_estimate(ax, mean_value, y_value, "right", color)


intensity_labels = {
    "SN": "{SN}",
    "SN-NE": "{SN,NE}",
    "NE": "{NE}",
    "NE-WN": "{NE,WN}",
    "WN": "{WN}",
    "WN-IF": "{WN,IF}",
    "IF": "{IF}",
    "IF-WP": "{IF,WP}",
    "WP": "{WP}",
    "WP-PO": "{WP,PO}",
    "PO": "{PO}",
    "PO-SP": "{PO,SP}",
    "SP": "{SP}",
}


def draw_intensity_labels(ax: Axes, metric: str, y: float, x_min: float, x_max: float, **kwargs: Any) -> Axes:
    # Write the labels for the areas on top of the shaded areas
    offset = 0.5
    if metric in ["Accuracy", "F1 Score"]:
        intensities = CorrectnessIntensity()
        text_rotation = kwargs.get("rotation", 0)
    else:
        intensities = EffectIntensity()
        text_rotation = kwargs.get("rotation", 90)

    for key, range in intensities.get_ranges().items():
        # draw text if the range is within the x-axis limits
        if x_min < range[0] and range[1] < x_max:
            ax = draw_text(
                x=(range[0] + range[1]) / 2,
                y=y + offset,
                text=intensity_labels[key],
                ax=ax,
                rotation=text_rotation,
            )
        elif range[0] < x_min < range[1] and range[1] < x_max:
            ax = draw_text(
                x=(range[1] + x_min) / 2,
                y=y + offset,
                text=intensity_labels[key],
                ax=ax,
                rotation=text_rotation,
            )
        elif x_min < range[0] and range[0] < x_max < range[1]:
            ax = draw_text(
                x=(range[0] + x_max) / 2,
                y=y + offset,
                text=intensity_labels[key],
                ax=ax,
                rotation=text_rotation,
            )
        else:
            continue
    return ax


intensity_colors = {
    "SN": "#67001F",
    "SN-NE": "#8E063B",
    "NE": "#B2182B",
    "NE-WN": "#D6604D",
    "WN": "#EA6B5F",
    "WN-IF": "#F4A582",
    "IF": "#999999",
    "IF-WP": "#D9F0D3",
    "WP": "#A6DBA0",
    "WP-PO": "#7FBC41",
    "PO": "#4DAC26",
    "PO-SP": "#1B7837",
    "SP": "#00441B",
}


def _intensity_scale_for_metric(metric: str) -> CorrectnessIntensity | EffectIntensity:
    return CorrectnessIntensity() if metric in CorrectnessMetrics.metrics() else EffectIntensity()


def _clip_range(range_start: float, range_end: float, x_min: float, x_max: float) -> tuple[float, float] | None:
    clipped_start = max(range_start, x_min)
    clipped_end = min(range_end, x_max)
    if clipped_start >= clipped_end:
        return None
    return clipped_start, clipped_end


def _extend_xticks(current_xticks: np.ndarray, candidate_ticks: list[float], x_min: float, x_max: float) -> np.ndarray:
    if not candidate_ticks:
        return current_xticks[(current_xticks >= x_min) & (current_xticks <= x_max)]

    bounded_current_ticks = current_xticks[
        (current_xticks < min(candidate_ticks)) | (current_xticks > max(candidate_ticks))
    ]
    merged_ticks = np.concatenate([bounded_current_ticks, np.array(candidate_ticks)])
    merged_ticks = np.unique(merged_ticks)
    merged_ticks = np.sort(merged_ticks)
    return merged_ticks[(merged_ticks >= x_min) & (merged_ticks <= x_max)]


def _rows_for_metric(df: pd.DataFrame, metric: str, *, with_mean: bool) -> pd.DataFrame:
    mean_mask = df["mean"].notnull() if with_mean else df["mean"].isnull()
    return df[mean_mask & (df["effect"] == metric)].copy()


def _build_metric_block(df: pd.DataFrame, metric: str, y_start: int) -> tuple[pd.DataFrame | None, int]:
    metric_rows = _rows_for_metric(df, metric, with_mean=True)
    if metric_rows.empty:
        return None, y_start - 1

    metric_header = _rows_for_metric(df, metric, with_mean=False)
    metric_rows = metric_rows.assign(
        _study_id_sort=metric_rows["id"].astype(str).map(study_id_numeric_rank)
    )
    # Lowest y is the bottom of the group: Aggregated first, then studies (high Study ID → low).
    is_aggregated = metric_rows["yticklabel"].astype(str).str.contains("Aggregated") | (
        metric_rows["id"].astype(str) == "Aggregated"
    )
    aggregated_rows = metric_rows.loc[is_aggregated].drop(columns="_study_id_sort")
    study_rows = (
        metric_rows.loc[~is_aggregated]
        .sort_values(by=["_study_id_sort", "evidence_id"], ascending=False)
        .drop(columns="_study_id_sort")
    )
    metric_rows = pd.concat([aggregated_rows, study_rows], ignore_index=True)
    metric_rows = pd.concat([metric_rows, metric_header], ignore_index=True).reset_index()
    metric_rows["index"] += y_start

    last_y = metric_rows["index"].iloc[-1] + 1
    if metric_header["yticklabel"].str.contains("\n").any():
        metric_rows.iat[-1, 0] += 1
        last_y += 1

    return metric_rows, last_y


def _order_forestplot_rows(df: pd.DataFrame, ax: Axes, main_effects: list[str], x_lim: float) -> pd.DataFrame:
    last_y = -1
    ordered_blocks = []
    for metric in reversed(main_effects):
        next_y = last_y + 1
        metric_block, last_y = _build_metric_block(df, metric, next_y)
        if metric_block is None:
            continue

        y_max = next_y + len(metric_block[metric_block["mean"].notnull()]) - 1
        intensities_y = np.arange(next_y - 0.5, y_max + 1, 1)
        draw_intensity_areas(ax, metric, intensities_y, -x_lim, x_lim)
        ordered_blocks.append(metric_block)

    return pd.concat(ordered_blocks).reset_index(drop=True)


def _append_main_header(df: pd.DataFrame, ordered_df: pd.DataFrame) -> pd.DataFrame:
    header_offset = 3 if ordered_df["yticklabel"].str.contains("\n").any() else 1.5
    main_header = df[df["yticklabel"].str.contains("Belief")].copy()
    main_header["index"] = ordered_df["index"].max() + header_offset
    return pd.concat([ordered_df, main_header])


def _plot_effect_markers(ax: Axes, ordered_df: pd.DataFrame) -> tuple[str, float]:
    plot_rows = ordered_df[ordered_df["mean"].notnull()]
    effects_data = plot_rows.loc[~plot_rows["yticklabel"].str.contains("Aggregated")]
    for _, row in effects_data.iterrows():
        _plot_marker_row(ax, row, marker="s", color="black", size=36)
    draw_ci(pl.from_pandas(effects_data), "mean", "index", "lower_ci", "upper_ci", ax)

    summary_data = plot_rows.loc[plot_rows["yticklabel"].str.contains("Aggregated")]
    for _, row in summary_data.iterrows():
        _plot_marker_row(ax, row, marker="D", color="blue", size=50)
    ax.scatter([], [], marker="D", color="blue", s=50, label="Aggregated")
    draw_ci(pl.from_pandas(summary_data), "mean", "index", "lower_ci", "upper_ci", ax, ecolor="blue")

    last_effect_row = plot_rows.iloc[-1]
    return last_effect_row["effect"], last_effect_row["index"].max() + 0.5


def _format_forestplot_axes(ax: Axes, ordered_df: pd.DataFrame) -> Axes:
    ax.set_yticks(ordered_df["index"], labels=ordered_df["yticklabel"], multialignment="left")
    right_flush_yticklabels(pl.from_pandas(ordered_df), "yticklabel", True, ax)

    header_labels_idx = ordered_df[ordered_df["mean"].isnull()].index
    for idx in header_labels_idx:
        label = ax.get_yticklabels()[idx]
        label.set_fontweight("bold")
        label.set_fontsize(12)

    ax.set_ylim(ordered_df["index"].min() - 0.5, ordered_df["index"].max())
    ax.tick_params(
        top=False, right=False, left=False, bottom=True, labelleft=True, labelbottom=True, labelfontfamily="monospace"
    )
    ax.tick_params(axis="x", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")
        label.set_rotation(45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    return ax


def draw_intensity_areas(ax: Axes, metric: str, y: np.ndarray, x_min: float, x_max: float) -> Axes:
    intensities = _intensity_scale_for_metric(metric)

    # Fill the areas with different colors
    x_ticks = []
    for key, intensity_range in intensities.get_ranges().items():
        clipped_range = _clip_range(intensity_range[0], intensity_range[1], x_min, x_max)
        if clipped_range is None:
            continue
        ax.fill_betweenx(
            y=y,
            x1=clipped_range[0],
            x2=clipped_range[1],
            color=intensity_colors[key],
            alpha=0.8,
            zorder=-1,
        )
        x_ticks.extend(clipped_range)

    ax.set_xticks(_extend_xticks(np.array(ax.get_xticks()), x_ticks, x_min, x_max))

    return ax


def draw_forestplot(
    df: pd.DataFrame,
    ax: Axes,
    main_effects: list[str],
    contains_header: bool = False,
    xlim: float | None = None,
) -> Axes:
    """
    Draws a forest plot on the given axis.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame containing the data to plot.
    ax : Axes
        The axis to draw on.
    main_effects : list[str]
        The list of main effects to plot.
    contains_header : bool
        Whether the data frame contains a header.
    xlim : float
        The x-axis limit.

    Returns
    -------
    Axes
        The axis with the forest plot drawn.
    """
    if xlim is not None:
        x_lim = xlim
    else:
        x_min = df["lower_ci"].min()
        x_max = df["upper_ci"].max()
        x_lim = round(max(abs(x_min), x_max) + 5)
    ax.set_xlim(-x_lim, x_lim)
    ordered_df = _order_forestplot_rows(df, ax, main_effects, x_lim)

    if contains_header:
        ordered_df = _append_main_header(df, ordered_df)

    top_effect, y_max = _plot_effect_markers(ax, ordered_df)
    draw_intensity_labels(ax, top_effect, y_max, -x_lim, x_lim, rotation=90)
    draw_ref_xline(ax, y_max, ["precision", "nobs"], [])
    return _format_forestplot_axes(ax, ordered_df)
