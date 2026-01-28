from collections.abc import Sequence
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import polars as pl

from src.effect_intensity import CorrectnessIntensity, CorrectnessMetrics, EffectIntensity


def draw_ci(
    data: pl.DataFrame, estimate: str, y_tick_label: str, lower_ci_col: str, higher_ci_col: str, ax: Axes, **kwargs: Any
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

    estimate_values = data.select(estimate).to_numpy().flatten()
    lower_ci = data.select(lower_ci_col).to_numpy().flatten()
    upper_ci = data.select(higher_ci_col).to_numpy().flatten()
    y_tick_label_values = data.select(y_tick_label).to_numpy().flatten()

    ax.errorbar(
        estimate_values,
        y=y_tick_label_values,
        xerr=[estimate_values - lower_ci, upper_ci - estimate_values],
        ecolor=ecolor,
        elinewidth=1.4,
        ls="none",
        zorder=0,
    )

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


def format_xticks(
    data: pl.DataFrame,
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


def draw_intensity_areas(ax: Axes, metric: str, y: np.ndarray, x_min: float, x_max: float) -> Axes:
    if metric in CorrectnessMetrics.metrics():
        intensities = CorrectnessIntensity()
    else:
        intensities = EffectIntensity()

    # Fill the areas with different colors
    x_ticks = []
    for key, range in intensities.get_ranges().items():
        if x_min < range[0] and range[1] < x_max:
            ax.fill_betweenx(
                y=y,
                x1=range[0],
                x2=range[1],
                color=intensity_colors[key],
                alpha=0.8,
                zorder=-1,
            )
            x_ticks.append(range[0])
        elif range[0] < x_min < range[1] and range[1] < x_max:
            ax.fill_betweenx(
                y=y,
                x1=x_min,
                x2=range[1],
                color=intensity_colors[key],
                alpha=0.8,
                zorder=-1,
            )
            x_ticks.append(range[1])
        elif x_min < range[0] and range[0] < x_max < range[1]:
            ax.fill_betweenx(
                y=y,
                x1=range[0],
                x2=x_max,
                color=intensity_colors[key],
                alpha=0.8,
                zorder=-1,
            )
            x_ticks.append(range[0])
        else:
            continue

    current_xticks = np.array(ax.get_xticks())

    # Remove the current x-ticks that are inside the x-ticks range
    # and are not in the x_ticks list
    current_xticks = current_xticks[(current_xticks < min(x_ticks)) | (current_xticks > max(x_ticks))]
    # Concatenate the new x-ticks with the current x-ticks
    # and remove duplicates
    x_ticks = np.concat([current_xticks, x_ticks])
    x_ticks = np.unique(x_ticks)
    x_ticks = np.sort(x_ticks)

    # Remove the x-ticks that are outside the x-axis limits
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
    # Add the x-ticks to the axis
    ax.set_xticks(x_ticks)

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
        x_lim = xlim + 5
    else:
        x_min = df["lower_ci"].min()
        x_max = df["upper_ci"].max()
        x_lim = round(max(abs(x_min), x_max) + 5)
    ax.set_xlim(-x_lim, x_lim)
    last_y = -1
    ordered_df = []
    for metric in reversed(main_effects):
        tmp = df[df["mean"].notnull() & df["effect"].str.contains(metric)].copy()
        if tmp.empty:
            continue

        y_min = last_y + 1
        y_max = y_min + len(tmp) - 1
        intensities_y = np.arange(y_min - 0.5, y_max + 1, 1)
        draw_intensity_areas(ax, metric, intensities_y, -x_lim, x_lim)

        metric_header = df[df["mean"].isnull() & df["effect"].str.contains(metric)]
        tmp = tmp.sort_values(by=["id", "evidence_id"], ascending=False).reset_index(drop=True)
        tmp = pd.concat([tmp, metric_header], ignore_index=True).reset_index()

        tmp["index"] += y_min
        last_y = tmp["index"].iloc[-1] + 1

        if metric_header["yticklabel"].str.contains("\n").any():
            tmp.iat[-1, 0] += 1
            last_y += 1

        ordered_df.append(tmp)

    ordered_df = pd.concat(ordered_df).reset_index(drop=True)

    if contains_header:
        header_offset = 3 if ordered_df["yticklabel"].str.contains("\n").any() else 1.5
        main_header = df[df["yticklabel"].str.contains("Belief")]
        main_header["index"] = ordered_df["index"].max() + header_offset
        ordered_df = pd.concat([ordered_df, main_header])

    effects_data = ordered_df.loc[~ordered_df["yticklabel"].str.contains("Aggregated")]
    ax.scatter(effects_data["mean"], effects_data["index"], marker="s", color="black")
    draw_ci(pl.from_pandas(effects_data), "mean", "index", "lower_ci", "upper_ci", ax)
    summary_data = ordered_df.loc[ordered_df["yticklabel"].str.contains("Aggregated")]
    ax.scatter(summary_data["mean"], summary_data["index"], marker="D", color="blue", s=50, label="Aggregated")
    draw_ci(pl.from_pandas(summary_data), "mean", "index", "lower_ci", "upper_ci", ax, ecolor="blue")

    tmp = ordered_df[ordered_df["mean"].notnull()].iloc[-1]
    top_effect = tmp["effect"]
    y_max = tmp["index"].max() + 0.5
    draw_intensity_labels(ax, top_effect, y_max, -x_lim, x_lim, rotation=90)

    draw_ref_xline(ax, y_max, ["precision", "nobs"], [])

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
    # Align negative x-axis labels to the right
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")
        label.set_rotation(45)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)

    return ax
