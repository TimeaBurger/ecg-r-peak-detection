"""Create figures of the detection and HRV results."""

from bokeh.plotting import figure


def create_hrv_scatter_plot(results, metric):
    """Return a scatter plot of detected versus reference HRV values."""
    plot = figure(
        title=f"{metric.upper()}: detected versus reference",
        x_axis_label=f"Reference {metric.upper()} (ms)",
        y_axis_label=f"Detected {metric.upper()} (ms)",
        x_axis_type="log",
        y_axis_type="log",
    )
    pan_tompkins_rows = results[results["detector_name"] == "pan_tompkins"]
    plot.scatter(
        x=pan_tompkins_rows[f"reference_{metric}"],
        y=pan_tompkins_rows[metric],
        legend_label="Pan-Tompkins",
    )
    lowest = min(results[f"reference_{metric}"].min(), results[metric].min())
    highest = max(results[f"reference_{metric}"].max(), results[metric].max())
    plot.line(
        x=[lowest, highest],
        y=[lowest, highest],
        legend_label="Identity line (y = x)",
        line_dash="dashed",
        color="gray",
    )
    baseline_rows = results[results["detector_name"] == "baseline"]
    plot.scatter(
        x=baseline_rows[f"reference_{metric}"],
        y=baseline_rows[metric],
        legend_label="Baseline",
        color="darkorange",
        marker="triangle",
    )
    return plot
