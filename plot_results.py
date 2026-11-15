"""Create the figures from the saved detection results."""

import os

from bokeh.plotting import output_file, save

from main import DATABASE_PATH, RESULTS_DIRECTORY
from rpeak_detection.database import load_results
from rpeak_detection.visualization import create_hrv_scatter_plot

FIGURES_DIRECTORY = os.path.join(RESULTS_DIRECTORY, "figures")
HRV_METRICS = ["sdnn", "rmssd"]


def main():
    """Load the saved results and save the HRV figures as HTML files."""
    results = load_results(DATABASE_PATH)
    os.makedirs(FIGURES_DIRECTORY, exist_ok=True)
    for metric in HRV_METRICS:
        plot = create_hrv_scatter_plot(results, metric)
        output_file(os.path.join(FIGURES_DIRECTORY, f"{metric}.html"))
        save(plot)


if __name__ == "__main__":
    main()
