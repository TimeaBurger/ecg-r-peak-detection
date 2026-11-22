# ECG R-Peak Detection

A Python project that detects heartbeats in ECG recordings by implementing the Pan–Tompkins algorithm from the original paper. It benchmarks the algorithm against a threshold-based baseline on the expert-annotated MIT-BIH Arrhythmia Database and analyzes how detection errors distort time-domain heart rate variability (HRV) measures.

Developed as part of the course *Programming with Python* at IU International University of Applied Sciences. Full title: *Python-Based R-Peak Detection in ECG Signals: Implementation and Evaluation of the Pan–Tompkins Algorithm on the MIT-BIH Arrhythmia Database*.

Status: submitted version (tag v1.0-submission).

## Installation

The project was tested with Python 3.13.1 on macOS. On Windows, use `python` instead of `python3`, and use `.venv\Scripts\activate` to activate the virtual environment.

1. Clone the repository:
   ```
   git clone https://github.com/TimeaBurger/ecg-r-peak-detection.git
   cd ecg-r-peak-detection
   ```
2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
4. Download the MIT-BIH Arrhythmia Database into data/mitdb:
   ```
   python3 -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
   ```

## Data

The data are taken from the MIT-BIH Arrhythmia Database, version 1.0.0, from PhysioNet. The database contains 48 records of about 30 min, sampled at 360 Hz with two channels. The MLII channel is used.
44 records are analyzed. The four records containing paced beats (102, 104, 107, 217) are excluded, since heart rate variability is not meaningful for paced beats.
The data are licensed under the Open Data Commons Attribution License v1.0. They are not stored in the repository and are downloaded in step 4 of the installation.
Citation: Moody and Mark (2001); data set: Moody and Mark (2005), https://doi.org/10.13026/C2F305

## Usage

```
python3 main.py
```

This command evaluates both detectors on 44 records. It takes a few minutes. After that, it saves the results table to `results/results.db` and prints the results and the overall metrics in the terminal.

```
python3 plot_results.py
```

This command reads the saved results and saves the HRV scatter plots (SDNN, RMSSD) as HTML files in `results/figures/`. Run `main.py` first.

## Project structure

```
ecg-r-peak-detection/
├── main.py                 # detection, evaluation and HRV analysis
├── plot_results.py         # HRV figures from the saved results
├── requirements.txt
├── rpeak_detection/
│   ├── data_loader.py      # loading of records and annotations
│   ├── preprocessing.py    # resampling and Pan-Tompkins filter stages
│   ├── detectors.py        # base class, baseline and Pan-Tompkins
│   ├── evaluation.py       # sensitivity, PPV and F1 score
│   ├── hrv.py              # SDNN and RMSSD
│   ├── database.py         # SQLite storage of the results
│   ├── visualization.py    # Bokeh figures
│   └── exceptions.py       # custom exceptions
└── tests/                  # unit tests
```

## Running the tests

```
python3 -m unittest
```
All 40 tests should pass.
