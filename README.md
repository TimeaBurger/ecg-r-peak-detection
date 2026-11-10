# ECG R-Peak Detection

A Python project that detects heartbeats in ECG recordings by implementing the Pan–Tompkins algorithm from the original paper. It benchmarks the algorithm against a threshold-based baseline on the expert-annotated MIT-BIH Arrhythmia Database and analyzes how detection errors distort clinical heart rate variability metrics.

Developed as part of the course *Programming with Python* at IU International University of Applied Sciences. Full title: *Python-Based R-Peak Detection in ECG Signals: Implementation and Evaluation of the Pan–Tompkins Algorithm on the MIT-BIH Arrhythmia Database*.

Status: in development

## Installation

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

## Running the tests

```
python3 -m unittest
```