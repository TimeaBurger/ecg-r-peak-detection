"""Run the R-peak detection pipeline."""

import pandas as pd

from rpeak_detection.data_loader import ECGRecordLoader
from rpeak_detection.detectors import (
    PanTompkinsDetector,
    ThresholdBasedDetector,
)
from rpeak_detection.evaluation import (
    evaluate_detection,
    select_beat_positions,
    EvaluationResult
)
from rpeak_detection.hrv import (
    compute_rmssd,
    compute_rr_intervals,
    compute_sdnn,
)

# All records of the MIT-BIH Arrhythmia Database (Moody and Mark, 2005)
ALL_RECORD_NAMES = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

# Records with paced beats, excluded from the evaluation
PACED_RECORD_NAMES = ["102", "104", "107", "217"]


def select_record_names(all_record_names, excluded_record_names):
    """Return the record names that are not in the excluded record names."""
    selected_record_names = []
    for record_name in all_record_names:
        if record_name not in excluded_record_names:
            selected_record_names.append(record_name)
    return selected_record_names


def compute_hrv_metrics(beat_positions, sampling_rate):
    """Return SDNN and RMSSD of the beat positions in milliseconds."""
    rr_intervals = compute_rr_intervals(beat_positions, sampling_rate)
    return {"sdnn": compute_sdnn(rr_intervals),
            "rmssd": compute_rmssd(rr_intervals)}


def evaluate_record(record, detector_name, detector):
    """Return the detection and HRV results of a detector for one record."""
    detected_positions = detector.detect_r_peaks(record.signal,
                                                 record.sampling_rate)
    evaluation_result = evaluate_detection(detected_positions,
                                           record.annotated_positions,
                                           record.annotated_symbols,
                                           record.sampling_rate)
    beat_positions = select_beat_positions(record.annotated_positions,
                                           record.annotated_symbols)
    detected_hrv = compute_hrv_metrics(detected_positions,
                                       record.sampling_rate)
    reference_hrv = compute_hrv_metrics(beat_positions, record.sampling_rate)
    return {
        "record_name": record.record_name,
        "detector_name": detector_name,
        "true_positives": evaluation_result.true_positives,
        "false_positives": evaluation_result.false_positives,
        "false_negatives": evaluation_result.false_negatives,
        "sdnn": detected_hrv["sdnn"],
        "rmssd": detected_hrv["rmssd"],
        "reference_sdnn": reference_hrv["sdnn"],
        "reference_rmssd": reference_hrv["rmssd"],
    }


def compute_overall_metrics(results):
    """Return sensitivity, PPV and F1 score per detector over all records."""
    totals = results.groupby("detector_name")[
        ["true_positives", "false_positives", "false_negatives"]
    ].sum()
    metric_rows = []
    for detector_name, counts in totals.iterrows():
        overall_result = EvaluationResult(counts["true_positives"],
                                          counts["false_positives"],
                                          counts["false_negatives"])
        metric_rows.append({
            "detector_name": detector_name,
            "sensitivity": overall_result.compute_sensitivity(),
            "positive_predictive_value": (
                overall_result.compute_positive_predictive_value()
            ),
            "f1_score": overall_result.compute_f1_score(),
        })
    return pd.DataFrame(metric_rows)


def main():
    """Evaluate both detectors on the selected records and print results."""
    loader = ECGRecordLoader()
    detectors = {
        "baseline": ThresholdBasedDetector(),
        "pan_tompkins": PanTompkinsDetector(),
    }
    record_names = select_record_names(ALL_RECORD_NAMES, PACED_RECORD_NAMES)
    rows = []
    for record_name in record_names[:2]:  # temporary: first two records
        record = loader.load_record(record_name)
        for detector_name, detector in detectors.items():
            row = evaluate_record(record, detector_name, detector)
            rows.append(row)
    results = pd.DataFrame(rows)
    print(results.to_string())
    overall_metrics = compute_overall_metrics(results)
    print(overall_metrics.to_string())


if __name__ == "__main__":
    main()
