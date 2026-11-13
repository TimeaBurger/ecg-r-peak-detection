"""Run the R-peak detection pipeline."""


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
