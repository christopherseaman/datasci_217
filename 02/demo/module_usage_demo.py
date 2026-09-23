#!/usr/bin/env python3
"""Demo 3: read a vitals file, save a report, and check what was saved."""

from pathlib import Path

from vitals_tools import (
    format_readings,
    get_systolic,
    highest_reading,
    mean_reading,
)


def read_encounters(data_path):
    """Return one record per usable row of a patient_id,systolic CSV file."""
    with data_path.open("r", encoding="utf-8") as data_file:
        rows = data_file.readlines()

    encounters = []
    for row in rows[1:]:                      # rows[0] is the header line
        if not row.strip():                   # an export often ends with a blank line
            print("Skipping a blank row.")
            continue
        fields = row.strip().split(",")
        if len(fields) != 2:                  # an extra comma leaves too many pieces to unpack
            print(f"Skipping a row with {len(fields)} fields: {row.strip()}")
            continue
        patient_id, raw_systolic = fields
        try:
            systolic = int(raw_systolic)
        except ValueError as error:
            print(f"Skipping {patient_id}: {error}")
        else:
            encounters.append({"patient_id": patient_id, "systolic": systolic})
    return encounters


def main():
    """Build, save, and verify a small vitals report."""
    data_path = Path("clinic_vitals.csv")
    if not data_path.exists():
        print(f"Cannot find {data_path}: run this script from the 02/demo folder.")
        return

    encounters = read_encounters(data_path)
    assert encounters, f"no usable readings in {data_path}"

    readings = get_systolic(encounters)
    lines = format_readings(encounters)
    lines.append(f"Average systolic: {mean_reading(readings):.1f} mmHg")
    lines.append(f"Highest systolic: {highest_reading(readings)} mmHg")
    report_text = "\n".join(lines) + "\n"

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)           # no error when output/ already exists
    report_path = output_dir / "vitals_report.txt"
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    with open(report_path, "r", encoding="utf-8") as report_file:
        saved_text = report_file.read()

    print(f"Read back from {report_path}:")
    print(saved_text, end="")
    print(f"Saved report matches: {saved_text == report_text}")
    assert saved_text == report_text, "the saved report does not match the text we built"

    one_line = saved_text.strip().replace("\n", " | ")
    print(f"Checkpoint passed: {len(lines)} lines saved to {report_path}")
    print(f"Report on one line: {one_line}")


if __name__ == "__main__":
    main()
