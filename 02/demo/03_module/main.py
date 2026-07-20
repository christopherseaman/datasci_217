from analysis_utils import format_summary


def main():
    """Create, read back, and print the measurement report."""
    records = [
        {"label": "Morning", "values": [18, 21, 24]},
        {"label": "Evening", "values": [20, 22, 26]},
        {"label": "Overnight", "values": []},
    ]

    with open("report.txt", "w", encoding="utf-8") as report_file:
        for record in records:
            summary_line = format_summary(record)
            report_file.write(summary_line + "\n")

    with open("report.txt", "r", encoding="utf-8") as report_file:
        saved_report = report_file.read()

    print(saved_report, end="")


if __name__ == "__main__":
    main()
