def mean(values):
    """Return the arithmetic mean, or None for empty input."""
    if not values:
        return None

    total = 0
    for value in values:
        total = total + value

    return total / len(values)


def format_summary(record):
    """Return a one-line summary for a measurement record."""
    average = mean(record["values"])

    if average is None:
        return f'{record["label"]} mean: no measurements'

    return f'{record["label"]} mean: {average:.1f}'


records = [
    {"label": "Morning", "values": [18, 21, 24]},
    {"label": "Evening", "values": [20, 22, 26]},
    {"label": "Overnight", "values": []},
]

for record in records:
    summary_line = format_summary(record)
    print(summary_line)
