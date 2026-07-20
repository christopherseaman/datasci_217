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
