#!/usr/bin/env python3
"""Demo 2: inspect values, walk sequences together, and build lists in one line."""


def main():
    """Run the Python collections practice before the NumPy examples."""
    print("Python Tools for Collections")
    print("=" * 50)

    print("\n=== Introspection ===")
    value = "88"  # a heart rate exported from a spreadsheet as text
    print("Original value:", value, "Type:", type(value))
    if isinstance(value, str):
        value = int(value)
    print("Converted value:", value, "Type:", type(value))
    print("Strings have split:", "split" in dir("88"))

    patient_ids = ["P001", "P002", "P003"]
    heart_rates = [88, 104, 112]

    print("\n=== Sequence functions ===")
    print("Numbered patients:")
    for number, patient_id in enumerate(patient_ids, start=1):
        print(f"  Patient {number}: {patient_id}")
    print("Paired records:")
    for patient_id, rate in zip(patient_ids, heart_rates):
        print(f"  {patient_id}: {rate} bpm")
    print(f"Reverse order: {list(reversed(patient_ids))}")
    print(f"Sorted heart rates: {sorted(heart_rates)}")

    print("\n=== List comprehensions ===")
    temps_f = [98.6, 101.2, 99.5, 103.1]
    fevers = [t for t in temps_f if t >= 100.4]
    print(f"Fevers (100.4 or above): {fevers}")

    doses_mg = [250, 500, 125]
    doses_g = [mg / 1000 for mg in doses_mg]
    print(f"Doses in grams: {doses_g}")

    # The wrist monitor reads 2 bpm low, so add its calibration offset.
    calibrated = [rate + 2 for rate in heart_rates]
    print(f"Calibrated heart rates: {calibrated}")

    tachycardic = [patient_id for patient_id, rate in zip(patient_ids, heart_rates) if rate >= 100]
    print(f"Tachycardic (100 bpm or above): {tachycardic}")


if __name__ == "__main__":
    main()
