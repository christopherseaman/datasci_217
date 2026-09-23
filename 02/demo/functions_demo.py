#!/usr/bin/env python3
"""Demo 2: summarize clinic encounters with containers, functions, and imports."""

import statistics as stats

from vitals_tools import get_systolic, highest_reading, mean_reading


session = ("Bayview Clinic", "2026-09-18")   # a tuple: a fixed record, never edited
clinic, visit_date = session                 # unpacking

encounters = [
    {"patient_id": "P001", "systolic": 128},
    {"patient_id": "P002", "systolic": 142},
    {"patient_id": "P003", "systolic": 118},
]

print(f"=== {clinic} {visit_date}: systolic summary ===")

print("Before: every script would repeat this loop.")
readings = []
for encounter in encounters:
    readings.append(encounter["systolic"])
print(f"Before loop extracted: {readings}")

print("After: reuse helpers from vitals_tools.")
readings = get_systolic(encounters)
print(f"After get_systolic() extracted: {readings}")

encounters.append({"patient_id": "P004", "systolic": 136})   # a list can grow; a tuple cannot
readings = get_systolic(encounters)     # read the log again: the old readings predate P004
ranked = sorted(readings)
print(f"P004 arrived late, so the log now holds {len(encounters)} encounters.")
print(f"Readings in order: {ranked}")
print(f"Two highest readings: {ranked[-2:]}")
print(f"Average systolic: {mean_reading(readings):.1f} mmHg")
print(f"Highest systolic: {highest_reading(readings)} mmHg")
print(f"statistics.mean agrees: {stats.mean(readings) == mean_reading(readings)}")

empty_average = mean_reading([])
if empty_average is None:               # `is None`, because `if not empty_average` also catches 0.0
    print("Average with no readings: nothing to average")
else:
    print(f"Average with no readings: {empty_average:.1f}")
print(f"Average of two zero pain scores: {mean_reading([0, 0])}")

record = encounters[0]                  # each encounter is a dictionary: field name to value
print("One encounter, field by field:")
for field, value in record.items():
    print(f"  {field}: {value}")
print(f"P001's systolic: {record['systolic']} mmHg")
print(f"P001's follow-up: {record.get('follow_up', 'none scheduled')}")

typed_cutoff = input("Flag systolic at or above (press Enter for 130): ")
if not typed_cutoff:                    # an empty answer means Enter alone
    typed_cutoff = "130"
cutoff = int(typed_cutoff)

flagged_ids = []
for encounter in encounters:
    if encounter["systolic"] >= cutoff:
        flagged_ids.append(encounter["patient_id"])

morning_session = {"P001", "P003", "P004"}   # a set: distinct IDs, no order
flagged = set(flagged_ids)                   # the same IDs as a set, so & can compare groups
print(f"Flagged ({cutoff} mmHg and above): {flagged_ids}")
print(f"Flagged patients in the morning session: {sorted(flagged & morning_session)}")
