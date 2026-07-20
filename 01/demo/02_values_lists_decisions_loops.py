study_name = "Sleep pilot"
participant_count_text = "4"
participant_count = int(participant_count_text)
sleep_hours = [7.5, 6.0, 8.0, 7.0]

first_measurement = sleep_hours[0]
total_hours = 0

print(f"Study: {study_name}")
print(f"Participant count text type: {type(participant_count_text)}")
print(f"Participant count number type: {type(participant_count)}")
print(f"First measurement: {first_measurement:.1f} hours")

for hours in sleep_hours:
    total_hours = total_hours + hours
    print(f"Measurement: {hours:.1f} hours")

mean_hours = total_hours / participant_count

if mean_hours >= 7.0:
    summary = "met the seven-hour threshold"
else:
    summary = "was below the seven-hour threshold"

print(f"Mean: {mean_hours:.1f} hours")
print(f"Summary: The study mean {summary}.")
