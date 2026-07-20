morning_values = [18, 21, 24]
morning_total = 0

for value in morning_values:
    morning_total = morning_total + value

morning_mean = morning_total / len(morning_values)
print(f"Morning mean: {morning_mean:.1f}")

evening_values = [20, 22, 26]
evening_total = 0

for value in evening_values:
    evening_total = evening_total + value

evening_mean = evening_total / len(evening_values)
print(f"Evening mean: {evening_mean:.1f}")
