systolic_readings = [128, 142, 118, 135, 151]
total = 0
count = 0
for systolic in systolic_readings:
    print("systolic:", systolic)
    total = total + systolic
    count = count + 1

print("total:", total)
print("count:", count)
print("average:", total / count)
for visit, systolic in enumerate(systolic_readings, start=1):
    print("visit", visit, "systolic", systolic)
