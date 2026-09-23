check = 1
while check <= 3:
    print("blood pressure check:", check)
    check += 1

systolic_readings = [128, 142, 118, 135, 151]
print("Stop at the first reading of 140 or above:")
for systolic in systolic_readings:
    if systolic >= 140:
        print("found:", systolic)
        break

print("Skip readings below 130:")
for systolic in systolic_readings:
    if systolic < 130:
        continue
    print("review:", systolic)
