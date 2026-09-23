heart_rates = [72, 104, 88, 112]
review_above = 100
total = 0
count = 0
review_count = 0

for visit, heart_rate in enumerate(heart_rates, start=1):
    total += heart_rate
    count += 1
    if heart_rate > review_above:
        status = "REVIEW"
        review_count += 1
    else:
        status = "OK"
    print("Visit", visit, "heart rate:", heart_rate, "bpm", status)

average = total / count
print("total:", total)
print("count:", count)
print("average:", average)
print("readings to review:", review_count)
