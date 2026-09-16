counter = 1
while counter <= 3:
    print("counter:", counter)
    counter = counter + 1

scores = [87, 92, 78, 95, 88]
print("Stop at the first score above 90:")
for score in scores:
    if score > 90:
        print("found:", score)
        break

print("Skip scores below 80:")
for score in scores:
    if score < 80:
        continue
    print("processing:", score)
