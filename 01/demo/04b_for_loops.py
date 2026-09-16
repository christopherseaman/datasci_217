scores = [87, 92, 78, 95, 88]
total = 0
count = 0
for score in scores:
    print("score:", score)
    total = total + score
    count = count + 1

print("total:", total)
print("count:", count)
print("average:", total / count)
for position, score in enumerate(scores, start=1):
    print("assignment", position, "score", score)
