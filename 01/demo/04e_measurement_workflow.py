scores = [92, 76, 88, 64]
passing_score = 70
total = 0
count = 0
passing = 0

for position, score in enumerate(scores, start=1):
    total = total + score
    count = count + 1
    if score >= passing_score:
        status = "PASS"
        passing = passing + 1
    else:
        status = "REVIEW"
    print("Student", position, "score:", score, status)

average = total / count
print("total:", total)
print("count:", count)
print("average:", average)
print("passing:", passing)
