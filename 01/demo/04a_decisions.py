systolic = 135
if systolic >= 140:
    print("systolic category: stage 2 hypertension")
elif systolic >= 130:
    print("systolic category: stage 1 hypertension")
elif systolic >= 120:
    print("systolic category: elevated")
else:
    print("systolic category: normal")

age_years = 67
has_consent = True
if age_years >= 65 and has_consent:
    print("eligible for fall-risk screening")
else:
    print("not eligible for fall-risk screening")
