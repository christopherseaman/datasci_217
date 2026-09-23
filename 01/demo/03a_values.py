patient_id = "P001"
age_years = 67
temperature_c = 37.8
has_consent = True
print("patient:", patient_id, "type:", type(patient_id))
print("age:", age_years, "type:", type(age_years))
print("temperature:", temperature_c, "type:", type(temperature_c))
print("consent:", has_consent, "type:", type(has_consent))
print("age next year:", age_years + 1)
print("fever:", temperature_c >= 38.0)
print("fall-risk screen:", age_years >= 65 and has_consent)
systolic_readings = [118, 142, 131]
print("readings:", systolic_readings, "count:", len(systolic_readings))
