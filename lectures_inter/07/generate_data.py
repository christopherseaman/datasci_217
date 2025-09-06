import csv
import random
from datetime import datetime, timedelta
import pandas as pd

def generate_bmi(age):
    base_bmi = random.uniform(18.5, 25.0)
    age_factor = (age - 18) / 72  # Normalize age to 0-1 range
    bmi_increase = random.uniform(0, 10) * age_factor
    return round(base_bmi + bmi_increase, 1)

def generate_blood_pressure(age, bmi):
    base_systolic = 90
    age_factor = (age - 18) / 72  # Normalize age to 0-1 range
    bmi_factor = (bmi - 18.5) / 21.5  # Normalize BMI to 0-1 range
    
    systolic_increase = (30 * age_factor) + (20 * bmi_factor)
    systolic = int(base_systolic + systolic_increase + random.uniform(-10, 10))
    
    return max(90, min(180, systolic))  # Ensure BP is between 90 and 180

# Generate data
data = []
for _ in range(1000):  # Generate 1000 records
    year = random.randint(2016, 2020)
    age = random.randint(18, 90)
    bmi = generate_bmi(age)
    blood_pressure = generate_blood_pressure(age, bmi)
    
    if age < 30:
        age_group = "Young Adult"
    elif age < 50:
        age_group = "Adult"
    else:
        age_group = "Senior"
    
    admissions = random.randint(1, 10)
    
    data.append([year, admissions, age, blood_pressure, bmi, age_group])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Year", "Admissions", "Age", "BloodPressure", "BMI", "AgeGroup"])

# Group by Year and sum Admissions
yearly_data = df.groupby("Year")["Admissions"].sum().reset_index()

# Sort by Year
yearly_data = yearly_data.sort_values("Year")


# Save the full dataset as well for other examples
df.to_csv('data.csv', index=False)

print("data.csv has been generated successfully with all individual records.")
