import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('messy_data.csv')
print("Initial data shape:", df.shape)
print("\nInitial data info:")
df.info()

# 1. Handling Missing Values
print("\n1. Handling Missing Values")
print("Missing values before:")
print(df.isnull().sum())

# Fill numeric columns with median
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Fill categorical columns with mode
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

print("\nMissing values after:")
print(df.isnull().sum())

# 2. Handling Duplicates
print("\n2. Handling Duplicates")
print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

# 3. Data Type Conversions
print("\n3. Data Type Conversions")
print("Data types before:")
print(df.dtypes)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Convert 'price' to numeric, coercing errors to NaN
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Convert 'customer_id' back to integer
df['customer_id'] = df['customer_id'].astype(int)

print("\nData types after:")
print(df.dtypes)

# 4. Handling Outliers
print("\n4. Handling Outliers")
def plot_boxplot(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

plot_boxplot(df, 'price')

# Remove outliers using IQR method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

print("Shape after removing outliers:", df.shape)
plot_boxplot(df, 'price')

# 5. Handling Inconsistent Categories
print("\n5. Handling Inconsistent Categories")
print("Unique categories before cleaning:")
print(df['category'].value_counts())

# Remove ' (incorrect)' from category names
df['category'] = df['category'].str.replace(' (incorrect)', '')

print("\nUnique categories after cleaning:")
print(df['category'].value_counts())

# 6. Feature Engineering
print("\n6. Feature Engineering")
# Extract year and month from date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Create a categorical column for price range
df['price_category'] = pd.cut(df['price'], bins=[0, 50, 100, 150, np.inf], 
                              labels=['Budget', 'Moderate', 'Premium', 'Luxury'])

print(df[['date', 'year', 'month', 'price', 'price_category']].head())

# 7. Encoding Categorical Variables
print("\n7. Encoding Categorical Variables")
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')
print("\nShape after one-hot encoding:", df_encoded.shape)

# 8. Final Data Quality Check
print("\n8. Final Data Quality Check")
print(df_encoded.describe())

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

print("\nFinal data info:")
df_encoded.info()

# 9. Comparing with Original Clean Data
print("\n9. Comparing with Original Clean Data")
df_clean = pd.read_csv('clean_data.csv')

print("Original clean data shape:", df_clean.shape)
print("Cleaned messy data shape:", df_encoded.shape)

common_columns = set(df_clean.columns) & set(df_encoded.columns)
for column in common_columns:
    if df_clean[column].dtype != df_encoded[column].dtype:
        print(f"Data type mismatch in column '{column}':")
        print(f"  Original: {df_clean[column].dtype}")
        print(f"  Cleaned: {df_encoded[column].dtype}")

# Save cleaned data
df_encoded.to_csv('cleaned_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_data.csv'")
