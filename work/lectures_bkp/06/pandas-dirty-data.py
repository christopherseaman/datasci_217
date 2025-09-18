import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_clean_data(n_rows=1000):
    """Generate a clean dataset"""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    
    # Generate other columns
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Toys']
    data = {
        'date': dates,
        'category': np.random.choice(categories, n_rows),
        'price': np.random.uniform(10, 1000, n_rows).round(2),
        'quantity': np.random.randint(1, 50, n_rows),
        'customer_id': np.random.randint(1000, 9999, n_rows),
        'rating': np.random.randint(1, 6, n_rows)
    }
    
    return pd.DataFrame(data)

def introduce_missing_values(df, percentage=0.1):
    """Introduce missing values to the dataset"""
    mask = np.random.rand(*df.shape) < percentage
    df_missing = df.mask(mask)
    return df_missing

def add_duplicates(df, percentage=0.05):
    """Add duplicate rows to the dataset"""
    n_duplicates = int(len(df) * percentage)
    duplicates = df.sample(n=n_duplicates, replace=True)
    return pd.concat([df, duplicates]).reset_index(drop=True)

def introduce_outliers(df, column, percentage=0.02):
    """Introduce outliers to a specific column"""
    n_outliers = int(len(df) * percentage)
    outlier_indices = np.random.choice(df.index, n_outliers, replace=False)
    multiplier = np.random.choice([10, 100], n_outliers)
    df.loc[outlier_indices, column] *= multiplier
    return df

def alter_datatypes(df):
    """Alter datatypes of some columns"""
    df['date'] = df['date'].astype(str)
    df['price'] = df['price'].astype(str)
    df['customer_id'] = df['customer_id'].astype(float)
    return df

def add_inconsistent_categories(df, column, percentage=0.05):
    """Add inconsistent categories to a categorical column"""
    n_inconsistent = int(len(df) * percentage)
    inconsistent_indices = np.random.choice(df.index, n_inconsistent, replace=False)
    df.loc[inconsistent_indices, column] = df.loc[inconsistent_indices, column] + ' (incorrect)'
    return df

def create_messy_dataset():
    """Create a messy dataset with various data quality issues"""
    df_clean = generate_clean_data()
    
    df_messy = df_clean.copy()
    df_messy = introduce_missing_values(df_messy)
    df_messy = add_duplicates(df_messy)
    df_messy = introduce_outliers(df_messy, 'price')
    df_messy = alter_datatypes(df_messy)
    df_messy = add_inconsistent_categories(df_messy, 'category')
    
    return df_clean, df_messy

# Generate clean and messy datasets
df_clean, df_messy = create_messy_dataset()

print("Clean dataset:")
print(df_clean.head())
print("\nClean dataset info:")
df_clean.info()

print("\n\nMessy dataset:")
print(df_messy.head())
print("\nMessy dataset info:")
df_messy.info()

# Save datasets
df_clean.to_csv('clean_data.csv', index=False)
df_messy.to_csv('messy_data.csv', index=False)
print("\nDatasets saved as 'clean_data.csv' and 'messy_data.csv'")

# Data cleaning steps will go here
# ... (to be filled with cleaning techniques)

