import pandas as pd
import numpy as np
from numpy.random import default_rng
import argparse
from tqdm import tqdm

# Default input and output filenames
DEFAULT_INPUT_FILE = 'ddf--datapoints--population--by--income_groups--age--gender--year.csv'
DEFAULT_OUTPUT_FILE = 'messy_population_data.csv'

# Create a random number generator
rng = default_rng(seed=42)

def load_data(file_path):
    """Load the original clean dataset."""
    return pd.read_csv(file_path)

def introduce_missing_values(df, percentage=0.05):
    """Introduce missing values to the dataset."""
    mask = rng.random(df.shape) < percentage
    return df.mask(mask)

def add_duplicates(df, percentage=0.03):
    """Add duplicate rows to the dataset."""
    n_duplicates = int(len(df) * percentage)
    duplicates = df.sample(n=n_duplicates, replace=True, random_state=rng)
    return pd.concat([df, duplicates]).reset_index(drop=True)

def introduce_outliers(df, column, percentage=0.02):
    """Introduce outliers to the population column."""
    n_outliers = int(len(df) * percentage)
    outlier_indices = rng.choice(df.index, n_outliers, replace=False)
    multiplier = rng.choice([100, 1000], n_outliers)
    df.loc[outlier_indices, column] *= multiplier
    return df

def alter_datatypes(df):
    """Alter datatypes of some columns."""
    df['year'] = df['year'].astype(str)
    df['population'] = df['population'].astype(str)
    return df

def add_inconsistent_categories(df, column, percentage=0.05):
    """Add inconsistent categories to a categorical column."""
    n_inconsistent = int(len(df) * percentage)
    inconsistent_indices = rng.choice(df.index, n_inconsistent, replace=False)
    
    if column == 'income_groups':
        df.loc[inconsistent_indices, column] = df.loc[inconsistent_indices, column] + '_typo'
    elif column == 'gender':
        # For gender, we'll introduce a new category
        df.loc[inconsistent_indices, column] = 3  # Assuming 1 and 2 are the original categories
    
    return df

def add_future_dates(df, num_future=50):
    """Add some future dates to the dataset."""
    # Convert 'year' to numeric, coercing errors to NaN
    years = pd.to_numeric(df['year'], errors='coerce')
    max_year = years.max()
    
    if pd.isna(max_year):
        max_year = 2023  # Use a default if all years are NaN
    
    future_data = df.sample(n=num_future, replace=True, random_state=rng)
    future_data['year'] = rng.integers(int(max_year) + 1, int(max_year) + 20, num_future).astype(str)
    return pd.concat([df, future_data]).reset_index(drop=True)

if __name__ == '__main__':
    # Parse command-line arguments (fall back to defaults)
    parser = argparse.ArgumentParser(description="Create a messy dataset from a clean CSV file.")
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="Path to the input CSV file")
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE, help="Path to save the messy CSV file")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    # Load the clean dataset
    df_clean = load_data(input_file)
    
    # Create a messy dataset with progress bar
    df_messy = df_clean.copy()
    
    # Define the steps for creating messy data
    messy_steps = [
        ("Introducing missing values", lambda df: introduce_missing_values(df)),
        ("Adding duplicates", lambda df: add_duplicates(df)),
        ("Introducing outliers", lambda df: introduce_outliers(df, 'population')),
        ("Altering datatypes", lambda df: alter_datatypes(df)),
        ("Adding inconsistent categories (income_groups)", lambda df: add_inconsistent_categories(df, 'income_groups')),
        ("Adding inconsistent categories (gender)", lambda df: add_inconsistent_categories(df, 'gender')),
        ("Adding future dates", lambda df: add_future_dates(df))
    ]
    
    # Apply messy steps with progress bar
    with tqdm(total=len(messy_steps), desc="Creating messy dataset") as pbar:
        for step_desc, step_func in messy_steps:
            df_messy = step_func(df_messy)
            pbar.set_description(f"Completed: {step_desc}")
            pbar.update(1)
    
    # Save the messy dataset
    df_messy.to_csv(output_file, index=False)
    print(f"\nMessy dataset saved as '{output_file}'")

    # Hint: Uncomment for insights
    # print("\nClean dataset info:")
    # print(df_clean.info())

    # print("\nMessy dataset info:")
    # print(df_messy.info())

    # print("\nSample of messy data:")
    # print(df_messy.sample(10))