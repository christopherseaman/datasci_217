# Assignment 4: Pandas Data Analysis
import pandas as pd

def load_and_explore():
    """
    Load orders.csv and explore the data.

    Returns:
        pd.DataFrame: The loaded data
    """
    # TODO: Implement this function
    # 1. Load orders.csv using pd.read_csv()
    # 2. Print the shape using df.shape
    # 3. Print first 5 rows using df.head()
    # 4. Return the DataFrame
    pass

def clean_and_filter(df):
    """
    Clean and filter the orders data.

    Args:
        df (pd.DataFrame): Raw orders data

    Returns:
        pd.DataFrame: Cleaned and filtered data
    """
    # TODO: Implement this function
    # 1. Remove rows where status is 'Cancelled'
    # 2. Fill missing quantity values with 1
    # 3. Keep only 'North' and 'South' regions
    # 4. Return the cleaned DataFrame
    pass

def analyze_orders(df_clean):
    """
    Analyze cleaned orders data.

    Args:
        df_clean (pd.DataFrame): Cleaned orders data

    Returns:
        dict: Dictionary with 'revenue_by_region' and 'top_3_products' Series
    """
    # TODO: Implement this function
    # 1. Create total_price column: quantity * price
    # 2. Calculate total revenue by region (groupby region, sum total_price)
    # 3. Find top 3 products by quantity (groupby product, sum quantity, get top 3)
    # 4. Return dict with both Series
    pass

if __name__ == "__main__":
    # Test your functions
    df = load_and_explore()
    df_clean = clean_and_filter(df)
    results = analyze_orders(df_clean)

    print("\n=== Revenue by Region ===")
    print(results['revenue_by_region'])

    print("\n=== Top 3 Products ===")
    print(results['top_3_products'])
