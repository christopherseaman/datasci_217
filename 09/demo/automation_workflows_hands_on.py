#!/usr/bin/env python3
"""
Lecture 09 Live Demo: Automation and Advanced Data Manipulation
DataSci 217 - Hands-on demonstration of automated workflows and advanced pandas

This demo shows practical automation techniques and advanced pandas operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import subprocess
import time
from datetime import datetime, timedelta
import argparse
import sys

def create_automation_datasets():
    """
    Create sample datasets for automation demonstration
    """
    print("Creating datasets for automation demo...")

    # Create directory structure
    data_dir = Path("automation_demo")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    (data_dir / "reports").mkdir(exist_ok=True)

    # Generate multiple daily sales files (simulating regular data drops)
    np.random.seed(42)
    for i in range(7):  # 7 days of data
        date = datetime.now() - timedelta(days=6-i)
        date_str = date.strftime("%Y-%m-%d")

        # Generate daily sales data
        n_transactions = np.random.randint(50, 150)
        daily_sales = pd.DataFrame({
            'timestamp': pd.date_range(f"{date_str} 09:00", f"{date_str} 17:00", periods=n_transactions),
            'customer_id': np.random.randint(1000, 9999, n_transactions),
            'product_code': np.random.choice(['A001', 'A002', 'B001', 'B002', 'C001'], n_transactions),
            'quantity': np.random.randint(1, 10, n_transactions),
            'unit_price': np.random.choice([9.99, 19.99, 29.99, 49.99, 99.99], n_transactions),
            'sales_rep': np.random.choice(['Alice', 'Bob', 'Carol', 'David'], n_transactions),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_transactions)
        })

        daily_sales['total_amount'] = daily_sales['quantity'] * daily_sales['unit_price']
        daily_sales.to_csv(data_dir / "raw" / f"sales_{date_str}.csv", index=False)

    # Create product master data
    products = pd.DataFrame({
        'product_code': ['A001', 'A002', 'B001', 'B002', 'C001'],
        'product_name': ['Basic Widget', 'Premium Widget', 'Standard Gadget', 'Deluxe Gadget', 'Super Tool'],
        'category': ['Widgets', 'Widgets', 'Gadgets', 'Gadgets', 'Tools'],
        'cost_price': [5.00, 10.00, 15.00, 25.00, 50.00]
    })
    products.to_csv(data_dir / "products.csv", index=False)

    print(f"✓ Created 7 daily sales files in {data_dir}/raw/")
    print(f"✓ Created product master data: {data_dir}/products.csv")
    print()

def demo_automated_data_processing():
    """
    Demonstrate automated data processing pipeline
    """
    print("=== AUTOMATED DATA PROCESSING PIPELINE ===")
    print()

    data_dir = Path("automation_demo")

    print("1. Automated File Discovery and Processing")
    print("=" * 45)

    # Discover all raw data files
    raw_files = list((data_dir / "raw").glob("sales_*.csv"))
    print(f"Found {len(raw_files)} raw data files:")
    for file in sorted(raw_files):
        print(f"  • {file.name}")
    print()

    # Load and combine all files
    print("2. Batch Data Loading and Validation")
    print("=" * 45)

    all_sales = []
    processing_log = []

    for file_path in sorted(raw_files):
        try:
            # Load file
            df = pd.read_csv(file_path)

            # Basic validation
            required_columns = ['timestamp', 'customer_id', 'product_code', 'quantity', 'unit_price', 'total_amount']
            missing_columns = set(required_columns) - set(df.columns)

            if missing_columns:
                error_msg = f"Missing columns in {file_path.name}: {missing_columns}"
                print(f"❌ {error_msg}")
                processing_log.append({"file": file_path.name, "status": "error", "message": error_msg})
                continue

            # Data quality checks
            if df['total_amount'].isnull().any():
                print(f"⚠️  Null values in total_amount for {file_path.name}")

            if (df['quantity'] <= 0).any():
                print(f"⚠️  Invalid quantities in {file_path.name}")

            # Add metadata
            df['source_file'] = file_path.name
            df['processed_at'] = datetime.now()

            all_sales.append(df)
            processing_log.append({"file": file_path.name, "status": "success", "records": len(df)})
            print(f"✓ Processed {file_path.name}: {len(df)} records")

        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            print(f"❌ {error_msg}")
            processing_log.append({"file": file_path.name, "status": "error", "message": error_msg})

    # Combine all data
    if all_sales:
        combined_sales = pd.concat(all_sales, ignore_index=True)
        print(f"\n✓ Combined dataset: {len(combined_sales)} total records")

        # Save processing log
        with open(data_dir / "processing_log.json", "w") as f:
            json.dump(processing_log, f, indent=2, default=str)

        return combined_sales
    else:
        print("❌ No data files successfully processed")
        return None

def demo_advanced_pandas_operations(sales_data):
    """
    Demonstrate advanced pandas operations for data analysis
    """
    print("\n=== ADVANCED PANDAS OPERATIONS ===")
    print()

    if sales_data is None:
        print("No data available for advanced operations")
        return

    # Convert timestamp to datetime
    sales_data['timestamp'] = pd.to_datetime(sales_data['timestamp'])
    sales_data['date'] = sales_data['timestamp'].dt.date
    sales_data['hour'] = sales_data['timestamp'].dt.hour

    print("1. Advanced Grouping and Aggregation")
    print("=" * 40)

    # Multi-level aggregation
    daily_summary = sales_data.groupby(['date', 'region']).agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_id': 'nunique'
    }).round(2)

    # Flatten column names
    daily_summary.columns = ['total_sales', 'avg_transaction', 'transaction_count', 'total_quantity', 'unique_customers']
    daily_summary = daily_summary.reset_index()

    print("Daily sales summary by region:")
    print(daily_summary.head(10))
    print()

    print("2. Rolling Calculations and Time Series")
    print("=" * 40)

    # Calculate rolling averages
    hourly_sales = sales_data.groupby(['date', 'hour'])['total_amount'].sum().reset_index()
    hourly_sales['datetime'] = pd.to_datetime(hourly_sales['date'].astype(str) + ' ' + hourly_sales['hour'].astype(str) + ':00:00')
    hourly_sales = hourly_sales.sort_values('datetime')

    # Rolling metrics
    hourly_sales['rolling_3h_avg'] = hourly_sales['total_amount'].rolling(window=3, min_periods=1).mean()
    hourly_sales['cumulative_daily'] = hourly_sales.groupby('date')['total_amount'].cumsum()

    print("Hourly sales with rolling averages:")
    print(hourly_sales[['datetime', 'total_amount', 'rolling_3h_avg', 'cumulative_daily']].head(10))
    print()

    print("3. Advanced Filtering and Transformation")
    print("=" * 40)

    # Load product data for enrichment
    products = pd.read_csv("automation_demo/products.csv")

    # Merge with product data
    enriched_sales = sales_data.merge(products, on='product_code', how='left')
    enriched_sales['profit'] = enriched_sales['total_amount'] - (enriched_sales['quantity'] * enriched_sales['cost_price'])

    # Complex filtering
    high_value_transactions = enriched_sales[
        (enriched_sales['total_amount'] > enriched_sales['total_amount'].quantile(0.8)) &
        (enriched_sales['profit'] > 0)
    ]

    print(f"High-value profitable transactions: {len(high_value_transactions)} out of {len(enriched_sales)}")
    print()

    # Pivot table analysis
    category_performance = pd.pivot_table(
        enriched_sales,
        values=['total_amount', 'profit', 'quantity'],
        index='category',
        columns='region',
        aggfunc='sum',
        fill_value=0
    ).round(2)

    print("Category performance by region (total sales):")
    print(category_performance['total_amount'])
    print()

    return enriched_sales

def demo_automated_reporting(sales_data):
    """
    Demonstrate automated report generation
    """
    print("=== AUTOMATED REPORT GENERATION ===")
    print()

    if sales_data is None:
        print("No data available for reporting")
        return

    data_dir = Path("automation_demo")

    print("1. Generating Business Intelligence Report")
    print("=" * 45)

    # Calculate key metrics
    total_revenue = sales_data['total_amount'].sum()
    total_transactions = len(sales_data)
    avg_transaction_value = sales_data['total_amount'].mean()
    unique_customers = sales_data['customer_id'].nunique()

    # Top performers
    top_products = sales_data.groupby('product_code')['total_amount'].sum().nlargest(3)
    top_regions = sales_data.groupby('region')['total_amount'].sum().nlargest(3)
    top_sales_reps = sales_data.groupby('sales_rep')['total_amount'].sum().nlargest(3)

    # Generate report
    report_content = f"""
AUTOMATED SALES REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Period: {sales_data['timestamp'].min().date()} to {sales_data['timestamp'].max().date()}

EXECUTIVE SUMMARY
================
Total Revenue: ${total_revenue:,.2f}
Total Transactions: {total_transactions:,}
Average Transaction Value: ${avg_transaction_value:.2f}
Unique Customers: {unique_customers:,}

TOP PERFORMERS
==============
Products by Revenue:
{chr(10).join([f"  {idx}: ${value:,.2f}" for idx, value in top_products.items()])}

Regions by Revenue:
{chr(10).join([f"  {idx}: ${value:,.2f}" for idx, value in top_regions.items()])}

Sales Representatives:
{chr(10).join([f"  {idx}: ${value:,.2f}" for idx, value in top_sales_reps.items()])}

DAILY TRENDS
============
"""

    # Add daily trends
    daily_totals = sales_data.groupby(sales_data['timestamp'].dt.date)['total_amount'].sum()
    for date, total in daily_totals.items():
        report_content += f"{date}: ${total:,.2f}\n"

    # Save report
    report_path = data_dir / "reports" / f"sales_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"✓ Report saved to: {report_path}")
    print("\nReport preview:")
    print(report_content[:500] + "...")
    print()

    print("2. Generating Data Export Files")
    print("=" * 35)

    # Export summary data
    summary_data = sales_data.groupby(['date', 'product_code', 'region']).agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'customer_id': 'nunique'
    }).reset_index()

    summary_path = data_dir / "processed" / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.csv"
    summary_data.to_csv(summary_path, index=False)
    print(f"✓ Summary data exported to: {summary_path}")

    # Export for external systems
    external_export = sales_data[['timestamp', 'customer_id', 'product_code', 'total_amount', 'region']].copy()
    external_export['export_date'] = datetime.now().strftime('%Y-%m-%d')

    external_path = data_dir / "processed" / f"external_export_{datetime.now().strftime('%Y%m%d')}.csv"
    external_export.to_csv(external_path, index=False)
    print(f"✓ External export created: {external_path}")

def demo_workflow_automation():
    """
    Demonstrate complete workflow automation
    """
    print("\n=== COMPLETE WORKFLOW AUTOMATION ===")
    print()

    print("1. Creating Automated Processing Script")
    print("=" * 40)

    # Create a standalone automation script
    automation_script = '''#!/usr/bin/env python3
"""
Automated Daily Sales Processing Script
Run this script daily to process new sales data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

def process_daily_sales():
    """Main processing function"""
    print(f"Starting daily sales processing: {datetime.now()}")

    data_dir = Path("automation_demo")

    # Find unprocessed files
    raw_files = list((data_dir / "raw").glob("sales_*.csv"))
    processed_log_path = data_dir / "processed_files.json"

    # Load list of already processed files
    processed_files = []
    if processed_log_path.exists():
        with open(processed_log_path, "r") as f:
            processed_files = json.load(f)

    # Find new files
    new_files = [f for f in raw_files if f.name not in processed_files]

    if not new_files:
        print("No new files to process")
        return

    print(f"Processing {len(new_files)} new files...")

    # Process each new file
    for file_path in new_files:
        try:
            df = pd.read_csv(file_path)
            # Add processing logic here
            print(f"Processed {file_path.name}: {len(df)} records")
            processed_files.append(file_path.name)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Update processed files log
    with open(processed_log_path, "w") as f:
        json.dump(processed_files, f, indent=2)

    print("Daily processing complete")

if __name__ == "__main__":
    process_daily_sales()
'''

    script_path = Path("automation_demo/daily_processor.py")
    with open(script_path, "w") as f:
        f.write(automation_script)

    print(f"✓ Created automation script: {script_path}")

    print("\n2. Creating Configuration Files")
    print("=" * 35)

    # Create configuration for automation
    config = {
        "data_sources": {
            "sales_data": "automation_demo/raw/",
            "product_data": "automation_demo/products.csv"
        },
        "output_paths": {
            "processed": "automation_demo/processed/",
            "reports": "automation_demo/reports/"
        },
        "processing_rules": {
            "file_pattern": "sales_*.csv",
            "required_columns": ["timestamp", "customer_id", "product_code", "total_amount"],
            "validation_checks": ["null_values", "negative_amounts", "duplicate_transactions"]
        },
        "reporting": {
            "frequency": "daily",
            "email_recipients": ["manager@company.com"],
            "alert_thresholds": {
                "min_daily_revenue": 1000,
                "max_transaction_value": 10000
            }
        }
    }

    config_path = Path("automation_demo/config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created configuration file: {config_path}")

    print("\n3. Demonstrating Error Handling")
    print("=" * 35)

    # Create a file with errors to test error handling
    bad_data = pd.DataFrame({
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', 'invalid_date'],
        'customer_id': [1001, 1002, None],
        'product_code': ['A001', 'B001', ''],
        'total_amount': [99.99, -50.00, 'invalid']  # Negative and invalid values
    })

    bad_file_path = Path("automation_demo/raw/sales_bad_data.csv")
    bad_data.to_csv(bad_file_path, index=False)
    print(f"✓ Created test file with data quality issues: {bad_file_path}")

    # Demonstrate robust error handling
    try:
        df = pd.read_csv(bad_file_path)
        print("\nData quality check results:")

        # Check for missing values
        missing_report = df.isnull().sum()
        print(f"Missing values per column:\n{missing_report}")

        # Check for invalid dates
        try:
            pd.to_datetime(df['timestamp'])
        except:
            print("❌ Invalid date formats detected")

        # Check for negative amounts
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                print(f"❌ Negative values detected in {col}")

    except Exception as e:
        print(f"❌ Error processing bad data file: {e}")

def demo_scheduling_and_monitoring():
    """
    Demonstrate scheduling and monitoring concepts
    """
    print("\n=== SCHEDULING AND MONITORING ===")
    print()

    print("1. Scheduling Options")
    print("=" * 25)

    print("Common scheduling approaches:")
    print("• Cron jobs (Linux/Mac): Schedule scripts to run at specific times")
    print("• Task Scheduler (Windows): GUI-based task scheduling")
    print("• Cloud services: AWS Lambda, Google Cloud Functions")
    print("• Workflow orchestrators: Apache Airflow, Prefect")
    print()

    print("Example cron job (runs daily at 9 AM):")
    print("0 9 * * * /usr/bin/python3 /path/to/daily_processor.py")
    print()

    print("2. Monitoring and Alerting")
    print("=" * 30)

    # Create a simple monitoring example
    monitoring_script = '''
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

def check_data_quality(file_path):
    """Check data quality and send alerts if needed"""
    df = pd.read_csv(file_path)

    issues = []

    # Check data freshness
    latest_timestamp = pd.to_datetime(df['timestamp']).max()
    hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600

    if hours_old > 24:
        issues.append(f"Data is {hours_old:.1f} hours old")

    # Check for anomalies
    daily_total = df['total_amount'].sum()
    if daily_total < 1000:  # Below expected threshold
        issues.append(f"Low daily revenue: ${daily_total:.2f}")

    # Send alert if issues found
    if issues:
        alert_message = "Data Quality Alert:\\n" + "\\n".join(issues)
        print(f"ALERT: {alert_message}")
        # In production: send_email_alert(alert_message)

    return len(issues) == 0

# Example usage
# check_data_quality("automation_demo/processed/daily_summary.csv")
'''

    print("Example monitoring function:")
    print(monitoring_script)

    print("\n3. Logging and Audit Trail")
    print("=" * 30)

    # Create audit log
    audit_log = {
        "timestamp": datetime.now().isoformat(),
        "process": "automated_sales_processing",
        "status": "completed",
        "files_processed": 7,
        "records_processed": 756,
        "errors": 0,
        "warnings": 1,
        "execution_time_seconds": 12.5,
        "next_scheduled_run": (datetime.now() + timedelta(days=1)).isoformat()
    }

    audit_path = Path("automation_demo/audit_log.json")
    with open(audit_path, "w") as f:
        json.dump(audit_log, f, indent=2)

    print(f"✓ Created audit log: {audit_path}")
    print("Audit log content:")
    for key, value in audit_log.items():
        print(f"  {key}: {value}")

def main():
    """
    Main demo execution function
    """
    print("Welcome to DataSci 217 - Lecture 09 Live Demo!")
    print("Automation and Advanced Data Manipulation")
    print("=" * 60)
    print()

    # Create sample data
    create_automation_datasets()

    # Run automation demos
    print("PART 1: DATA PROCESSING AUTOMATION")
    print("=" * 50)
    sales_data = demo_automated_data_processing()

    if sales_data is not None:
        print("\nPART 2: ADVANCED PANDAS OPERATIONS")
        print("=" * 50)
        enriched_data = demo_advanced_pandas_operations(sales_data)

        print("\nPART 3: AUTOMATED REPORTING")
        print("=" * 50)
        demo_automated_reporting(sales_data)

    print("\nPART 4: WORKFLOW AUTOMATION")
    print("=" * 50)
    demo_workflow_automation()

    print("\nPART 5: SCHEDULING AND MONITORING")
    print("=" * 50)
    demo_scheduling_and_monitoring()

    print("\n" + "="*60)
    print("Demo complete!")
    print("\nKey takeaways:")
    print("1. Automation reduces manual errors and saves time")
    print("2. Always include data validation and error handling")
    print("3. Advanced pandas operations enable complex analysis")
    print("4. Automated reporting provides consistent insights")
    print("5. Monitoring and alerting prevent silent failures")
    print("6. Configuration files make automation flexible")
    print("7. Audit trails ensure accountability and debugging")

if __name__ == "__main__":
    main()