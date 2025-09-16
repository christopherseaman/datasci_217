#!/usr/bin/env python3
"""
Lecture 04 Live Demo: CLI Text Processing and Python Functions
DataSci 217 - Hands-on demonstration of practical data workflows

This demo shows real-world CLI text processing and Python function design
for data analysis tasks.
"""

import subprocess
import pandas as pd
from pathlib import Path
import tempfile
import os

def create_sample_data():
    """
    Create sample datasets for CLI processing demonstration
    """
    # Create a temporary directory for demo files
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)

    # Sample log file (server access logs)
    log_data = """192.168.1.100 - - [2024-01-15 10:23:45] "GET /api/users HTTP/1.1" 200 1234
192.168.1.101 - - [2024-01-15 10:24:12] "POST /api/login HTTP/1.1" 401 89
192.168.1.100 - - [2024-01-15 10:24:30] "GET /api/data HTTP/1.1" 200 5678
10.0.0.50 - - [2024-01-15 10:25:01] "GET /api/users HTTP/1.1" 500 234
192.168.1.102 - - [2024-01-15 10:25:15] "DELETE /api/users/123 HTTP/1.1" 204 0
192.168.1.100 - - [2024-01-15 10:26:00] "GET /api/data HTTP/1.1" 200 3456
10.0.0.50 - - [2024-01-15 10:26:30] "POST /api/data HTTP/1.1" 201 789"""

    with open(demo_dir / "server_logs.txt", "w") as f:
        f.write(log_data)

    # Sample CSV file (sales data)
    csv_data = """Date,Customer,Product,Amount,Region
2024-01-15,Alice Johnson,Widget A,150.00,North
2024-01-15,Bob Smith,Widget B,200.50,South
2024-01-15,Carol Davis,Widget A,150.00,East
2024-01-16,Alice Johnson,Widget C,300.75,North
2024-01-16,David Wilson,Widget B,200.50,West
2024-01-16,Bob Smith,Widget A,150.00,South
2024-01-17,Carol Davis,Widget C,300.75,East"""

    with open(demo_dir / "sales_data.csv", "w") as f:
        f.write(csv_data)

    # Sample employee data (fixed-width format)
    employee_data = """ID  Name           Department  Salary
001 Alice Johnson  Engineering 75000
002 Bob Smith      Sales       65000
003 Carol Davis    Marketing   70000
004 David Wilson   Engineering 80000
005 Eve Brown      Sales       60000
006 Frank Miller   Marketing   68000"""

    with open(demo_dir / "employees.txt", "w") as f:
        f.write(employee_data)

    print("Sample data files created in demo_data/:")
    print("- server_logs.txt: Web server access logs")
    print("- sales_data.csv: Customer sales transactions")
    print("- employees.txt: Employee data (fixed-width)")
    print()

def demo_cli_text_processing():
    """
    Demonstrate practical CLI text processing techniques
    """
    print("=== CLI TEXT PROCESSING DEMO ===")
    print()

    # Ensure we have demo data
    if not Path("demo_data").exists():
        create_sample_data()

    print("1. GREP: Finding patterns in log files")
    print("=" * 40)

    # Find all successful requests (200 status)
    print("Finding all successful requests (status 200):")
    print("Command: grep '200' demo_data/server_logs.txt")
    result = subprocess.run(['grep', '200', 'demo_data/server_logs.txt'],
                          capture_output=True, text=True)
    print(result.stdout)

    # Find all requests from specific IP
    print("Finding requests from 192.168.1.100:")
    print("Command: grep '192.168.1.100' demo_data/server_logs.txt")
    result = subprocess.run(['grep', '192.168.1.100', 'demo_data/server_logs.txt'],
                          capture_output=True, text=True)
    print(result.stdout)

    print("2. CUT: Extracting specific columns")
    print("=" * 40)

    # Extract just the IP addresses (first field)
    print("Extracting IP addresses from logs:")
    print("Command: cut -d' ' -f1 demo_data/server_logs.txt")
    result = subprocess.run(['cut', '-d', ' ', '-f1', 'demo_data/server_logs.txt'],
                          capture_output=True, text=True)
    print(result.stdout)

    # Extract customer names from CSV
    print("Extracting customer names from sales data:")
    print("Command: cut -d',' -f2 demo_data/sales_data.csv")
    result = subprocess.run(['cut', '-d', ',', '-f2', 'demo_data/sales_data.csv'],
                          capture_output=True, text=True)
    print(result.stdout)

    print("3. SORT and UNIQ: Data analysis operations")
    print("=" * 40)

    # Count unique IP addresses
    print("Counting unique IP addresses:")
    print("Command: cut -d' ' -f1 demo_data/server_logs.txt | sort | uniq -c")
    p1 = subprocess.run(['cut', '-d', ' ', '-f1', 'demo_data/server_logs.txt'],
                       capture_output=True, text=True)
    p2 = subprocess.run(['sort'], input=p1.stdout, capture_output=True, text=True)
    p3 = subprocess.run(['uniq', '-c'], input=p2.stdout, capture_output=True, text=True)
    print(p3.stdout)

    # Sort sales data by amount
    print("Products sorted by popularity:")
    print("Command: cut -d',' -f3 demo_data/sales_data.csv | tail -n +2 | sort | uniq -c | sort -nr")
    p1 = subprocess.run(['cut', '-d', ',', '-f3', 'demo_data/sales_data.csv'],
                       capture_output=True, text=True)
    p2 = subprocess.run(['tail', '-n', '+2'], input=p1.stdout, capture_output=True, text=True)
    p3 = subprocess.run(['sort'], input=p2.stdout, capture_output=True, text=True)
    p4 = subprocess.run(['uniq', '-c'], input=p3.stdout, capture_output=True, text=True)
    p5 = subprocess.run(['sort', '-nr'], input=p4.stdout, capture_output=True, text=True)
    print(p5.stdout)

def demo_python_functions():
    """
    Demonstrate effective Python function design for data analysis
    """
    print("=== PYTHON FUNCTIONS FOR DATA ANALYSIS ===")
    print()

    # Function 1: Data validation
    def validate_sales_record(record):
        """
        Validate a sales record for completeness and data quality

        Args:
            record (dict): Sales record with date, customer, product, amount, region

        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        required_fields = ['Date', 'Customer', 'Product', 'Amount', 'Region']
        for field in required_fields:
            if field not in record or not record[field]:
                errors.append(f"Missing required field: {field}")

        # Validate amount is numeric
        if 'Amount' in record:
            try:
                amount = float(record['Amount'])
                if amount <= 0:
                    errors.append("Amount must be positive")
            except ValueError:
                errors.append("Amount must be a valid number")

        # Validate region
        valid_regions = ['North', 'South', 'East', 'West']
        if 'Region' in record and record['Region'] not in valid_regions:
            errors.append(f"Invalid region: {record['Region']}")

        return len(errors) == 0, errors

    # Function 2: Data processing
    def process_log_file(log_file_path):
        """
        Process web server log file and extract key metrics

        Args:
            log_file_path (str): Path to log file

        Returns:
            dict: Summary statistics
        """
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        total_requests = len(lines)
        status_codes = {}
        ip_addresses = set()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 9:
                ip = parts[0]
                status = parts[8].strip('"')

                ip_addresses.add(ip)
                status_codes[status] = status_codes.get(status, 0) + 1

        return {
            'total_requests': total_requests,
            'unique_ips': len(ip_addresses),
            'status_codes': status_codes,
            'success_rate': status_codes.get('200', 0) / total_requests * 100
        }

    # Function 3: Data transformation
    def calculate_customer_summary(sales_file_path):
        """
        Calculate customer purchase summaries from sales data

        Args:
            sales_file_path (str): Path to CSV sales file

        Returns:
            dict: Customer summaries
        """
        df = pd.read_csv(sales_file_path)

        customer_summary = df.groupby('Customer').agg({
            'Amount': ['sum', 'mean', 'count'],
            'Product': lambda x: list(x.unique())
        }).round(2)

        # Flatten column names
        customer_summary.columns = ['Total_Spent', 'Avg_Purchase', 'Purchase_Count', 'Products']

        return customer_summary.to_dict('index')

    # Demonstrate the functions
    print("1. Data Validation Function")
    print("=" * 30)

    # Test validation function
    test_record = {
        'Date': '2024-01-15',
        'Customer': 'Alice Johnson',
        'Product': 'Widget A',
        'Amount': '150.00',
        'Region': 'North'
    }

    is_valid, errors = validate_sales_record(test_record)
    print(f"Valid record: {is_valid}")
    if errors:
        print(f"Errors: {errors}")

    # Test with invalid record
    invalid_record = {
        'Date': '',
        'Customer': 'Bob Smith',
        'Amount': '-50',
        'Region': 'Mars'
    }

    is_valid, errors = validate_sales_record(invalid_record)
    print(f"\nInvalid record: {is_valid}")
    print(f"Errors: {errors}")

    print("\n2. Log Processing Function")
    print("=" * 30)

    if Path("demo_data/server_logs.txt").exists():
        log_stats = process_log_file("demo_data/server_logs.txt")
        print(f"Log Analysis Results:")
        for key, value in log_stats.items():
            print(f"  {key}: {value}")

    print("\n3. Customer Summary Function")
    print("=" * 30)

    if Path("demo_data/sales_data.csv").exists():
        customer_summaries = calculate_customer_summary("demo_data/sales_data.csv")
        print("Customer Purchase Summaries:")
        for customer, summary in customer_summaries.items():
            print(f"  {customer}:")
            print(f"    Total Spent: ${summary['Total_Spent']}")
            print(f"    Average Purchase: ${summary['Avg_Purchase']}")
            print(f"    Purchase Count: {summary['Purchase_Count']}")
            print(f"    Products: {summary['Products']}")
            print()

def demo_integration_workflow():
    """
    Demonstrate integrating CLI and Python for complete data workflow
    """
    print("=== INTEGRATED CLI + PYTHON WORKFLOW ===")
    print()

    print("Scenario: Analyze server logs and sales data together")
    print("=" * 50)

    # Step 1: Use CLI to preprocess logs
    print("Step 1: Extract error logs using CLI")
    print("Command: grep -v '200' demo_data/server_logs.txt")

    if Path("demo_data/server_logs.txt").exists():
        result = subprocess.run(['grep', '-v', '200', 'demo_data/server_logs.txt'],
                              capture_output=True, text=True)
        error_logs = result.stdout.strip().split('\n')
        print(f"Found {len(error_logs)} non-200 status requests")
        print()

    # Step 2: Process with Python
    print("Step 2: Process sales data with Python")
    if Path("demo_data/sales_data.csv").exists():
        df = pd.read_csv("demo_data/sales_data.csv")
        df['Amount'] = df['Amount'].astype(float)

        daily_sales = df.groupby('Date')['Amount'].sum()
        top_customer = df.groupby('Customer')['Amount'].sum().idxmax()

        print(f"Daily sales totals:")
        for date, total in daily_sales.items():
            print(f"  {date}: ${total:.2f}")
        print(f"Top customer: {top_customer}")
        print()

    # Step 3: Create combined report
    print("Step 3: Generate combined analysis report")

    report = f"""
DAILY OPERATIONS REPORT
======================
Date: 2024-01-15 to 2024-01-17

SERVER METRICS:
- Total requests processed: 7
- Error rate: {len(error_logs)/7*100:.1f}%
- Unique IP addresses: 3

SALES METRICS:
- Total revenue: ${daily_sales.sum():.2f}
- Average daily sales: ${daily_sales.mean():.2f}
- Top customer: {top_customer}

RECOMMENDATIONS:
- Monitor error rate (currently {len(error_logs)/7*100:.1f}%)
- Focus retention efforts on top customers
- Analyze regional sales patterns for expansion
"""

    print(report)

    # Save report
    with open("demo_data/daily_report.txt", "w") as f:
        f.write(report)
    print("Report saved to demo_data/daily_report.txt")

def main():
    """
    Main demo execution function
    """
    print("Welcome to DataSci 217 - Lecture 04 Live Demo!")
    print("CLI Text Processing and Python Functions")
    print("=" * 50)
    print()

    # Create sample data
    create_sample_data()

    # Run demos
    demo_cli_text_processing()
    print("\n" + "="*60 + "\n")

    demo_python_functions()
    print("\n" + "="*60 + "\n")

    demo_integration_workflow()
    print("\n" + "="*60 + "\n")

    print("Demo complete! Check the demo_data/ directory for all generated files.")
    print("\nKey takeaways:")
    print("1. CLI tools excel at quick data filtering and extraction")
    print("2. Python functions should be focused and well-documented")
    print("3. Combining CLI and Python creates powerful workflows")
    print("4. Always validate your data and handle edge cases")

if __name__ == "__main__":
    main()