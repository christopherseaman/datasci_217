#!/bin/bash
# Download actual NYC TLC Taxi Trip Data
# Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "Downloading NYC TLC Yellow Taxi Trip Data..."
echo "Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
echo ""

# Download January 2023 Yellow Taxi data (Parquet format - smaller file)
# This is a sample month - you can download other months as needed
URL="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

echo "Downloading: yellow_tripdata_2023-01.parquet"
curl -L -o "$DATA_DIR/yellow_tripdata_2023-01.parquet" "$URL"

if [ -f "$DATA_DIR/yellow_tripdata_2023-01.parquet" ]; then
    echo "✅ Download complete!"
    echo "File: $DATA_DIR/yellow_tripdata_2023-01.parquet"
    echo ""
    echo "To convert to CSV (optional):"
    echo "  python -c \"import pandas as pd; df = pd.read_parquet('$DATA_DIR/yellow_tripdata_2023-01.parquet'); df.to_csv('$DATA_DIR/yellow_tripdata_2023-01.csv', index=False)\""
else
    echo "❌ Download failed"
    exit 1
fi

