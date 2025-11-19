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
    echo "✅ Taxi trip data download complete!"
    echo "File: $DATA_DIR/yellow_tripdata_2023-01.parquet"
    echo ""
    echo "Note: Parquet files require pyarrow. Install with: pip install pyarrow"
else
    echo "❌ Taxi trip data download failed"
    exit 1
fi

# Download NYC Taxi Zone Lookup Table
echo "Downloading NYC Taxi Zone Lookup Table..."
echo "Source: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
echo ""

ZONE_URL="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_FILE="taxi_zone_lookup.csv"

echo "Downloading: $ZONE_FILE"
curl -L -o "$DATA_DIR/$ZONE_FILE" "$ZONE_URL"

if [ -f "$DATA_DIR/$ZONE_FILE" ]; then
    echo "✅ Zone lookup download complete!"
    echo "File: $DATA_DIR/$ZONE_FILE"
    echo "Number of zones: $(tail -n +2 "$DATA_DIR/$ZONE_FILE" | wc -l)"
    echo ""
else
    echo "❌ Zone lookup download failed"
    exit 1
fi

echo "✅ All downloads complete!"

