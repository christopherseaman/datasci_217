#!/bin/bash
# Download Chicago Beach Weather Sensors Dataset
# Source: data.gov - Chicago Beach Weather Sensors
# This script ensures all students have the exact same dataset for auto-grading

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "Downloading Chicago Beach Weather Sensors dataset..."
echo "Source: data.gov / City of Chicago Data Portal"
echo ""

# Dataset: Beach Weather Stations - Automated Sensors
# Source: https://data.cityofchicago.org/Parks-Recreation/Beach-Weather-Stations-Automated-Sensors/k7hf-8y75/about_data
# View ID: k7hf-8y75

URL="https://data.cityofchicago.org/api/views/k7hf-8y75/rows.csv?accessType=DOWNLOAD"

OUTPUT_FILE="$DATA_DIR/beach_sensors.csv"

echo "Downloading from: $URL"
curl -L -o "$OUTPUT_FILE" "$URL" || {
    echo "❌ Download failed"
    echo ""
    echo "If the automatic download fails, you can manually download from:"
    echo "  https://data.cityofchicago.org/Parks-Recreation/Beach-Weather-Stations-Automated-Sensors/k7hf-8y75/about_data"
    echo "  Click 'Export' → 'CSV'"
    echo "  Save as: $OUTPUT_FILE"
    exit 1
}

if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    LINES=$(wc -l < "$OUTPUT_FILE" || echo "unknown")
    
    # Check if file is too small (likely an error page)
    if [ "$LINES" -lt 10 ]; then
        echo "❌ Download appears to have failed - file is too small ($LINES lines)"
        echo "This might be an error page. Please check the URL."
        rm -f "$OUTPUT_FILE"
        exit 1
    fi
    
    echo "✅ Download complete!"
    echo "File: $OUTPUT_FILE"
    echo "Size: $SIZE"
    echo "Rows: $LINES"
    echo ""
    echo "Dataset ready for analysis!"
else
    echo "❌ Download failed - file not created"
    exit 1
fi
