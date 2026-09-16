#!/bin/bash

set -euo pipefail

# Release URLs use the public main branch for the published course materials.
readonly BASE_URL="https://raw.githubusercontent.com/christopherseaman/datasci_217/main/11/demo/data"
readonly DATA_DIR="data"

mkdir -p "$DATA_DIR"

download_and_verify() {
    local filename="$1"
    local expected_sha256="$2"
    local destination="$DATA_DIR/$filename"

    if [[ ! -f "$destination" ]]; then
        curl --fail --location --output "$destination" "$BASE_URL/$filename"
    fi

    local actual_sha256
    actual_sha256="$(sha256sum "$destination" | cut -d ' ' -f 1)"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        printf 'Hash mismatch for %s\nExpected: %s\nActual:   %s\n' \
            "$destination" "$expected_sha256" "$actual_sha256" >&2
        exit 1
    fi
    printf 'Verified %s\n' "$destination"
}

download_and_verify "yellow_taxi_2023_h1_event_sample.parquet" \
    "0a2fdc27ce787c5d042dc73b71eecd7d7dca326c97cebd3a78f9e68d2fe4c16f"
download_and_verify "yellow_taxi_2023_h1_zone_hour_counts.parquet" \
    "f1f55ea809119757ee26995ae1eae4ff6be21b70273194c35aa388cfbfe13ad3"
download_and_verify "taxi_zone_lookup.csv" \
    "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed"
download_and_verify "demo_release_manifest.json" \
    "558c28a8ab5a16769ac6ef9d170e7bd7f4ae4ef5d2a9e2b11fd2fb84d79b2c9d"

printf 'Frozen Lecture 11 data are ready in %s/\n' "$DATA_DIR"
