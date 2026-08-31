#!/bin/bash

set -euo pipefail

# Development URLs use 2026-refresh until the immutable annual release tag is frozen.
readonly BASE_URL="https://raw.githubusercontent.com/christopherseaman/datasci_217/2026-refresh/11/demo/data"
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
    "750bcc85f0267f9189dc9842ef44827168c384d4a7e5a8678e9a996348fc4b7d"
download_and_verify "yellow_taxi_2023_h1_zone_hour_counts.parquet" \
    "6c5658bd1d076930a9c552372fb3fb3d5dd71efbc4e4a736b5695e14f5d7b574"
download_and_verify "taxi_zone_lookup.csv" \
    "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed"
download_and_verify "demo_release_manifest.json" \
    "553a1d732c0e0bdee9b8d79d7262a3f361109c23af6c33776f79ae661bca5fc6"

printf 'Frozen Lecture 11 data are ready in %s/\n' "$DATA_DIR"
