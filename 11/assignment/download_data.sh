#!/bin/bash
set -euo pipefail

DATA_DIR="data"
MANIFEST="$DATA_DIR/release_manifest.json"
RELEASE="$DATA_DIR/chicago_beach_sensors_2022_2024.csv"
EXPECTED_SHA256="7209cddd9b80e9475f9af17169b935e1ac2ef4a7a32fb72963ad0566b3474139"
EXPECTED_BYTES="4731351"

if [[ ! -f "$MANIFEST" || ! -f "$RELEASE" ]]; then
    printf 'Missing committed assignment data. Restore data/ from the course repository.\n' >&2
    exit 1
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_COMMAND=(python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_COMMAND=(python3)
elif command -v uv >/dev/null 2>&1; then
    PYTHON_COMMAND=(uv run --no-project --python 3.12.13 python)
else
    printf 'Python or uv is required to verify the release manifest.\n' >&2
    exit 1
fi

"${PYTHON_COMMAND[@]}" - "$MANIFEST" "$RELEASE" "$EXPECTED_SHA256" "$EXPECTED_BYTES" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

manifest_path, release_path, expected_hash, expected_size = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)

release_bytes = Path(release_path).read_bytes()
actual_hash = sha256(release_bytes).hexdigest()
actual_size = len(release_bytes)
if actual_hash != expected_hash:
    raise SystemExit(f"Release checksum mismatch: {release_path}")
if actual_size != int(expected_size):
    raise SystemExit(
        f"Release size mismatch: expected {expected_size} bytes, observed {actual_size} bytes"
    )

assert manifest["schema"] == "datasci217/assignment11-release/v1"
assert manifest["release_filename"] == "chicago_beach_sensors_2022_2024.csv"
assert manifest["release_sha256"] == expected_hash
assert manifest["release_byte_size"] == int(expected_size)
assert manifest["row_count"] == 50895
assert manifest["column_count"] == 15
assert manifest["columns"] == [
    "station_name",
    "measurement_timestamp",
    "air_temperature_c",
    "wet_bulb_temperature_c",
    "relative_humidity_pct",
    "rain_intensity_mm_per_hour",
    "interval_rain_mm",
    "total_rain_mm",
    "precipitation_type_code",
    "wind_direction_deg",
    "wind_speed_mps",
    "maximum_wind_speed_mps",
    "barometric_pressure_hpa",
    "solar_radiation_w_m2",
    "battery_voltage_v",
]
assert manifest["stations"] == [
    "Foster Weather Station",
    "Oak Street Weather Station",
]
assert manifest["source_timezone"] == "America/Chicago"

print(f"Verified frozen release and manifest: {release_path} ({actual_size} bytes)")
PY
