
# Public Datasets

- https://github.com/awesomedata/awesome-public-datasets
- https://catalog.data.gov/dataset/?tags=sensors
- https://physionet.org/about/database/#open

## 1. NYC Taxi Trip Dataset
**Source:** NYC Taxi & Limousine Commission  
**Description:** Detailed taxi trip records including pickup/dropoff timestamps and locations, fare amounts, passenger counts, and trip distances. Contains millions of rows with data from Yellow, Green, and For-Hire vehicles.  
**Special Tools:** None required - standard CSV/Parquet files  
**McKinney Techniques:**
- DateTime manipulation (parsing timestamps, extracting hour/day/month)
- GroupBy operations (aggregate by hour, location, day of week)
- Data cleaning (outliers like 24+ hour trips, invalid coordinates)
- Merging/joining (combining with zone lookup tables)
- Memory management for large files (chunking, dtype optimization)

## 2. California Wildfire Dataset
**Source:** CAL FIRE / data.gov  
**Description:** Historical fire incidents dating back to 1878, including fire perimeter data, acreage burned, containment dates, and agency response information. Mix of complete and incomplete historical records.  
**Special Tools:** None required - CSV format  
**McKinney Techniques:**
- Handling missing data (incomplete historical records)
- Time series analysis (trend analysis over decades)
- Categorical data processing (agency codes, fire causes)
- Aggregation and pivoting (fires by year, county, season)
- String manipulation (parsing location names)

## 3. Chicago Beach Weather Sensors
**Source:** data.gov (sensors category)  
**Description:** Real-time weather sensor readings from Lake Michigan beaches including temperature, wind speed, water conditions, and air quality measurements. Contains typical sensor issues like dropouts and irregular sampling.  
**Special Tools:** None required - CSV/JSON formats available  
**McKinney Techniques:**
- Resampling irregular time series data
- Handling missing values (sensor dropouts)
- Rolling window calculations (moving averages)
- Multiple dataset merging (different sensor streams)
- Data type conversions (mixed numeric/string formats)

## 4. MIT-BIH Arrhythmia Database
**Source:** PhysioNet  
**Description:** ECG recordings with beat annotations, containing multi-lead cardiac signals sampled at high frequency (360 Hz). Includes patient metadata and clinical annotations for various heart conditions.  
**Special Tools:** `wfdb` package for reading PhysioNet format (easily converts to pandas DataFrames)  
**McKinney Techniques:**
- High-frequency time series handling
- Multi-index operations (patient ID + time)
- Resampling and interpolation
- Window functions for feature extraction
- Joining annotation data with signal data
- Boolean indexing for event detection

All datasets provide real-world complexity while remaining manageable for exam settings, with clear applications of core pandas functionality from McKinney's text.