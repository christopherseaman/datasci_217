# Assignment 8 Data Schema

This document describes the healthcare/insurance EHR data structure used in Assignment 8.

## Overview

The assignment uses three main datasets representing a healthcare system:
- **Providers**: Healthcare professionals (doctors, nurses, etc.)
- **Facilities**: Healthcare locations (hospitals, clinics, etc.)
- **Encounters**: Patient visits/insurance claims

## Data Files

### `provider_data.csv`

Healthcare provider (clinician) information.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `provider_id` | string | Unique provider identifier | `PR0001` |
| `provider_name` | string | Provider full name | `"Dr. Alex Smith"` |
| `provider_type` | string | Type of provider | `"Physician"`, `"Nurse Practitioner"`, `"Physician Assistant"`, `"Nurse"`, `"Therapist"` |
| `facility_id` | string | Facility where provider works (links to `facility_data.csv`) | `FAC001` |
| `specialty` | string | Medical specialty | `"Cardiology"`, `"Oncology"`, `"Primary Care"`, etc. |
| `years_experience` | int | Years of professional experience | `5`, `15`, `30` |
| `license_number` | string | Professional license number | `"LIC000001"` |

**Example Row:**
```csv
provider_id,provider_name,provider_type,facility_id,specialty,years_experience,license_number
PR0001,Alex Smith,Physician,FAC001,Cardiology,15,LIC000001
```

### `facility_data.csv`

Healthcare facility (hospital, clinic, etc.) information.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `facility_id` | string | Unique facility identifier | `FAC001` |
| `facility_name` | string | Facility name | `"City General Hospital"` |
| `facility_type` | string | Type of facility | `"Hospital"`, `"Clinic"`, `"Urgent Care"`, `"Specialty Center"` |
| `region` | string | Geographic region | `"North"`, `"South"`, `"East"`, `"West"` |
| `beds` | int | Number of beds (0 for non-inpatient facilities) | `250`, `50`, `0` |
| `established_date` | string | Date facility was established | `"1985-03-15"` |

**Example Row:**
```csv
facility_id,facility_name,facility_type,region,beds,established_date
FAC001,City General Hospital,Hospital,North,250,1985-03-15
```

### `encounter_data.csv`

Patient encounters (visits/insurance claims) data.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `encounter_id` | string | Unique encounter identifier | `ENC00001` |
| `patient_id` | string | Patient identifier | `PAT00001` |
| `provider_id` | string | Provider who saw patient (links to `provider_data.csv`) | `PR0001` |
| `facility_id` | string | Facility where encounter occurred (links to `facility_data.csv`) | `FAC001` |
| `encounter_date` | string | Date of encounter | `"2023-06-15"` |
| `encounter_type` | string | Type of encounter | `"Office Visit"`, `"Emergency"`, `"Inpatient"`, `"Outpatient Procedure"`, `"Lab"` |
| `diagnosis_code` | string | ICD-10 diagnosis code | `"E11.9"` (Type 2 diabetes), `"I10"` (Hypertension) |
| `procedure_code` | string | CPT procedure code | `"99213"` (Office visit), `"36415"` (Blood draw) |
| `service_charge` | float | Total charge for the service | `250.00`, `1250.50` |
| `insurance_paid` | float | Amount covered by insurance | `200.00`, `1000.40` |
| `patient_paid` | float | Amount paid by patient | `50.00`, `250.10` |
| `region` | string | Geographic region (derived from facility) | `"North"`, `"South"`, `"East"`, `"West"` |

**Note:** `service_charge = insurance_paid + patient_paid` (insurance typically covers 70-90%)

**Example Row:**
```csv
encounter_id,patient_id,provider_id,facility_id,encounter_date,encounter_type,diagnosis_code,procedure_code,service_charge,insurance_paid,patient_paid,region
ENC00001,PAT00001,PR0001,FAC001,2023-06-15,Office Visit,E11.9,99213,250.00,200.00,50.00,North
```

## Common Codes Reference

### ICD-10 Diagnosis Codes (used in this dataset)

| Code | Description |
|------|-------------|
| `E11.9` | Type 2 diabetes mellitus without complications |
| `I10` | Essential (primary) hypertension |
| `M79.3` | Panniculitis, unspecified |
| `K21.9` | Gastro-esophageal reflux disease without esophagitis |
| `J44.9` | Chronic obstructive pulmonary disease, unspecified |
| `N18.9` | Chronic kidney disease, unspecified |
| `F32.9` | Major depressive disorder, single episode, unspecified |
| `G43.9` | Migraine, unspecified |
| `M25.5` | Pain in joint |
| `H52.9` | Unspecified disorder of refraction |

### CPT Procedure Codes (used in this dataset)

| Code | Description |
|------|-------------|
| `99213` | Office visit, established patient (low complexity) |
| `99214` | Office visit, established patient (moderate complexity) |
| `99283` | Emergency department visit (moderate severity) |
| `36415` | Routine venipuncture for collection of specimen(s) |
| `80053` | Comprehensive metabolic panel |
| `85025` | Complete blood count (CBC) |
| `93000` | Electrocardiogram, routine ECG |
| `72040` | Radiologic examination, chest |
| `70450` | CT head/brain without contrast |
| `70551` | MRI brain without contrast |

## Data Relationships

```
facility_data.csv (facility_id)
    ↑
    │
provider_data.csv (facility_id, provider_id)
    ↑
    │
encounter_data.csv (facility_id, provider_id)
```

- Each provider belongs to one facility (`provider.facility_id` → `facility.facility_id`)
- Each encounter has one provider and one facility (`encounter.provider_id` → `provider.provider_id`, `encounter.facility_id` → `facility.facility_id`)
- Encounter `region` is derived from the facility's region

## Typical Analysis Patterns

When analyzing this data, you'll commonly:

1. **Group by facility** to analyze:
   - Total encounter charges per facility
   - Average provider experience per facility
   - Number of encounters per facility

2. **Group by provider** to analyze:
   - Number of encounters per provider
   - Total charges per provider
   - Average insurance coverage per provider

3. **Group by region** to analyze:
   - Regional healthcare utilization
   - Regional cost patterns
   - Regional provider distribution

4. **Group by specialty** to analyze:
   - Provider distribution across specialties
   - Encounter patterns by specialty
   - Cost analysis by specialty

5. **Cross-tabulation** of:
   - Facility type × Region
   - Encounter type × Diagnosis code
   - Provider type × Specialty

## Data Generation Notes

- **Providers**: 500 providers across 10 facilities
- **Facilities**: 10 facilities across 4 regions
- **Encounters**: 5,000 encounters over 1 year (2023)
- **Random seed**: 42 (for reproducibility)
- **Insurance coverage**: Randomly 70-90% of service charge
- **Patient payment**: Remaining amount after insurance coverage

