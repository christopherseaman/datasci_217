Clinical and Research Context Applications

This is bonus content for DataSci 217 - Lecture 11. These specialized techniques address challenges common in clinical and research environments.

Research Data Types and Challenges

Common Research Data Scenarios

**Reference:**
```python
def handle_research_data_challenges():
    """
    Demonstrate handling common research data challenges
    """
    print("=== COMMON RESEARCH DATA CHALLENGES ===")

    # Challenge 1: Missing data patterns in longitudinal studies
    print("\n1. LONGITUDINAL MISSING DATA")

    # Simulate longitudinal study data
    np.random.seed(42)
    participants = 100
    timepoints = 6

    # Create participant data with realistic missing patterns
    data = []
    for participant in range(participants):
        for timepoint in range(timepoints):
            # Simulate dropout - higher probability of missing at later timepoints
            if np.random.random() > (0.95 - timepoint * 0.1):
                continue  # Missing data point

            data.append({
                'participant_id': f'P{participant:03d}',
                'timepoint': timepoint,
                'outcome_score': np.random.normal(50 + timepoint * 2, 10),
                'age_at_baseline': np.random.randint(18, 80),
                'treatment_group': np.random.choice(['Control', 'Treatment'])
            })

    longitudinal_df = pd.DataFrame(data)

    print(f"Longitudinal dataset: {len(longitudinal_df)} observations")
    print(f"Unique participants: {longitudinal_df['participant_id'].nunique()}")
    print("Missing data pattern by timepoint:")

    missing_pattern = longitudinal_df.groupby('timepoint').size()
    for tp, count in missing_pattern.items():
        print(f"  Timepoint {tp}: {count} participants ({count/participants*100:.1f}%)")

    # Challenge 2: Multi-site data harmonization
    print("\n2. MULTI-SITE DATA HARMONIZATION")

    # Simulate data from different sites with slightly different protocols
    sites = ['Site_A', 'Site_B', 'Site_C']
    harmonized_data = []

    for site in sites:
        n_participants = np.random.randint(50, 150)

        for i in range(n_participants):
            record = {
                'site': site,
                'participant_id': f'{site}_{i:03d}',
                'age': np.random.normal(45, 15),
                'primary_outcome': np.random.normal(100, 20)
            }

            # Site-specific variations
            if site == 'Site_A':
                record['measurement_device'] = 'Device_X'
                record['primary_outcome'] += 5  # Systematic bias
            elif site == 'Site_B':
                record['measurement_device'] = 'Device_Y'
                record['secondary_measure'] = np.random.normal(50, 10)  # Extra measure
            else:  # Site_C
                record['measurement_device'] = 'Device_Z'
                record['age_category'] = 'Adult' if record['age'] >= 18 else 'Minor'

        harmonized_data.extend([record])

    multisite_df = pd.DataFrame(harmonized_data)

    print("Multi-site dataset characteristics:")
    print(multisite_df.groupby('site').agg({
        'participant_id': 'count',
        'age': 'mean',
        'primary_outcome': 'mean'
    }).round(2))

    # Challenge 3: Sensitive data handling
    print("\n3. SENSITIVE DATA PROTECTION")

    def create_deidentified_dataset(df, quasi_identifiers=['age'], direct_identifiers=['participant_id']):
        """
        Demonstrate basic de-identification techniques
        """
        df_deidentified = df.copy()

        # Remove direct identifiers
        for col in direct_identifiers:
            if col in df_deidentified.columns:
                df_deidentified[col] = df_deidentified[col].apply(lambda x: f'ID_{hash(str(x)) % 10000:04d}')

        # Generalize quasi-identifiers
        for col in quasi_identifiers:
            if col in df_deidentified.columns:
                if col == 'age':
                    # Age generalization to 5-year bins
                    df_deidentified[col] = (df_deidentified[col] // 5) * 5

        return df_deidentified

    deidentified_df = create_deidentified_dataset(multisite_df)
    print("Sample de-identified data:")
    print(deidentified_df.head())

    return longitudinal_df, multisite_df, deidentified_df

# Run the demonstration
longitudinal_data, multisite_data, deidentified_data = handle_research_data_challenges()
```

Ethical Considerations in Data Analysis

**Reference:**
```python
class EthicalDataAnalysis:
    """
    Framework for ensuring ethical considerations in data analysis
    """

    def __init__(self):
        self.ethical_checklist = {
            'consent_and_privacy': [
                'Data collection had appropriate consent',
                'Personal identifiers are properly protected',
                'Data sharing agreements are followed',
                'Participant privacy is maintained'
            ],
            'bias_and_fairness': [
                'Analysis methods don\'t discriminate against protected groups',
                'Sample is representative of target population',
                'Historical biases in data are acknowledged',
                'Results are interpreted fairly across groups'
            ],
            'transparency_and_reproducibility': [
                'Analysis methods are fully documented',
                'Code and data are available for verification',
                'Limitations and assumptions are clearly stated',
                'Conflicts of interest are disclosed'
            ],
            'beneficence_and_non_maleficence': [
                'Research aims to benefit participants/society',
                'Potential harms are minimized',
                'Results won\'t be used to harm individuals',
                'Scientific integrity is maintained'
            ]
        }

    def conduct_ethical_review(self, analysis_plan):
        """
        Review analysis plan for ethical considerations
        """
        print("=== ETHICAL ANALYSIS REVIEW ===")

        review_results = {}

        for category, criteria in self.ethical_checklist.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            category_results = []

            for criterion in criteria:
                # In practice, this would involve actual review
                # Here we demonstrate the process
                print(f"  □ {criterion}")
                print(f"    Status: [Requires reviewer assessment]")
                print(f"    Notes: [Document compliance or concerns]")
                category_results.append({
                    'criterion': criterion,
                    'status': 'pending_review',
                    'notes': ''
                })

            review_results[category] = category_results

        return review_results

    def create_data_protection_protocol(self, data_types):
        """
        Create data protection protocol based on data types
        """
        protocol = {
            'data_classification': {},
            'access_controls': {},
            'storage_requirements': {},
            'sharing_restrictions': {}
        }

        for data_type in data_types:
            if 'personal' in data_type.lower() or 'identifiable' in data_type.lower():
                protocol['data_classification'][data_type] = 'Highly Sensitive'
                protocol['access_controls'][data_type] = 'Restricted - Named individuals only'
                protocol['storage_requirements'][data_type] = 'Encrypted storage required'
                protocol['sharing_restrictions'][data_type] = 'IRB approval required'

            elif 'health' in data_type.lower() or 'medical' in data_type.lower():
                protocol['data_classification'][data_type] = 'Sensitive'
                protocol['access_controls'][data_type] = 'Role-based access'
                protocol['storage_requirements'][data_type] = 'Secure storage required'
                protocol['sharing_restrictions'][data_type] = 'Data use agreement required'

            else:
                protocol['data_classification'][data_type] = 'Standard'
                protocol['access_controls'][data_type] = 'Project team access'
                protocol['storage_requirements'][data_type] = 'Standard security'
                protocol['sharing_restrictions'][data_type] = 'Follow institutional policy'

        return protocol

# Example usage
ethical_framework = EthicalDataAnalysis()

# Review an analysis plan
analysis_plan = {
    'study_type': 'observational',
    'data_sources': ['survey responses', 'administrative records'],
    'population': 'adult patients',
    'outcomes': ['treatment effectiveness', 'cost analysis']
}

review_results = ethical_framework.conduct_ethical_review(analysis_plan)

# Create data protection protocol
data_types = ['Personal Health Information', 'Survey Responses', 'Administrative Data']
protection_protocol = ethical_framework.create_data_protection_protocol(data_types)

print("\n=== DATA PROTECTION PROTOCOL ===")
for category, details in protection_protocol.items():
    print(f"\n{category.upper().replace('_', ' ')}:")
    for data_type, requirement in details.items():
        print(f"  {data_type}: {requirement}")
```

Key Applications

This bonus content addresses:

1. **Longitudinal Studies** - Handle dropout patterns and missing data over time
2. **Multi-Site Research** - Harmonize data from different collection sites
3. **Sensitive Data** - De-identification and privacy protection techniques
4. **Ethical Review** - Systematic evaluation of analysis plans
5. **Data Classification** - Security protocols based on data sensitivity

These techniques are essential for clinical research, healthcare analytics, and any research involving sensitive or protected information.