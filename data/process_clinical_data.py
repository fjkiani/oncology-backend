#!/usr/bin/env python3

import pandas as pd
import json
import os

# Define input and output file paths relative to script location
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

input_tsv_path = os.path.join(script_dir, "brca_tcga_clinical_data.tsv")
output_json_path = os.path.join(script_dir, "brca_tcga_clinical_data.json")

print(f"Script location: {script_dir}")
print(f"Loading clinical data from: {input_tsv_path}")

# Check if input file exists
if not os.path.exists(input_tsv_path):
    print(f"Error: Input file not found at {input_tsv_path}")
    exit(1)

# Load your clinical data file (TSV)
try:
    # Use sep='\t' for TSV files
    df = pd.read_csv(input_tsv_path, sep='\t', engine='python')
except Exception as e:
    print(f"Error reading TSV file: {e}")
    exit(1)

print(f"Loaded {df.shape[0]} records and {df.shape[1]} columns.")

# Optionally, select only columns you care about
# Map desired concepts to actual column names found in the TSV
columns_of_interest = [
    'Patient ID',                               # Mapped from 'case_id'
    'Diagnosis Age',                            # Mapped from 'age_at_diagnosis'
    'Neoplasm Disease Stage American Joint Committee on Cancer Code', # Mapped from 'tumor_stage'
    'Overall Survival Status',                  # Mapped from 'vital_status' (often represented this way)
    'Overall Survival (Months)',                # Keeping this as potentially useful alongside status
    # 'days_to_death' - No direct match found, Overall Survival (Months/Status) might be used instead
    'Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value', # Mapped from 'days_to_last_follow_up'
    'Sex',                                      # Mapped from 'gender'
    'Race Category',                            # Mapped from 'race'
    'Ethnicity Category'                        # Mapped from 'ethnicity'
]

# Filter the DataFrame to keep only existing columns from the interest list
existing_columns = [col for col in columns_of_interest if col in df.columns]
if not existing_columns:
    print("Error: None of the specified columns_of_interest were found in the TSV file.")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

df_filtered = df[existing_columns]
print(f"Selected {len(existing_columns)} columns: {existing_columns}")

# Convert to list of dicts (JSON objects)
# Handle potential NaN values which are not valid JSON - convert them to None (null in JSON)
clinical_json = df_filtered.where(pd.notnull(df_filtered), None).to_dict(orient='records')

# Save as JSON
print(f"Saving JSON data to: {output_json_path}")
try:
    with open(output_json_path, 'w') as f:
        json.dump(clinical_json, f, indent=2)
except IOError as e:
    print(f"Error writing JSON file: {e}")
    exit(1)

print(f"Successfully converted {len(clinical_json)} records to JSON. Ready for LLM ingestion!") 