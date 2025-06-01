#!/usr/bin/env python3

import json
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, '..', '..')) # Go up two levels

# Define file paths relative to script location
mutation_file = os.path.join(script_dir, "brca_tcga_mutations.json")
clinical_file = os.path.join(script_dir, "brca_tcga_clinical_data.json")
output_file = os.path.join(script_dir, "merged_mutation_clinical.json")

print(f"Script directory: {script_dir}")
# print(f"Workspace root: {workspace_root}") # No longer needed for path construction

print(f"Loading mutation data from: {mutation_file}")
print(f"Loading clinical data from: {clinical_file}")

# --- Load Mutation Data ---
try:
    with open(mutation_file, 'r') as f:
        mutations = json.load(f)
except FileNotFoundError:
    print(f"Error: Mutation file not found at {mutation_file}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {mutation_file}: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading {mutation_file}: {e}")
    exit(1)

print(f"Loaded {len(mutations)} mutation records.")

# --- Load Clinical Data ---
try:
    with open(clinical_file, 'r') as f:
        clinical_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Clinical file not found at {clinical_file}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from {clinical_file}: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading {clinical_file}: {e}")
    exit(1)

print(f"Loaded {len(clinical_data)} clinical records.")

# --- Build Clinical Lookup ---
# Use 'Patient ID' as the key, as defined in the clinical JSON
clinical_lookup = {row['Patient ID']: row for row in clinical_data if 'Patient ID' in row}

if not clinical_lookup:
    print("Warning: Clinical lookup dictionary is empty. Check 'Patient ID' column in clinical data.")

print(f"Built clinical lookup with {len(clinical_lookup)} unique patient IDs.")

# --- Annotate Mutations ---
merged_count = 0
for mut in mutations:
    # Use 'patientId' as the key from the mutation data
    patient_id = mut.get("patientId")
    if patient_id:
        # Find clinical data, default to empty dict if not found
        patient_clinical_data = clinical_lookup.get(patient_id, {})
        mut["clinical_data"] = patient_clinical_data
        if patient_clinical_data: # Count if we actually found matching clinical data
            merged_count += 1
    else:
        mut["clinical_data"] = {} # Add empty dict if no patientId in mutation record

print(f"Annotated {merged_count} mutation records with corresponding clinical data.")

# --- Save Merged Data ---
print(f"Saving merged data to: {output_file}")
try:
    with open(output_file, 'w') as f:
        json.dump(mutations, f, indent=2)
except IOError as e:
    print(f"Error writing merged JSON file: {e}")
    exit(1)

print(f"Successfully merged data and saved to {output_file}") 