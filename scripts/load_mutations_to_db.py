#!/usr/bin/env python3

import sqlite3
import json
import os
import sys

# --- Get Project Root --- 
# Assuming this script is in backend/scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Configuration --- 
INPUT_JSON_PATH = os.path.join(BACKEND_DIR, "data", "merged_mutation_clinical.json")
DB_PATH = os.path.join(BACKEND_DIR, "data", "patient_data.db")
TABLE_NAME = "mutations"

# Define the columns based on the desired schema and merged JSON keys
# Use lowercase and underscores for column names (standard SQL practice)
# Map JSON keys to SQL columns
COLUMN_MAPPING = {
    "uniqueSampleKey": "unique_sample_key",
    "uniquePatientKey": "unique_patient_key",
    "molecularProfileId": "molecular_profile_id",
    "sampleId": "sample_id",
    "patientId": "patient_id",
    "hugoGeneSymbol": "hugo_gene_symbol",
    "variantType": "variant_type",
    "chromosome": "chromosome",
    "startPosition": "start_position",
    "endPosition": "end_position",
    "proteinChange": "protein_change",
    "mutationStatus": "mutation_status",
    "validationStatus": "validation_status"
}

# SQL types for the columns
COLUMN_TYPES = {
    "unique_sample_key": "TEXT PRIMARY KEY NOT NULL",
    "unique_patient_key": "TEXT",
    "molecular_profile_id": "TEXT",
    "sample_id": "TEXT",
    "patient_id": "TEXT NOT NULL",
    "hugo_gene_symbol": "TEXT NOT NULL",
    "variant_type": "TEXT",
    "chromosome": "TEXT",
    "start_position": "INTEGER",
    "end_position": "INTEGER",
    "protein_change": "TEXT",
    "mutation_status": "TEXT",
    "validation_status": "TEXT"
}

INDEXED_COLUMNS = ["patient_id", "hugo_gene_symbol"]
# Ensure top-level keys for validation. `hugoGeneSymbol` is nested under `gene`.
REQUIRED_JSON_FIELDS = ["patientId", "gene", "uniqueSampleKey"]

def create_database_schema(conn):
    """Creates the mutations table and necessary indexes."""
    cursor = conn.cursor()
    
    # Build the CREATE TABLE statement dynamically
    column_defs = [f'"{col_name}" {col_type}' for col_name, col_type in COLUMN_TYPES.items()]
    create_table_sql = f"CREATE TABLE IF NOT EXISTS \"{TABLE_NAME}\" ({ ', '.join(column_defs) });"
    
    print(f"Executing: {create_table_sql}")
    cursor.execute(create_table_sql)
    
    # Create indexes
    for col in INDEXED_COLUMNS:
        index_name = f"idx_{TABLE_NAME}_{col}"
        create_index_sql = f"CREATE INDEX IF NOT EXISTS \"{index_name}\" ON \"{TABLE_NAME}\" (\"{col}\");"
        print(f"Executing: {create_index_sql}")
        cursor.execute(create_index_sql)
    
    conn.commit()
    print(f"Table '{TABLE_NAME}' and indexes ensured.")

def process_record_for_batch(record, column_mapping, sql_columns, required_json_fields):
    """Processes a single JSON record into a tuple of values for SQL insertion, adding validation."""
    # Validate required JSON fields
    for field in required_json_fields:
        if field == "gene": # Special check for the nested hugoGeneSymbol
            if not record.get("gene") or record.get("gene", {}).get("hugoGeneSymbol") is None:
                # print(f"Skipping record missing required field 'gene.hugoGeneSymbol': Patient ID {record.get('patientId', 'N/A')}")
                return None
        elif record.get(field) is None: 
            # print(f"Skipping record missing required field '{field}': Patient ID {record.get('patientId', 'N/A')}")
            return None # Signal to skip this record
            
    values_dict = {}
    for json_key, sql_col in column_mapping.items():
        if json_key == "hugoGeneSymbol": # Special handling for nested key
            value = record.get("gene", {}).get("hugoGeneSymbol")
        else:
            value = record.get(json_key)
        values_dict[sql_col] = value
    
    # Ensure order matches sql_columns for the tuple
    ordered_values = [values_dict.get(col) for col in sql_columns]
    return tuple(ordered_values)

def load_data_to_db(conn, data):
    """Loads mutation data into the specified table using executemany for performance."""
    sql_columns = list(COLUMN_TYPES.keys())
    sql_placeholders = ", ".join(["?"] * len(sql_columns))
    formatted_sql_columns = ", ".join([f'"{c}"' for c in sql_columns])
    insert_sql = f"INSERT OR IGNORE INTO \"{TABLE_NAME}\" ({formatted_sql_columns}) VALUES ({sql_placeholders});"
    
    batch_size = 1000
    processed_records_for_batch = []
    
    total_json_records = len(data)
    print(f"Preparing to insert {total_json_records} records in batches of {batch_size}...")
    
    records_processed_count = 0
    newly_inserted_count = 0
    skipped_due_to_missing_fields = 0
    skipped_due_to_duplicate_pk = 0

    for i, record in enumerate(data):
        processed_values = process_record_for_batch(record, COLUMN_MAPPING, sql_columns, REQUIRED_JSON_FIELDS)
        if processed_values:
            processed_records_for_batch.append(processed_values)
        else:
            skipped_due_to_missing_fields += 1
            
        if len(processed_records_for_batch) == batch_size or (i == total_json_records - 1 and processed_records_for_batch):
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.executemany(insert_sql, processed_records_for_batch)
                    if cursor.rowcount != -1:
                        newly_inserted_count += cursor.rowcount
                        skipped_due_to_duplicate_pk += (len(processed_records_for_batch) - cursor.rowcount)
                    else:
                        skipped_due_to_duplicate_pk += len(processed_records_for_batch)

                    records_processed_count += len(processed_records_for_batch)
                    print(f"  Processed batch ending at record {i+1}/{total_json_records}. Cumulative new inserts: {newly_inserted_count}")
            except sqlite3.Error as e:
                print(f"Error during batch insert (records {i-len(processed_records_for_batch)+1} to {i+1}): {e}")
            finally:
                processed_records_for_batch = []
            
    print("\nInsertion attempt complete.")
    print(f"  Total JSON records processed: {total_json_records}")
    print(f"  Records processed for DB insertion: {records_processed_count}")
    print(f"  Newly inserted records: {newly_inserted_count}")
    print(f"  Skipped due to missing required fields: {skipped_due_to_missing_fields}")
    print(f"  Skipped (likely due to existing PK or other IGNORE condition): {skipped_due_to_duplicate_pk}") 

def main():
    print(f"Database path: {DB_PATH}")
    print(f"Input JSON path: {INPUT_JSON_PATH}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: Input JSON file not found at {INPUT_JSON_PATH}")
        sys.exit(1)
    
    # Connect to SQLite database
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        print("Database connection established.")
        
        # Create schema
        create_database_schema(conn)
        
        # Load JSON data
        print(f"Loading JSON data from {INPUT_JSON_PATH}...")
        with open(INPUT_JSON_PATH, 'r') as f:
            mutation_data = json.load(f)
        print(f"Loaded {len(mutation_data)} records from JSON.")
        
        # JSON Schema Sanity Check
        if not mutation_data:
            print("Error: Mutation data is empty. Nothing to load.")
            sys.exit(1)
        
        # Check keys of the first record against a subset of expected keys
        # For the sanity check, we verify top-level keys. `hugoGeneSymbol` is nested.
        expected_keys_subset = {"patientId", "uniqueSampleKey", "proteinChange", "gene"}
        first_record_keys = set(mutation_data[0].keys())
        if not expected_keys_subset.issubset(first_record_keys):
            print("Error: JSON schema mismatch. First record is missing expected keys.")
            print(f"  Expected (subset): {expected_keys_subset}")
            print(f"  Found in first record: {first_record_keys}")
            missing = expected_keys_subset - first_record_keys
            if missing:
                print(f"  Missing critical keys: {missing}")
            # Additionally, check for the nested hugoGeneSymbol in the first record for a more complete sanity check
            if "gene" in first_record_keys and not mutation_data[0].get("gene", {}).get("hugoGeneSymbol"):
                print("  Additionally, 'hugoGeneSymbol' is missing within the 'gene' object of the first record.")
            sys.exit(1)
        # Check the nested hugoGeneSymbol specifically if 'gene' is present
        if "gene" in first_record_keys and not mutation_data[0].get("gene", {}).get("hugoGeneSymbol"):
             print("Error: JSON schema sanity check failed. 'hugoGeneSymbol' is missing within the 'gene' object of the first record.")
             sys.exit(1)

        print("JSON schema sanity check passed.")

        # Load data into DB
        load_data_to_db(conn, mutation_data)
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main() 