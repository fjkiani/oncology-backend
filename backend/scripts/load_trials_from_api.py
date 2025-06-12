# backend/scripts/load_trials_from_api.py
from pathlib import Path
import sys
import os
# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import logging
import time
import asyncio
from backend.utils.database_connections import DatabaseConnections
# We will need the API fetching logic from the research script.
# This can be refactored and moved to a shared utility later.
from backend.research.clinicaltrials_utils import fetch_all_trials_generator
from sentence_transformers import SentenceTransformer
import json
from astrapy.constants import VectorMetric
from astrapy.info import CollectionDefinition
import torch
import nltk
import argparse
from typing import Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# --- NLTK setup ---
# Download the 'punkt' tokenizer models if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt', quiet=True)
    logger.info("'punkt' model downloaded successfully.")
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("NLTK 'punkt_tab' model not found. Downloading...")
    nltk.download('punkt_tab', quiet=True)
    logger.info("'punkt_tab' model downloaded successfully.")

def wipe_databases(db_manager: DatabaseConnections):
    """
    Deletes all existing data from the clinical trials tables to ensure a fresh start.
    """
    logger.info("Starting database wipe process...")

    # Wipe SQLite data
    try:
        sqlite_conn = db_manager.get_sqlite_connection()
        if not sqlite_conn:
            raise ConnectionError("Failed to get SQLite connection.")
            
        logger.info("Dropping and recreating SQLite 'trials' table...")
        cursor = sqlite_conn.cursor()
        
        # Drop the table if it exists to ensure a clean slate
        cursor.execute("DROP TABLE IF EXISTS trials")
        
        # Create the table with a schema matching our parsed data
        cursor.execute("""
            CREATE TABLE trials (
                id TEXT PRIMARY KEY,
                title TEXT,
                status TEXT,
                phases TEXT,
                summary TEXT,
                conditions TEXT,
                interventions TEXT,
                source TEXT,
                inclusion_criteria TEXT,
                exclusion_criteria TEXT
            )
        """)
        
        sqlite_conn.commit()
        logger.info("SQLite 'trials' table created successfully.")
    except Exception as e:
        logger.error(f"Failed to create SQLite 'trials' table: {e}")
        raise

    # Wipe AstraDB data
    try:
        astra_db = db_manager.get_vector_db_connection()
        if not astra_db:
            raise ConnectionError("Failed to get AstraDB connection.")

        collection_name = "clinical_trials"
        # Correct way to check for a collection in astrapy v2
        collection_names = astra_db.list_collection_names()
        if collection_name in collection_names:
            logger.info(f"Deleting existing AstraDB collection '{collection_name}'...")
            astra_db.drop_collection(collection_name)
            logger.info(f"AstraDB collection '{collection_name}' deleted.")
        else:
            logger.info(f"AstraDB collection '{collection_name}' does not exist, no need to delete.")

        # Recreate the collection
        logger.info(f"Recreating AstraDB collection '{collection_name}'...")
        embedding_dim = 384 
        astra_db.create_collection(
            collection_name, 
            definition={
                "vector": {
                    "dimension": embedding_dim,
                    "metric": VectorMetric.COSINE
                }
            }
        )
        logger.info(f"AstraDB collection '{collection_name}' created successfully with dimension {embedding_dim}.")

    except Exception as e:
        logger.error(f"Failed to wipe and recreate AstraDB collection: {e}")
        raise

    logger.info("Database wipe process complete.")

# Global model instance
model = None

async def fetch_and_load_data(db_manager: DatabaseConnections, max_pages: Optional[int] = None, batch_size=100):
    """
    Fetches trial data from the API page by page and loads it into the databases.
    This will orchestrate the work of both the Extractor and the Loader.
    """
    logger.info("Starting data fetch and load process...")
    
    # --- Database and Model Setup ---
    sqlite_conn = db_manager.get_sqlite_connection()
    astra_db = db_manager.get_vector_db_connection()
    if not sqlite_conn or not astra_db:
        logger.error("Database connections could not be established. Aborting.")
        return

    trials_collection = astra_db.get_collection("clinical_trials")
    sql_cursor = sqlite_conn.cursor()
    
    global model
    if model is None:
        logger.info("Initializing sentence transformer model...")
        # Use a smaller, efficient model suitable for MPS
        model = SentenceTransformer('all-MiniLM-L6-v2', device='mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Sentence transformer model initialized on device: {model.device}")

    # --- Data Extractor and Loader Integration ---
    # To fetch ALL trials, we leave the search criteria empty.
    search_criteria = {}
    
    total_trials_processed = 0
    page_num = 0
    sql_batch = []
    vector_batch = []

    # Use the max_pages argument here
    async for page_of_trials in fetch_all_trials_generator(search_criteria, page_size=100, max_pages=max_pages):
        logger.info(f"Processing page with {len(page_of_trials)} trials...")

        for trial in page_of_trials:
            parsed_data = trial # The generator now yields parsed data
            
            # 1. Prepare data for SQLite (metadata)
            sql_data = parsed_data.copy()
            
            # Split eligibility criteria into inclusion and exclusion
            eligibility_text = sql_data.pop('eligibility_criteria', '')
            inclusion_criteria = ""
            exclusion_criteria = ""
            
            # Simple heuristic: Split on "Exclusion" or "Key Exclusion"
            if "Key Exclusion Criteria:" in eligibility_text:
                parts = eligibility_text.split("Key Exclusion Criteria:", 1)
                inclusion_criteria = parts[0].replace("Key Inclusion Criteria:", "").strip()
                exclusion_criteria = parts[1].strip()
            elif "Exclusion Criteria:" in eligibility_text:
                parts = eligibility_text.split("Exclusion Criteria:", 1)
                inclusion_criteria = parts[0].replace("Inclusion Criteria:", "").strip()
                exclusion_criteria = parts[1].strip()
            else:
                inclusion_criteria = eligibility_text

            # Convert lists to JSON strings for SQLite
            for key in ['phases', 'conditions', 'interventions']:
                if key in sql_data and isinstance(sql_data[key], list):
                    sql_data[key] = json.dumps(sql_data[key])
            
            # Add eligibility criteria
            sql_data['inclusion_criteria'] = inclusion_criteria
            sql_data['exclusion_criteria'] = exclusion_criteria
            
            sql_batch.append(sql_data)

            # 2. Prepare and chunk text for vector DB
            eligibility_text = parsed_data.get('eligibility_criteria', '')
            if not eligibility_text:
                continue # Skip if no eligibility criteria

            # Chunk the text to respect AstraDB's limits by splitting into sentences first
            sentences = nltk.sent_tokenize(eligibility_text)
            
            # Group sentences into chunks of a safe size
            text_chunks = []
            current_chunk = ""
            safe_limit = 4000 # Keep it well below the 8000 byte limit

            for sentence in sentences:
                # If a single sentence is already too long, split it
                if len(sentence) > safe_limit:
                    # If there's a current chunk, save it first
                    if current_chunk:
                        text_chunks.append(current_chunk.strip())
                        current_chunk = ""
                    # Split the long sentence and add its parts as separate chunks
                    for i in range(0, len(sentence), safe_limit):
                        text_chunks.append(sentence[i:i+safe_limit])
                    continue

                # If adding the next sentence exceeds the safe limit, finalize the current chunk
                if len(current_chunk) + len(sentence) + 1 > safe_limit:
                    text_chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
            
            # Add the last chunk if it's not empty
            if current_chunk:
                text_chunks.append(current_chunk.strip())

            for i, chunk in enumerate(text_chunks):
                if not chunk: continue # Skip empty chunks
                vector_batch.append({
                    "_id": f"{parsed_data['id']}_{i}", # Create a unique ID for each chunk
                    "text": chunk,
                    "nct_id": parsed_data['id'] # Keep a reference to the parent trial, using nct_id to match search
                })

        # Every BATCH_SIZE, execute a batch insert
        if len(sql_batch) >= batch_size:
            logger.info(f"Executing batch insert for {len(sql_batch)} SQL records and {len(vector_batch)} vector records.")
            # Convert list of dicts to list of tuples for executemany
            sql_batch_tuples = [tuple(d.values()) for d in sql_batch]
            _execute_batch_inserts(sql_cursor, sql_batch_tuples, trials_collection, vector_batch)
            
            # Reset batches
            sql_batch = []
            vector_batch = []

        total_trials_processed += len(page_of_trials)
        logger.info(f"Processed {total_trials_processed} trials so far.")

    # If there are any remaining items, execute one final batch insert
    if sql_batch or vector_batch:
        logger.info(f"Executing final batch insert for {len(sql_batch)} SQL records and {len(vector_batch)} vector records.")
        # Convert list of dicts to list of tuples for executemany
        sql_batch_tuples = [tuple(d.values()) for d in sql_batch]
        _execute_batch_inserts(sql_cursor, sql_batch_tuples, trials_collection, vector_batch)

    logger.info(f"Data fetch and load process complete. Total trials processed: {total_trials_processed}")

def _execute_batch_inserts(cursor, sql_batch: list, trials_collection, vector_batch: list):
    """
    Executes batch inserts for both SQLite and AstraDB.
    Now handles pre-chunked vector data.
    """
    if not sql_batch:
        return

    # --- SQLite Batch Insert ---
    try:
        placeholders = ", ".join(["?"] * len(sql_batch[0]))
        sql = f"INSERT INTO trials (id, title, status, phases, summary, conditions, interventions, source, inclusion_criteria, exclusion_criteria) VALUES ({placeholders})"
        cursor.executemany(sql, sql_batch)
        cursor.connection.commit()
        logger.info(f"Successfully inserted batch of {len(sql_batch)} records into SQLite.")
    except Exception as e:
        logger.error(f"Error batch inserting into SQLite: {e}")
        cursor.connection.rollback()
        raise

    # --- AstraDB Batch Insert ---
    if not vector_batch:
        return
        
    try:
        # Embed all text chunks in the batch
        texts_to_embed = [item['text'] for item in vector_batch]
        embeddings = model.encode(texts_to_embed, show_progress_bar=False).tolist()

        # Add the generated vector to each item in the batch
        documents_to_insert = []
        for i, item in enumerate(vector_batch):
            documents_to_insert.append({
                "_id": item["_id"],  # Fixed: access _id field as stored in vector_batch
                "$vector": embeddings[i],
                "text": item["text"],
                "nct_id": item["nct_id"]  # Changed from trial_id to nct_id
            })

        # Use insert_many for batch insertion
        trials_collection.insert_many(documents_to_insert)
        logger.info(f"Successfully inserted batch of {len(vector_batch)} vector documents into AstraDB.")
    except Exception as e:
        logger.error(f"Error batch inserting into AstraDB: {e}")
        # Note: We don't raise here to allow the pipeline to continue, but we log the error.
        # In a production system, you might send these failed batches to a dead-letter queue.

def main():
    """Main function to orchestrate the ETL pipeline."""
    parser = argparse.ArgumentParser(description="Clinical Trials ETL Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of pages to fetch from the API for testing.")
    args = parser.parse_args()

    logger.info("--- Clinical Trials ETL Pipeline Started ---")
    start_time = time.time()

    # Use the context manager for database connections
    with DatabaseConnections() as db_manager:
        # Step 1: Wipe existing data for a full refresh
        wipe_databases(db_manager)
        
        # Step 2: Fetch new data and load it
        # Pass the limit from command line args to the async function
        asyncio.run(fetch_and_load_data(db_manager, max_pages=args.limit))
        
    end_time = time.time()
    logger.info(f"--- Clinical Trials ETL Pipeline Finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main() 