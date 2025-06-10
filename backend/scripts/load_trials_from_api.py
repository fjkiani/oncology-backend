# backend/scripts/load_trials_from_api.py
import logging
import time
import os
import sys
from pathlib import Path

# Add project root to Python path for module imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backend.utils.database_connections import DatabaseConnections
# We will need the API fetching logic from the research script.
# This can be refactored and moved to a shared utility later.
from backend.research.clinicaltrials_utils import search_clinical_trials, parse_study

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def wipe_databases(db_manager: DatabaseConnections):
    """
    Deletes all existing data from the clinical trials tables to ensure a fresh start.
    TASK for Data Loader (Friend 2).
    """
    logger.info("Starting database wipe process...")
    # TODO: Implement logic to clear the SQLite 'trials' table.
    # TODO: Implement logic to clear/drop the AstraDB collection.
    logger.info("Database wipe process complete.")
    pass

def fetch_and_load_data(db_manager: DatabaseConnections):
    """
    Fetches trial data from the API page by page and loads it into the databases.
    This will orchestrate the work of both the Extractor and the Loader.
    """
    logger.info("Starting data fetch and load process...")
    
    # --- This section is for the Data Extractor (Friend 1) ---
    # TODO: Use the enhanced `search_clinical_trials` function to create a robust
    #       iterator or generator that fetches all pages from the API.
    #       The search criteria should be broad enough to get all trials.
    #       e.g., criteria = {'query.cond': 'cancer'} or similar.
    
    # --- This section is for the Data Loader (Friend 2) ---
    # Loop through each trial from the API response:
    # for trial in all_trials_from_api:
    #   1. Parse the trial data (using `parse_study` or an enhanced version).
    #   2. Load metadata into SQLite.
    #   3. Create embedding for eligibility criteria.
    #   4. Load vector and metadata into AstraDB.
    #   5. Log progress every N trials.
    
    logger.info("Data fetch and load process complete.")
    pass

def main():
    """Main function to orchestrate the ETL pipeline."""
    logger.info("--- Clinical Trials ETL Pipeline Started ---")
    start_time = time.time()
    
    # Use the context manager for database connections
    with DatabaseConnections() as db_manager:
        # Step 1: Wipe existing data for a full refresh
        wipe_databases(db_manager)
        
        # Step 2: Fetch new data and load it
        fetch_and_load_data(db_manager)
        
    end_time = time.time()
    logger.info(f"--- Clinical Trials ETL Pipeline Finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main() 