# backend/scripts/load_trials_to_astra.py

import os
import pprint
import json
import time
from typing import List, Optional, Dict
import re

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp, JsonConfig
from sentence_transformers import SentenceTransformer
from astrapy.db import AstraDB
from astrapy.constants import VectorMetric

# --- Configuration --- 
load_dotenv() # Load environment variables from .env file

# Astra DB Credentials
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE_NAME", "default_keyspace")
COLLECTION_NAME = "clinical_trials"

# Firecrawl API Key
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY', 'YOUR_FIRECRAWL_API_KEY')

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384 # Dimension of the 'all-MiniLM-L6-v2' model

# NCI Web Page URL Template
NCI_PAGE_URL_TEMPLATE = "https://www.cancer.gov/research/participate/clinical-trials-search/v?id={nci_id}"

def connect_astra_db():
    """Connects to Astra DB using environment variables."""
    print(f"Connecting to Astra DB endpoint {ASTRA_DB_API_ENDPOINT}...")
    if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        print("Missing Astra DB credentials in environment variables")
        return None
    try:
        astra_db = AstraDB(
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE
        )
        print("Successfully connected to Astra DB.")
        return astra_db
    except Exception as e:
        print(f"Failed to connect to Astra DB: {e}")
        return None

def create_collection(astra_db):
    """Creates the collection if it doesn't exist."""
    try:
        # Check if collection exists
        collection_names = astra_db.list_collection_names()
        if COLLECTION_NAME in collection_names:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
            return astra_db.collection(COLLECTION_NAME)
        
        # Create collection with vector search enabled
        collection = astra_db.create_collection(
            COLLECTION_NAME,
            definition={
                "vector": {
                    "dimension": EMBEDDING_DIMENSION,
                    "metric": VectorMetric.COSINE
                }
            }
        )
        print(f"Created collection '{COLLECTION_NAME}' with vector search enabled.")
        return collection
    except Exception as e:
        print(f"ERROR creating collection: {e}")
        raise

def process_and_load_trial(nci_id: str, firecrawl_app: FirecrawlApp, collection, embedding_model):
    """Process a single trial and load it into AstraDB."""
    trial_url = NCI_PAGE_URL_TEMPLATE.format(nci_id=nci_id)
    print(f"Processing trial {nci_id} from {trial_url}")
    
    try:
        # 1. Fetch and parse trial data
        parsed_data = firecrawl_app.crawl_url(trial_url)
        if not parsed_data:
            print(f"No data found for {nci_id}")
            return False
            
        # 2. Prepare document for AstraDB
        document = {
            "id": nci_id,  # Use nci_id as the document ID
            "nct_id": parsed_data.get('nct_id'),
            "title": parsed_data.get('title'),
            "status": parsed_data.get('status'),
            "phase": parsed_data.get('phase'),
            "lead_org": parsed_data.get('lead_org'),
            "description": parsed_data.get('raw_description_text'),
            "inclusion_criteria": parsed_data.get('raw_inclusion_criteria_text'),
            "exclusion_criteria": parsed_data.get('raw_exclusion_criteria_text'),
            "objectives": parsed_data.get('raw_objectives_text'),
            "contacts": parsed_data.get('raw_contacts_text'),
            "source_url": trial_url
        }
        
        # 3. Generate eligibility vector
        eligibility_text = " ".join(filter(None, [
            parsed_data.get('raw_inclusion_criteria_text', ''),
            parsed_data.get('raw_exclusion_criteria_text', '')
        ]))
        
        if eligibility_text:
            vector = embedding_model.encode(eligibility_text).tolist()
            document["vector"] = vector
        
        # 4. Insert into AstraDB
        result = collection.upsert(
            id=nci_id,
            document=document
        )
        print(f"Successfully loaded trial {nci_id} into AstraDB")
        return True
        
    except Exception as e:
        print(f"ERROR processing trial {nci_id}: {e}")
        return False

def main():
    # Initialize connections and models
    astra_db = connect_astra_db()
    if not astra_db:
        print("Failed to connect to AstraDB. Exiting.")
        return
        
    collection = create_collection(astra_db)
    if not collection:
        print("Failed to create/access collection. Exiting.")
        return
        
    firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Process list of NCI IDs (you'll need to provide this list)
    nci_ids = ["NCI-2023-12345"]  # Example - replace with your list
    
    for nci_id in nci_ids:
        success = process_and_load_trial(nci_id, firecrawl_app, collection, embedding_model)
        if not success:
            print(f"Failed to process {nci_id}")
        time.sleep(1)  # Be nice to the API
        
if __name__ == "__main__":
    main() 