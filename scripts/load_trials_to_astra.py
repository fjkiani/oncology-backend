# backend/scripts/load_trials_to_astra.py

import os
import pprint
import json
import time
from typing import List, Optional, Dict
import re

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp, JsonConfig
from sentence_transformers import SentenceTransformer

# --- Configuration --- 
load_dotenv() # Load environment variables from .env file

# Astra DB Credentials (Replace with your actual values or load from env)
ASTRA_DB_SECURE_BUNDLE_PATH = os.getenv('ASTRA_DB_SECURE_BUNDLE_PATH', 'path/to/your/secure-connect.zip')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN', 'AstraCS:...') 
ASTRA_DB_KEYSPACE_NAME = os.getenv('ASTRA_DB_KEYSPACE_NAME', 'default_keyspace') # Default to default_keyspace if not set
# Use 'clinical_trials' as the table name, matching user setup
ASTRA_DB_TABLE_NAME = os.getenv('ASTRA_DB_TABLE_NAME', 'clinical_trials') 

# Firecrawl API Key
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY', 'YOUR_FIRECRAWL_API_KEY')

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384 # Dimension of the 'all-MiniLM-L6-v2' model

# NCI Web Page URL Template
NCI_PAGE_URL_TEMPLATE = "https://www.cancer.gov/research/participate/clinical-trials-search/v?id={nci_id}"

# --- Pydantic Schema for Firecrawl Extraction (No longer used by Firecrawl call, but useful reference) --- 
class ContactDetail(BaseModel):
    location_name: Optional[str] = Field(description="Name of the hospital or location", default=None)
    city: Optional[str] = Field(description="City of the location", default=None)
    state: Optional[str] = Field(description="State of the location", default=None)
    status: Optional[str] = Field(description="Recruitment status at this location", default=None)
    contact_name: Optional[str] = Field(description="Name of the contact person", default=None)
    contact_phone: Optional[str] = Field(description="Phone number of the contact", default=None)
    contact_email: Optional[str] = Field(description="Email address of the contact", default=None)

class ClinicalTrialExtractionSchema(BaseModel):
    # ... (as before, now just a reference for desired fields)
    pass

# --- Database Functions --- 
def connect_astra_db():
    """Connects to Astra DB using credentials from environment variables or defaults."""
    if not os.path.exists(ASTRA_DB_SECURE_BUNDLE_PATH):
        print(f"ERROR: Astra DB secure connect bundle not found at: {ASTRA_DB_SECURE_BUNDLE_PATH}")
        return None, None
    if not ASTRA_DB_APPLICATION_TOKEN or 'AstraCS:' not in ASTRA_DB_APPLICATION_TOKEN:
        print(f"ERROR: Invalid or missing Astra DB application token (ASTRA_DB_APPLICATION_TOKEN) in environment.")
        return None, None
        
    cloud_config= {
            'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
    }
    
    # --- Corrected Authentication for Application Tokens ---
    # Use "token" as username and the full Application Token string as password
    auth_provider = PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN) 
    # --- End Correction ---
    
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    try:
        session = cluster.connect(ASTRA_DB_KEYSPACE_NAME)
        print(f"Connected to Astra DB keyspace: {ASTRA_DB_KEYSPACE_NAME}")
        return cluster, session
    except Exception as e:
        print(f"ERROR connecting to Astra DB: {e}")
        # Provide more details from the exception if available
        if hasattr(e, 'errors'):
             print("Specific errors:", e.errors)
        return None, None

def create_schema_astra_db(session):
    """Creates the table and vector index if they don't exist."""
    try:
        # Create Table
        create_table_cql = f"""
        CREATE TABLE IF NOT EXISTS {ASTRA_DB_KEYSPACE_NAME}.{ASTRA_DB_TABLE_NAME} (
            nci_id TEXT PRIMARY KEY,
            nct_id TEXT,
            title TEXT,
            status TEXT,
            phase TEXT,
            lead_org TEXT,
            raw_description_text TEXT,
            raw_inclusion_criteria_text TEXT,
            raw_exclusion_criteria_text TEXT,
            raw_objectives_text TEXT,
            structured_contacts_json TEXT, 
            raw_contacts_text TEXT,
            source_url TEXT,
            eligibility_vector vector<float, {EMBEDDING_DIMENSION}> 
        )
        """
        session.execute(create_table_cql)
        print(f"Table '{ASTRA_DB_TABLE_NAME}' created or already exists.")
        
        # Create Vector Index (Important for ANN search)
        create_index_cql = f"""
        CREATE CUSTOM INDEX IF NOT EXISTS eligibility_vector_idx 
        ON {ASTRA_DB_KEYSPACE_NAME}.{ASTRA_DB_TABLE_NAME} (eligibility_vector) 
        USING 'StorageAttachedIndex'
        """
        session.execute(create_index_cql)
        print(f"Index 'eligibility_vector_idx' created or already exists.")
        
    except Exception as e:
        print(f"ERROR creating schema in Astra DB: {e}")
        raise # Re-raise the exception to stop the script if schema fails

# --- Manual Parsing Helper Function (Implemented) ---
def parse_markdown_content(markdown_content: str, nci_id: str) -> Dict:
    """ 
    Parses the raw markdown scraped from the NCI page to extract structured fields using regex.
    Args: markdown_content, nci_id
    Returns: Dictionary with extracted fields.
    """
    print(f"Parsing markdown for {nci_id} (Length: {len(markdown_content)} chars)")
    extracted_data = {
        "nci_id": nci_id,
        "nct_id": None,
        "title": None,
        "status": None,
        "phase": None,
        "lead_org": None,
        "raw_description_text": None,
        "raw_inclusion_criteria_text": None,
        "raw_exclusion_criteria_text": None,
        "raw_objectives_text": None,
        "raw_contacts_text": None, # Keep raw contacts text
    }

    # --- REGEX PARSING LOGIC --- 
    try:
        # Title (First H1 heading)
        title_match = re.search(r"^#\s*(.*)", markdown_content, re.MULTILINE)
        if title_match: extracted_data["title"] = title_match.group(1).strip()

        # Status (Line starting with "Trial Status:")
        status_match = re.search(r"^Trial Status:\s*(\w+)", markdown_content, re.MULTILINE | re.IGNORECASE)
        if status_match: extracted_data["status"] = status_match.group(1).strip()

        # --- Function to extract text between two headings --- 
        def get_text_between(start_heading, end_heading, text):
            # Use DOTALL to make '.' match newlines
            # Use non-greedy matching .*?
            pattern = re.escape(start_heading) + r"(.*?)" + re.escape(end_heading)
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            # Handle case where start_heading is the last section
            match_start_only = re.search(re.escape(start_heading) + r"(.*)", text, re.IGNORECASE | re.DOTALL)
            if match_start_only:
                 return match_start_only.group(1).strip()
            return None
            
        # Description
        extracted_data["raw_description_text"] = get_text_between(
            "## Description", 
            "## Eligibility Criteria", 
            markdown_content
        )
        
        # Objectives
        extracted_data["raw_objectives_text"] = get_text_between(
            "## Trial Objectives and Outline", 
            "## Trial Phase & Type", 
            markdown_content
        )

        # Raw Contacts Text
        extracted_data["raw_contacts_text"] = get_text_between(
            "## Locations & Contacts",
            "## Trial Objectives and Outline", # Assuming Objectives follows Contacts
            markdown_content
        )
        if not extracted_data["raw_contacts_text"]:
             # Fallback if objectives doesn't follow contacts
              extracted_data["raw_contacts_text"] = get_text_between(
                "## Locations & Contacts",
                "##", # Try to find next heading generically
                 markdown_content
             )

        # Eligibility Criteria Section (then split inclusion/exclusion)
        eligibility_section_text = get_text_between(
            "## Eligibility Criteria", 
            "## Locations & Contacts", 
            markdown_content
        )
        
        if eligibility_section_text:
            # Inclusion Criteria (from start or '### Inclusion' up to '### Exclusion')
            inc_pattern = r"(?:###\s*Inclusion Criteria)(.*?)(?:###\s*Exclusion Criteria|\Z)" 
            inc_match = re.search(inc_pattern, eligibility_section_text, re.IGNORECASE | re.DOTALL)
            if inc_match:
                extracted_data["raw_inclusion_criteria_text"] = inc_match.group(1).strip()
            else: 
                 # If no explicit subheadings, assume all is inclusion initially
                 extracted_data["raw_inclusion_criteria_text"] = eligibility_section_text.strip()
            
            # Exclusion Criteria (from '### Exclusion' to end of section)
            exc_pattern = r"###\s*Exclusion Criteria(.*)"
            exc_match = re.search(exc_pattern, eligibility_section_text, re.IGNORECASE | re.DOTALL)
            if exc_match:
                extracted_data["raw_exclusion_criteria_text"] = exc_match.group(1).strip()
                # If exclusion is found, refine inclusion if it wasn't explicitly bounded
                if inc_match is None and extracted_data["raw_inclusion_criteria_text"] == eligibility_section_text.strip():
                     extracted_data["raw_inclusion_criteria_text"] = eligibility_section_text[:exc_match.start()].strip()
        
        # Phase (Look within specific section or standalone line)
        phase_match = re.search(r"\*\*Trial Phase\*\*\s*(.*)", markdown_content, re.IGNORECASE | re.MULTILINE)
        if phase_match: extracted_data["phase"] = phase_match.group(1).strip()
        
        # Lead Org (Look within specific section or standalone line)
        lead_org_match = re.search(r"\*\*Lead Organization\*\*\s*(.*)", markdown_content, re.IGNORECASE | re.MULTILINE)
        if lead_org_match: extracted_data["lead_org"] = lead_org_match.group(1).strip()

        # NCT ID (Look for pattern [NCT...] or ClinicalTrials.gov ID** [NCT...])
        nct_match = re.search(r"(?:ClinicalTrials\.gov ID\*\*\s*\[|\[)(NCT\d+)\]", markdown_content, re.IGNORECASE)
        if nct_match: extracted_data["nct_id"] = nct_match.group(1).strip()
        
    except Exception as e:
        print(f"Error during regex markdown parsing for {nci_id}: {e}")

    # --- END OF PARSING LOGIC --- 
    print(f"Finished parsing markdown for {nci_id}. Found-> Title: {bool(extracted_data.get('title'))}, Status: {bool(extracted_data.get('status'))}, Phase: {bool(extracted_data.get('phase'))}, NCT ID: {bool(extracted_data.get('nct_id'))}, Desc: {bool(extracted_data.get('raw_description_text'))}, Incl: {bool(extracted_data.get('raw_inclusion_criteria_text'))}, Excl: {bool(extracted_data.get('raw_exclusion_criteria_text'))}, Obj: {bool(extracted_data.get('raw_objectives_text'))}, Contacts: {bool(extracted_data.get('raw_contacts_text'))}")
    return extracted_data

# --- Processing Function --- 
def process_and_load_trial(nci_id: str, firecrawl_app: FirecrawlApp, session, embedding_model):
    """Scrapes (Markdown), parses, embeds, and loads data for a single trial ID."""
    print(f"\n--- Processing Trial: {nci_id} ---")
    trial_url = NCI_PAGE_URL_TEMPLATE.format(nci_id=nci_id)
    
    # 1. Scrape Markdown using Firecrawl
    print(f"Scraping URL for Markdown: {trial_url}")
    markdown_content = None
    try:
        scrape_result = firecrawl_app.scrape_url(trial_url, formats=["markdown"])
        if scrape_result and scrape_result.success and scrape_result.data and scrape_result.data.markdown:
            markdown_content = scrape_result.data.markdown
            print(f"Successfully scraped markdown for {nci_id} (Length: {len(markdown_content)}).")
        else:
            print(f"ERROR: Firecrawl markdown scrape failed or returned no data for {nci_id}.")
            pprint.pprint(scrape_result)
            return False
    except Exception as e:
        print(f"ERROR during Firecrawl markdown scrape for {nci_id}: {e}")
        return False

    # 2. Parse Markdown Content
    parsed_data = parse_markdown_content(markdown_content, nci_id)
    if not parsed_data:
        print(f"ERROR: Failed to parse markdown for {nci_id}. Skipping.")
        return False

    # 3. Insert/Update Metadata and Raw Text (using parsed data)
    print(f"Inserting/Updating base data for {nci_id}...")
    try:
        # Store raw contacts text if parsed
        contacts_text = parsed_data.get('raw_contacts_text') 
             
        insert_cql = f""" 
        INSERT INTO {ASTRA_DB_TABLE_NAME} (
            nci_id, nct_id, title, status, phase, lead_org, 
            raw_description_text, raw_inclusion_criteria_text, raw_exclusion_criteria_text, 
            raw_objectives_text, raw_contacts_text, source_url 
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) 
        """
        session.execute(insert_cql, (
            parsed_data.get('nci_id') or nci_id,
            parsed_data.get('nct_id'),
            parsed_data.get('title'),
            parsed_data.get('status'),
            parsed_data.get('phase'),
            parsed_data.get('lead_org'),
            parsed_data.get('raw_description_text'),
            parsed_data.get('raw_inclusion_criteria_text'),
            parsed_data.get('raw_exclusion_criteria_text'),
            parsed_data.get('raw_objectives_text'),
            contacts_text, 
            trial_url
        ))
        print(f"Base data inserted for {nci_id}.")
    except Exception as e:
        print(f"ERROR inserting base data for {nci_id}: {e}")
        return False

    # 4. Generate and Store Eligibility Vector (MVP: Single vector for combined text)
    print(f"Generating eligibility vector for {nci_id}...")
    try:
        inc_text = parsed_data.get('raw_inclusion_criteria_text', '') or ""
        exc_text = parsed_data.get('raw_exclusion_criteria_text', '') or ""
        full_eligibility_text = f"Inclusion Criteria:\n{inc_text}\n\nExclusion Criteria:\n{exc_text}"

        if not full_eligibility_text.strip() or full_eligibility_text == "Inclusion Criteria:\n\n\nExclusion Criteria:\n":
            print(f"Warning: No eligibility text parsed for {nci_id}. Skipping vector update.")
            return True 

        embedding = embedding_model.encode([full_eligibility_text])[0]
        update_cql = f"UPDATE {ASTRA_DB_TABLE_NAME} SET eligibility_vector = ? WHERE nci_id = ?"
        session.execute(update_cql, (embedding.tolist(), parsed_data.get('nci_id') or nci_id))
        print(f"Eligibility vector stored for {nci_id}.")
        return True
        
    except Exception as e:
        print(f"ERROR generating/storing vector for {nci_id}: {e}")
        return False

# --- Main Execution Block --- 
if __name__ == "__main__":
    # --- Setup --- 
    print("Initializing clients...")
    if FIRECRAWL_API_KEY == 'YOUR_FIRECRAWL_API_KEY':
        print("ERROR: Please set your FIRECRAWL_API_KEY in .env or script.")
        exit()
        
    try: 
        firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    except Exception as e:
        print(f"ERROR initializing FirecrawlApp: {e}")
        exit()
        
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"ERROR loading embedding model: {e}")
        exit()
        
    cluster, session = connect_astra_db()
    if not session:
        print("Exiting due to Astra DB connection failure.")
        exit()
        
    # Ensure schema exists
    try:
        create_schema_astra_db(session)
    except Exception as e:
         print("Exiting due to schema creation failure.")
         cluster.shutdown()
         exit()

    # --- Define Trial IDs to Process --- 
    nci_ids_to_process = [
        "NCI-2017-01579", 
        "NCI-2021-08397"
    ]
    print(f"\nPlanning to process {len(nci_ids_to_process)} trial IDs: {nci_ids_to_process}")

    # --- Process Trials --- 
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for nci_id in nci_ids_to_process:
        if process_and_load_trial(nci_id, firecrawl_app, session, embedding_model):
            success_count += 1
        else:
            fail_count += 1
        time.sleep(1) 

    end_time = time.time()
    print(f"\n--- Processing Complete --- ")
    print(f"Successfully processed: {success_count}")
    print(f"Failed/Skipped: {fail_count}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    if cluster:
        print("Shutting down Astra DB connection.")
        cluster.shutdown() 