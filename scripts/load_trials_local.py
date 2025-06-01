import os
import json
import re
import pprint
import sqlite3
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import logging
from pathlib import Path
import google.generativeai as genai
import time

# --- Configuration ---
# Force DEBUG level logging to see detailed parsing output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Path to the root of the Git repository (oncology-backend)
# Assumes 'scripts/' directory is at the root of the 'oncology-backend' project.
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent 

SQLITE_DB_PATH = str(PROJECT_ROOT_DIR / "backend" / "data" / "clinical_trials.db")
# CHROMA_DB_PATH = "./chroma_db" # Old relative path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # oncology-coPilot-main
CHROMA_DB_PATH = str(PROJECT_ROOT_DIR / "backend" / "data" / "chroma_data")
CHROMA_COLLECTION_NAME = "clinical_trials_eligibility"
SOURCE_JSON_PATH = str(PROJECT_ROOT_DIR / "documents.json") # Path relative to project root
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # Removed: Specific to SentenceTransformer
# VECTOR_DIMENSION = 384 # Removed: Specific to SentenceTransformer

# --- LLM Configuration for Summarization & Embeddings ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = "gemini-1.5-flash" # For summarization
EMBEDDING_MODEL_GOOGLE = "models/embedding-001" # For ChromaDB embeddings

# --- Trial Summary Prompt Template (Copied from Agent) ---
TRIAL_SUMMARY_PROMPT_TEMPLATE = """
Summarize the key aspects of the following clinical trial based ONLY on the provided Description and Objectives.
Focus on: 
1. The primary condition being studied.
2. The main intervention(s) being tested.
3. The key target patient population (briefly).

Respond with a concise summary (2-4 sentences maximum). Do not include information not present in the text below.

**Trial Description:**
{description}

**Trial Objectives:**
{objectives}

**Summary:**
"""

# --- Helper Functions (Copied from previous script, adapted slightly) ---

def get_text_between(text, start_heading, end_heading=None):
    """Extracts text between two headings (e.g., ## Heading1 and ## Heading2)."""
    start_pattern = re.escape(start_heading)
    start_regex = r"(?m)^[ \t]*" + start_pattern + r"[ \t]*\n"
    start_match = re.search(start_regex, text, re.IGNORECASE)
    if not start_match:
        return None
    start_pos = start_match.end()
    end_pos = len(text)
    if end_heading:
        # Regex to find the next L2 or L3 heading more reliably
        end_regex = r"(?m)^[ \t]*#{2,3}[ \t]+"
        end_match = re.search(end_regex, text[start_pos:], re.IGNORECASE)
        if end_match:
            # Check if the found heading is the specific end_heading if provided
            # This logic might need refinement depending on markdown consistency
            found_heading = text[start_pos + end_match.start():].split('\n')[0].strip()
            # Simple check: if end_heading exists, try to match it approximately
            # This part is tricky, relying on the next heading marker is safer
            # if end_heading.lower() in found_heading.lower():
            end_pos = start_pos + end_match.start()
            # else: # Found a heading, but not the specific end one - still use it as boundary
            #     end_pos = start_pos + end_match.start()
    return text[start_pos:end_pos].strip()

def parse_markdown_content(markdown):
    """Parses the markdown text to extract key sections, with more robust eligibility parsing."""
    data = {}
    logging.debug("--- Parsing Markdown Start ---")

    # --- More Robust Eligibility Parsing --- 
    eligibility_block = get_text_between(markdown, "## Eligibility Criteria", "## Locations & Contacts") # Try to get the whole block first
    inc_text = None
    exc_text = None

    if eligibility_block:
        logging.debug("Found ## Eligibility Criteria block.")
        # Try finding specific subheadings within the block
        inc_start_match = re.search(r"(?m)^[ \t]*### Inclusion Criteria[ \t]*\n", eligibility_block, re.IGNORECASE)
        exc_start_match = re.search(r"(?m)^[ \t]*### Exclusion Criteria[ \t]*\n", eligibility_block, re.IGNORECASE)
        
        if inc_start_match and exc_start_match:
            # Found both specific headings
            logging.debug("Found both ### Inclusion and ### Exclusion Criteria headings.")
            inc_start_pos = inc_start_match.end()
            inc_end_pos = exc_start_match.start()
            inc_text = eligibility_block[inc_start_pos:inc_end_pos].strip()
            
            exc_start_pos = exc_start_match.end()
            # Exclusion text runs to the end of the eligibility_block
            exc_text = eligibility_block[exc_start_pos:].strip()
            
        elif inc_start_match:
            # Found only Inclusion heading, assume Exclusion follows
            logging.debug("Found ### Inclusion Criteria, assuming Exclusion follows.")
            inc_start_pos = inc_start_match.end()
            # Try to find the *next* heading within the block as the end for inclusion
            next_heading_match = re.search(r"(?m)^[ \t]*#{3}[ \t]+", eligibility_block[inc_start_pos:], re.IGNORECASE)
            if next_heading_match:
                 inc_end_pos = inc_start_pos + next_heading_match.start()
                 exc_text = eligibility_block[inc_end_pos:].strip() # Assume rest is exclusion
            else:
                 inc_end_pos = len(eligibility_block) # Runs to end if no other ### found
                 exc_text = None # Cannot determine exclusion start
                 
            inc_text = eligibility_block[inc_start_pos:inc_end_pos].strip()

        else:
            # Did not find specific ### headings, treat whole block as combined?
            # Or maybe just inclusion? This is ambiguous based on current info.
            # Safest fallback for now: assign to inclusion, leave exclusion None.
            logging.warning("Could not find specific ### Inclusion/Exclusion headings within ## Eligibility Criteria block. Assigning block to inclusion.")
            inc_text = eligibility_block # Treat the whole block as inclusion for now
            exc_text = None
            
    else:
         # Fallback to original method if main ## Eligibility Criteria block isn't found
         logging.warning("Could not find ## Eligibility Criteria block, falling back to original parsing method.")
         inc_text = get_text_between(markdown, "### Inclusion Criteria", "### Exclusion Criteria")
         exc_text = get_text_between(markdown, "### Exclusion Criteria", "## Locations & Contacts")
         
    data['inclusion_criteria_text'] = inc_text
    data['exclusion_criteria_text'] = exc_text
    logging.debug(f"Extracted Inclusion Criteria (robust): {inc_text[:100] if inc_text else 'None'}...")
    logging.debug(f"Extracted Exclusion Criteria (robust): {exc_text[:100] if exc_text else 'None'}...")
    # --- End Robust Eligibility Parsing ---
    
    # Extract Description
    desc_text = get_text_between(markdown, "## Description", "## Eligibility Criteria")
    data['description_text'] = desc_text
    logging.debug(f"Extracted Description (raw): {desc_text[:100] if desc_text else 'None'}...")
    
    # Extract Objectives
    obj_text = get_text_between(markdown, "## Trial Objectives and Outline", "## Trial Phase & Type")
    data['objectives_text'] = obj_text
    logging.debug(f"Extracted Objectives (raw): {obj_text[:100] if obj_text else 'None'}...")

    phase_type_section = get_text_between(markdown, "## Trial Phase & Type", "## Lead Organization")
    if phase_type_section:
        phase_match = re.search(r"\*\*Trial Phase\*\*\s+(.*)", phase_type_section)
        data['phase'] = phase_match.group(1).strip() if phase_match else None
    else:
        data['phase'] = None
        logging.debug("Phase/Type section not found.")

    ids_section = get_text_between(markdown, "## Trial IDs", "Share this clinical trial")
    if ids_section:
        primary_id_match = re.search(r"- \*\*Primary ID\*\*\s+([^\n]+)", ids_section)
        data['primary_id'] = primary_id_match.group(1).strip() if primary_id_match else None
        nct_id_match = re.search(r"- \*\*ClinicalTrials\.gov ID\*\*\s+\[(NCT\d+)\]\(.*\)", ids_section)
        data['nct_id'] = nct_id_match.group(1).strip() if nct_id_match else None
    else:
        data['primary_id'] = None
        data['nct_id'] = None
        logging.debug("IDs section not found.")

    title_match = re.search(r"(?m)^# (.*)", markdown)
    data['title_from_markdown'] = title_match.group(1).strip() if title_match else None
    
    status_match = re.search(r"Trial Status:\s*([^\n]+)", markdown) 
    data['status'] = status_match.group(1).strip() if status_match else None

    # Combine eligibility text for embedding (use the potentially updated inc_text/exc_text)
    inc = data.get('inclusion_criteria_text', "") or ""
    exc = data.get('exclusion_criteria_text', "") or ""
    data['eligibility_text'] = f"Inclusion Criteria:\n{inc}\n\nExclusion Criteria:\n{exc}".strip()

    if not data.get('title_from_markdown'):
        logging.warning("Could not parse title from markdown H1.")
        
    logging.debug("--- Parsing Markdown End ---") # Add debug end
    return data

def initialize_sqlite(db_path):
    """Connects to SQLite DB and creates table if not exists."""
    logging.info(f"Initializing SQLite database at {db_path}...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Add ai_summary column
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS clinical_trials (
            source_url TEXT PRIMARY KEY,
            nct_id TEXT,
            primary_id TEXT,
            title TEXT,
            status TEXT,
            phase TEXT,
            description_text TEXT,
            inclusion_criteria_text TEXT,
            exclusion_criteria_text TEXT,
            objectives_text TEXT,
            eligibility_text TEXT,
            raw_markdown TEXT,
            metadata_json TEXT,
            ai_summary TEXT DEFAULT NULL  -- Add new column for precomputed summary
        );
        """
        cursor.execute(create_table_sql)
        # Attempt to add column if table already exists (idempotent)
        try:
            cursor.execute("ALTER TABLE clinical_trials ADD COLUMN ai_summary TEXT DEFAULT NULL;")
            logging.info("Added 'ai_summary' column to existing table.")
        except sqlite3.OperationalError as alter_err:
            # Ignore error if column already exists
            if "duplicate column name" in str(alter_err).lower():
                logging.info("'ai_summary' column already exists.")
            else:
                raise # Re-raise unexpected alter table errors

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON clinical_trials (status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase ON clinical_trials (phase);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nct_id ON clinical_trials (nct_id);")
        conn.commit()
        logging.info("SQLite database and table initialized/updated successfully.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"SQLite error during initialization/update: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error initializing/updating SQLite: {e}")
        return None

def initialize_chromadb(path, collection_name):
    """Initializes ChromaDB client and gets/creates a collection using Google Embeddings."""
    logging.info(f"Initializing ChromaDB at {path} for collection '{collection_name}' with Google Embeddings...")
    if not GOOGLE_API_KEY:
        logging.error("GOOGLE_API_KEY not found. Cannot initialize Google Embedding Function for ChromaDB.")
        return None
    try:
        google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_API_KEY,
            model_name=EMBEDDING_MODEL_GOOGLE
        )
        client = chromadb.PersistentClient(path=path)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=google_ef, # Pass the Google embedding function
            metadata={"hnsw:space": "cosine"} 
        )
        logging.info("ChromaDB client and collection initialized successfully with Google Embeddings.")
        return collection
    except Exception as e:
        logging.error(f"Failed to initialize ChromaDB with Google Embeddings: {e}", exc_info=True)
        return None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Clinical Trial Local Loading Script --- (with AI Summary Pre-computation)")

    # Initialize LLM Client for Summarization (and ensure GOOGLE_API_KEY is checked for it)
    llm_client = None
    if GOOGLE_API_KEY:
        try:
            logging.info("Configuring Google Generative AI for summarization...")
            genai.configure(api_key=GOOGLE_API_KEY) # This configures genai globally for the API key
            llm_client = genai.GenerativeModel(LLM_MODEL_NAME)
            logging.info("Google Generative AI client for summarization initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Google Generative AI client for summarization: {e}. AI summaries will be skipped.")
    else:
        logging.warning("GOOGLE_API_KEY not found. AI summaries will be skipped (and ChromaDB embedding will fail if not already handled in initialize_chromadb).")

    # 1. Initialize SQLite (now includes ai_summary column)
    sql_conn = initialize_sqlite(SQLITE_DB_PATH)
    if not sql_conn:
        exit(1)
    sql_cursor = sql_conn.cursor()

    # 2. Initialize ChromaDB (this will use google_ef internally if GOOGLE_API_KEY is set)
    chroma_collection = initialize_chromadb(CHROMA_DB_PATH, CHROMA_COLLECTION_NAME)
    if not chroma_collection: # initialize_chromadb now returns None on GOOGLE_API_KEY failure
        logging.error("ChromaDB collection could not be initialized (likely missing GOOGLE_API_KEY or other init error). Exiting.")
        sql_conn.close()
        exit(1)

    # 3. Load Source JSON
    logging.info(f"Loading source JSON file: {SOURCE_JSON_PATH}")
    try:
        with open(SOURCE_JSON_PATH, 'r') as f:
            trials_data = json.load(f)
        logging.info(f"Successfully loaded {len(trials_data)} trials from JSON.")
    except Exception as e:
        logging.error(f"Error loading JSON file {SOURCE_JSON_PATH}: {e}")
        sql_conn.close()
        exit(1)

    # 4. Process and Load Trials
    logging.info("--- Starting data processing and loading (including AI summaries) ---")
    processed_count = 0
    error_count = 0
    chroma_batch_size = 100 
    # REMOVED 'embeddings' key from batch preparation
    chroma_batch = {"ids": [], "documents": [], "metadatas": []}

    for i, trial in enumerate(trials_data):
        logging.info(f"Processing trial {i+1}/{len(trials_data)}...")
        metadata = trial.get("metadata", {})
        markdown = trial.get("markdown", "")
        source_url = metadata.get("sourceURL")

        if not source_url:
            logging.warning(f"Skipping trial {i+1} due to missing 'sourceURL' in metadata.")
            error_count += 1
            continue

        try:
            parsed_data = parse_markdown_content(markdown)
            
            # --- Generate AI Summary ---
            ai_summary = "AI summary skipped: LLM client not initialized." # Default
            if llm_client and parsed_data.get("description_text") and parsed_data.get("objectives_text"):
                try:
                    summary_prompt = TRIAL_SUMMARY_PROMPT_TEMPLATE.format(
                        description=parsed_data["description_text"],
                        objectives=parsed_data["objectives_text"]
                    )
                    for attempt in range(2): # Simple retry
                        try:
                            response = llm_client.generate_content(summary_prompt)
                            ai_summary = response.text.strip()
                            logging.info(f"  Successfully generated AI summary for {source_url}.")
                            break 
                        except Exception as llm_err:
                            logging.warning(f"  LLM summarization attempt {attempt+1} failed for {source_url}: {llm_err}")
                            if attempt == 1: raise llm_err 
                            time.sleep(1) 
                except Exception as e:
                    logging.error(f"  Failed to generate AI summary for {source_url}: {e}")
                    ai_summary = "AI summary generation failed."
            elif not llm_client:
                pass 
            else:
                logging.warning(f"  Skipping AI summary for {source_url} due to missing description or objectives.")
                ai_summary = "AI summary skipped: Missing description/objectives."

            title = parsed_data.get('title_from_markdown') or metadata.get('title', 'Title Not Found')
            nct_id = parsed_data.get('nct_id')

            sql_data = (
                source_url, nct_id, parsed_data.get('primary_id'), title, parsed_data.get('status'),
                parsed_data.get('phase'), parsed_data.get('description_text'),
                parsed_data.get('inclusion_criteria_text'), parsed_data.get('exclusion_criteria_text'),
                parsed_data.get('objectives_text'), parsed_data.get('eligibility_text'),
                markdown, json.dumps(metadata), ai_summary
            )
            sql_cursor.execute("INSERT OR REPLACE INTO clinical_trials (source_url, nct_id, primary_id, title, status, phase, description_text, inclusion_criteria_text, exclusion_criteria_text, objectives_text, eligibility_text, raw_markdown, metadata_json, ai_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sql_data)

            eligibility_text = parsed_data.get('eligibility_text', '')
            if eligibility_text:
                # REMOVED manual embedding: embedding = google_ef([eligibility_text])[0].tolist()
                chroma_batch["ids"].append(source_url)
                # REMOVED: chroma_batch["embeddings"].append(embedding)
                chroma_batch["documents"].append(eligibility_text)
                chroma_batch["metadatas"].append({
                    "nct_id": nct_id, 
                    "source_url": source_url, 
                    "title": title, 
                    "status": parsed_data.get('status')
                })
            else:
                logging.warning(f"  Skipping ChromaDB entry for {source_url} due to missing eligibility text.")

            if (i + 1) % 50 == 0:
                sql_conn.commit()
                logging.info(f"  Committed SQLite changes after {i+1} trials.")
            
            if len(chroma_batch["ids"]) >= chroma_batch_size:
                logging.info(f"  Upserting batch of {len(chroma_batch['ids'])} documents to ChromaDB (embeddings will be generated by collection).")
                try:
                    chroma_collection.upsert(
                        ids=chroma_batch["ids"],
                        # REMOVED: embeddings=chroma_batch["embeddings"],
                        documents=chroma_batch["documents"],
                        metadatas=chroma_batch["metadatas"]
                    )
                    logging.info(f"  Successfully upserted batch to ChromaDB.")
                    chroma_batch = {"ids": [], "documents": [], "metadatas": []} # Reset batch
                except Exception as chroma_err:
                    logging.error(f"  Error upserting batch to ChromaDB: {chroma_err}")
                    chroma_batch = {"ids": [], "documents": [], "metadatas": []} # Reset batch
                    error_count += chroma_batch_size 

            processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process trial {i+1} ({source_url}): {e}", exc_info=True)
            error_count += 1

    if chroma_batch["ids"]:
        logging.info(f"Upserting final batch of {len(chroma_batch['ids'])} documents to ChromaDB (embeddings will be generated by collection).")
        try:
            chroma_collection.upsert(
                ids=chroma_batch["ids"],
                # REMOVED: embeddings=chroma_batch["embeddings"],
                documents=chroma_batch["documents"],
                metadatas=chroma_batch["metadatas"]
            )
            logging.info("Successfully upserted final batch to ChromaDB.")
        except Exception as chroma_err:
            logging.error(f"Error upserting final batch to ChromaDB: {chroma_err}")
            error_count += len(chroma_batch["ids"])

    sql_conn.commit()
    logging.info("Final SQLite commit complete.")
    sql_conn.close()
    logging.info("SQLite connection closed.")

    logging.info(f"--- Processing complete ---")
    logging.info(f"Successfully processed: {processed_count} trials")
    logging.info(f"Errors encountered: {error_count} trials") 