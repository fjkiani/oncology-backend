import os
import json
import re
import pprint
from dotenv import load_dotenv
from astrapy.db import AstraDB
from sentence_transformers import SentenceTransformer
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Load environment variables from .env file

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE_NAME", "default_keyspace") # Use the correct env var name if different
COLLECTION_NAME = "clinical_trials" # Your Astra DB collection name
SOURCE_JSON_PATH = "backend/documents.json" # Path to your downloaded JSON file
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # Or your chosen sentence transformer model
VECTOR_DIMENSION = 384 # Dimension of the chosen embedding model

# --- Helper Functions ---

def connect_astra_db():
    """Connects to Astra DB using environment variables."""
    logging.info(f"Connecting to Astra DB endpoint {ASTRA_DB_API_ENDPOINT} in keyspace {ASTRA_DB_KEYSPACE}...")
    if not ASTRA_DB_API_ENDPOINT or not ASTRA_DB_APPLICATION_TOKEN:
        logging.error("Missing Astra DB credentials in environment variables (ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN)")
        return None
    try:
        astra_db = AstraDB(
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )
        logging.info("Successfully connected to Astra DB.")
        return astra_db
    except Exception as e:
        logging.error(f"Failed to connect to Astra DB: {e}")
        return None

def get_text_between(text, start_heading, end_heading=None):
    """Extracts text between two headings (e.g., ## Heading1 and ## Heading2)."""
    # Escape special regex characters in headings
    start_pattern = re.escape(start_heading)
    # Regex to find the start heading (ensuring it's at the start of a line optionally preceded by spaces)
    start_regex = r"(?m)^[ \t]*" + start_pattern + r"[ \t]*\n"

    start_match = re.search(start_regex, text, re.IGNORECASE)
    if not start_match:
        return None # Start heading not found

    start_pos = start_match.end()

    end_pos = len(text) # Default to end of text if no end_heading
    if end_heading:
        end_pattern = re.escape(end_heading)
        # Regex to find the next heading (## or ###) at the start of a line
        # This makes it more robust if the exact end_heading isn't present or markdown levels change
        end_regex = r"(?m)^[ \t]*#{2,3}[ \t]+"
        end_match = re.search(end_regex, text[start_pos:], re.IGNORECASE)
        if end_match:
            end_pos = start_pos + end_match.start()

    return text[start_pos:end_pos].strip()

def parse_markdown_content(markdown):
    """Parses the markdown text to extract key sections."""
    data = {}

    # --- Extracting specific sections using headings ---
    # More specific headings first
    data['inclusion_criteria_text'] = get_text_between(markdown, "### Inclusion Criteria", "### Exclusion Criteria")
    data['exclusion_criteria_text'] = get_text_between(markdown, "### Exclusion Criteria", "## Locations & Contacts") # Assuming next L2 heading

    # General L2 headings
    data['description_text'] = get_text_between(markdown, "## Description", "## Eligibility Criteria")
    data['objectives_text'] = get_text_between(markdown, "## Trial Objectives and Outline", "## Trial Phase & Type")

    # --- Extracting Key Value pairs often found under specific headings ---
    # Trial Phase & Type Section
    phase_type_section = get_text_between(markdown, "## Trial Phase & Type", "## Lead Organization")
    if phase_type_section:
        phase_match = re.search(r"\*\*Trial Phase\*\*\s+(.*)", phase_type_section)
        data['phase'] = phase_match.group(1).strip() if phase_match else None
        # type_match = re.search(r"\*\*Trial Type\*\*\s+(.*)", phase_type_section) # Example if needed
        # data['trial_type'] = type_match.group(1).strip() if type_match else None

    # Trial IDs Section
    ids_section = get_text_between(markdown, "## Trial IDs", "Share this clinical trial")
    if ids_section:
        primary_id_match = re.search(r"- \*\*Primary ID\*\*\s+([^\n]+)", ids_section)
        data['primary_id'] = primary_id_match.group(1).strip() if primary_id_match else None
        nct_id_match = re.search(r"- \*\*ClinicalTrials\.gov ID\*\*\s+\[(NCT\d+)\]\(.*\)", ids_section)
        data['nct_id'] = nct_id_match.group(1).strip() if nct_id_match else None

    # --- Extracting information often near the top ---
    # Title (usually the first H1)
    title_match = re.search(r"(?m)^# (.*)", markdown)
    data['title_from_markdown'] = title_match.group(1).strip() if title_match else None

    # Status (often right below title)
    status_match = re.search(r"Trial Status:\s*(\w+)", markdown)
    data['status'] = status_match.group(1).strip() if status_match else None

    # Combine eligibility
    inc = data.get('inclusion_criteria_text', "") or ""
    exc = data.get('exclusion_criteria_text', "") or ""
    data['eligibility_text'] = f"Inclusion Criteria:\n{inc}\n\nExclusion Criteria:\n{exc}".strip()

    # Fallback/Verification for title using metadata title if markdown parse fails
    if not data.get('title_from_markdown'):
        logging.warning("Could not parse title from markdown H1.")
        # Title from metadata will be used later

    return data


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Clinical Trial JSON Loading Script ---")

    # 1. Connect to Astra DB
    astra_db = connect_astra_db()
    if not astra_db:
        exit(1) # Stop if connection fails

    try:
        collection = astra_db.collection(COLLECTION_NAME)
        logging.info(f"Accessed collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logging.error(f"Failed to access collection '{COLLECTION_NAME}': {e}")
        logging.info("Please ensure the collection exists and the API endpoint/token are correct.")
        logging.info("You might need to run the CQL commands from the previous step in the Astra DB console.")
        exit(1)

    # 2. Load Source JSON
    logging.info(f"Loading source JSON file: {SOURCE_JSON_PATH}")
    try:
        with open(SOURCE_JSON_PATH, 'r') as f:
            trials_data = json.load(f)
        logging.info(f"Successfully loaded {len(trials_data)} trials from JSON.")
    except FileNotFoundError:
        logging.error(f"Error: JSON file not found at {SOURCE_JSON_PATH}")
        exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred loading the JSON file: {e}")
        exit(1)

    # 3. Initialize Embedding Model
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        # Verify dimension (optional but recommended)
        test_embedding = model.encode(["test sentence"])
        if len(test_embedding[0]) != VECTOR_DIMENSION:
             logging.warning(f"Model dimension mismatch! Expected {VECTOR_DIMENSION}, got {len(test_embedding[0])}. Update VECTOR_DIMENSION if using a different model.")
             # Adjust VECTOR_DIMENSION if necessary, or handle error
             # VECTOR_DIMENSION = len(test_embedding[0]) # Example adjustment
        logging.info("Embedding model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load sentence transformer model '{EMBEDDING_MODEL}': {e}")
        exit(1)

    # 4. Process and Load Trials
    logging.info("--- Starting data processing and loading ---")
    processed_count = 0
    error_count = 0
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
            # Parse markdown content
            parsed_data = parse_markdown_content(markdown)

            # Combine parsed data with metadata
            astra_document = {
                "_id": source_url, # Using source_url as the document ID
                "source_url": source_url,
                "nct_id": parsed_data.get('nct_id'),
                "primary_id": parsed_data.get('primary_id'),
                "title": metadata.get("title") or parsed_data.get('title_from_markdown', 'N/A'), # Prefer metadata title
                "status": parsed_data.get('status'),
                "phase": parsed_data.get('phase'),
                "description_text": parsed_data.get('description_text'),
                "inclusion_criteria_text": parsed_data.get('inclusion_criteria_text'),
                "exclusion_criteria_text": parsed_data.get('exclusion_criteria_text'),
                "objectives_text": parsed_data.get('objectives_text'),
                "eligibility_text": parsed_data.get('eligibility_text'),
                "raw_markdown": markdown, # Store original markdown
                "metadata_json": json.dumps(metadata) # Store original metadata
            }

            # Generate embedding only if eligibility text exists
            if astra_document["eligibility_text"]:
                 # Ensure text is not excessively long - truncate if necessary (adjust limit as needed)
                 max_embed_length = 50000 # Example limit
                 if len(astra_document["eligibility_text"]) > max_embed_length:
                     logging.warning(f"Trial {source_url}: Eligibility text exceeds {max_embed_length} chars, truncating for embedding.")
                     text_to_embed = astra_document["eligibility_text"][:max_embed_length]
                 else:
                     text_to_embed = astra_document["eligibility_text"]

                 embedding = model.encode([text_to_embed])[0]
                 astra_document["eligibility_vector"] = embedding.tolist() # Convert numpy array to list
            else:
                 logging.warning(f"Trial {source_url}: No eligibility text found to generate vector.")
                 astra_document["eligibility_vector"] = None # Or omit the field

            # Upsert document into Astra DB
            # find_one_and_replace is safer if you need to ensure full overwrite based on _id
            # result = collection.find_one_and_replace(
            #      filter={"_id": source_url},
            #      replacement=astra_document,
            #      upsert=True
            # )

            # Using insert_one and catching duplicate key errors might be simpler if _id is guaranteed unique per run
            # Or use upsert (depending on astrapy version/preference)
            # Let's use simple insert and rely on _id uniqueness or script running once clean.
            # If rerunning is needed, delete collection or use upsert logic.
            # Update: Using upsert with the filter on _id is generally the robust approach for reruns.
            update_result = collection.find_one_and_update(
                filter={"_id": source_url},
                update={"$set": astra_document},
                upsert=True
            )

            # logging.info(f"Upserted trial {source_url}. Matched: {update_result.matched_count}, Modified: {update_result.modified_count}, Upserted ID: {update_result.upserted_id}")
            logging.info(f"Successfully processed and upserted trial {source_url}")
            processed_count += 1

        except Exception as e:
            logging.error(f"Error processing trial {i+1} ({source_url}): {e}", exc_info=True) # Log stack trace
            error_count += 1

    logging.info("--- Loading Complete ---")
    logging.info(f"Successfully processed: {processed_count} trials.")
    logging.info(f"Errors encountered: {error_count} trials.") 