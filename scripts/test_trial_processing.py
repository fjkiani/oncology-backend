# backend/scripts/test_trial_processing.py
import os
import pprint
from typing import List, Optional, Dict

from firecrawl import FirecrawlApp, JsonConfig
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
# Consider adding: from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# IMPORTANT: Set your Firecrawl API key as an environment variable 
# or replace 'YOUR_API_KEY' directly (less secure)
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY', 'YOUR_API_KEY')
EXAMPLE_TRIAL_URL = 'https://www.cancer.gov/research/participate/clinical-trials-search/v?id=NCI-2017-01579' # Follicular Lymphoma Example

# Mock Patient Context for Eligibility Search
MOCK_PATIENT_CONTEXT = "Patient with newly diagnosed Grade 2 Follicular Lymphoma, High Tumor Burden (FLIPI 4), ECOG 1, no prior chemo, ANC > 1000, Platelets > 75k, CrCl > 50"

# --- Step 1: Define Pydantic Schema for Firecrawl Extraction ---
class ClinicalTrialSchema(BaseModel):
    nct_id: Optional[str] = Field(description="The NCT ID of the trial", default=None)
    primary_id: Optional[str] = Field(description="The Primary Trial ID (e.g., PrE0403)", default=None)
    title: Optional[str] = Field(description="The main title of the clinical trial", default=None)
    status: Optional[str] = Field(description="The current status of the trial (e.g., Active, Complete)", default=None)
    phase: Optional[str] = Field(description="The phase of the trial (e.g., Phase II)", default=None)
    lead_org: Optional[str] = Field(description="The lead organization running the trial", default=None)
    raw_description_text: Optional[str] = Field(description="The full text content of the 'Description' section", default=None)
    raw_inclusion_criteria_text: Optional[str] = Field(description="The full text content of the 'Inclusion Criteria' section", default=None)
    raw_exclusion_criteria_text: Optional[str] = Field(description="The full text content of the 'Exclusion Criteria' section", default=None)
    raw_objectives_text: Optional[str] = Field(description="The full text content of the 'Trial Objectives and Outline' section", default=None)
    contacts_json: Optional[List[Dict[str, str]]] = Field(description="List of locations and contacts as structured objects if possible, otherwise null", default=None)
    raw_contacts_text: Optional[str] = Field(description="The raw text of the 'Locations & Contacts' section if structured extraction fails", default=None)
    # source_url will likely come from metadata, but can include if needed

# --- Main Test Execution --- 
def run_test():
    print(f"--- Starting Test for URL: {EXAMPLE_TRIAL_URL} ---")

    # --- Step 2: Run firecrawl.scrape_url with schema ---
    print("\n--- Step 2: Running Firecrawl Extraction ---")
    if FIRECRAWL_API_KEY == 'YOUR_API_KEY':
        print("ERROR: Please set your FIRECRAWL_API_KEY before running.")
        return

    try:
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        json_config = JsonConfig(
            extractionSchema=ClinicalTrialSchema.model_json_schema(),
            mode="llm-extraction",
            pageOptions={"onlyMainContent": False} # Get whole page to ensure sections aren't missed
        )
        llm_extraction_result = app.scrape_url(
            EXAMPLE_TRIAL_URL,
            formats=["json"],
            json_options=json_config
        )
    except Exception as e:
        print(f"ERROR during Firecrawl API call: {e}")
        return

    # --- Step 3: Verify structured JSON output ---
    print("\n--- Step 3: Verifying Firecrawl Output ---")
    if not llm_extraction_result or not llm_extraction_result.get('success'):
        print("ERROR: Firecrawl scrape was unsuccessful or returned empty.")
        pprint.pprint(llm_extraction_result)
        return

    structured_data = llm_extraction_result.get('data', {}).get('json')
    metadata = llm_extraction_result.get('data', {}).get('metadata')
    source_url = metadata.get('sourceURL', EXAMPLE_TRIAL_URL)

    if not structured_data:
        print("ERROR: Firecrawl returned success=true but no JSON data found.")
        pprint.pprint(llm_extraction_result)
        return

    print("Successfully extracted structured data:")
    pprint.pprint(structured_data)
    print("Metadata:")
    pprint.pprint(metadata)

    # --- Step 4: Extract and combine eligibility text ---
    print("\n--- Step 4: Extracting Eligibility Text ---")
    inclusion_text = structured_data.get('raw_inclusion_criteria_text', '') or ""
    exclusion_text = structured_data.get('raw_exclusion_criteria_text', '') or ""
    eligibility_text = f"INCLUSION CRITERIA:\n{inclusion_text}\n\nEXCLUSION CRITERIA:\n{exclusion_text}"
    if not inclusion_text and not exclusion_text:
        print("WARNING: No inclusion or exclusion criteria text extracted.")
    else:
        print(f"Combined Eligibility Text Length: {len(eligibility_text)} characters")
        # print(eligibility_text[:500] + "...") # Print start for verification

    # --- Step 5: Chunk eligibility text ---
    print("\n--- Step 5: Chunking Eligibility Text ---")
    # Simple newline splitting - consider RecursiveCharacterTextSplitter for more robustness
    chunks = [chunk for chunk in eligibility_text.split('\n\n') if chunk.strip()] 
    # Filter empty chunks
    if not chunks:
         print("WARNING: No chunks generated from eligibility text.")
         return # Cannot proceed without chunks
         
    print(f"Generated {len(chunks)} chunks.")
    print("First 3 Chunks:")
    pprint.pprint(chunks[:3])

    # --- Step 6: Embed chunks ---
    print("\n--- Step 6: Embedding Chunks (using 'all-MiniLM-L6-v2') ---")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(chunks)
        print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}.")
    except Exception as e:
        print(f"ERROR during embedding generation: {e}")
        return

    # --- Step 7: Load into in-memory chromadb ---
    print("\n--- Step 7: Loading into In-Memory ChromaDB Collection ---")
    try:
        client = chromadb.Client() # In-memory client
        # Ensure collection name is valid and delete if it exists from previous run
        collection_name = "trial_eligibility_test"
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except: 
            pass # Collection didn't exist, which is fine
            
        collection = client.create_collection(collection_name)
        collection.add(
            embeddings=embeddings.tolist(), # ChromaDB expects list of lists/np.ndarray
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        print(f"Added {collection.count()} items to Chroma collection '{collection_name}'.")
    except Exception as e:
        print(f"ERROR setting up or adding to ChromaDB: {e}")
        return

    # --- Step 8: Define mock patient context ---
    print("\n--- Step 8: Using Mock Patient Context ---")
    print(f"Context: {MOCK_PATIENT_CONTEXT}")

    # --- Step 9: Embed context and query chromadb ---
    print("\n--- Step 9: Querying ChromaDB for Relevant Chunks ---")
    try:
        context_embedding = embedding_model.encode([MOCK_PATIENT_CONTEXT])
        results = collection.query(
            query_embeddings=context_embedding.tolist(),
            n_results=5 # Get top 5 relevant chunks
        )
    except Exception as e:
        print(f"ERROR querying ChromaDB: {e}")
        return

    # --- Step 10: Verify retrieved chunks ---
    print("\n--- Step 10: Verifying Retrieved Chunks ---")
    if results and results.get('documents'):
        print("Top 5 relevant chunks retrieved based on patient context:")
        retrieved_docs = results['documents'][0] # Query returns a list of lists
        for i, doc in enumerate(retrieved_docs):
            print(f"  {i+1}. {doc[:150]}...") # Print snippet of each chunk
    else:
        print("ERROR: No documents retrieved from ChromaDB query.")
        pprint.pprint(results)

    # --- Step 11: Verify easy access to structured metadata ---
    print("\n--- Step 11: Verifying Metadata Access ---")
    print(f"Title: {structured_data.get('title', 'N/A')}")
    print(f"Status: {structured_data.get('status', 'N/A')}")
    print(f"Phase: {structured_data.get('phase', 'N/A')}")
    print(f"NCT ID: {structured_data.get('nct_id', 'N/A')}")
    print(f"Description Snippet: {(structured_data.get('raw_description_text', '') or "")[:100]}...")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_test() 