"""
Agent responsible for finding relevant clinical trials.
"""

import json
import os
import sqlite3
import pprint
import logging
import re # <-- Import re
import asyncio # <-- Import asyncio
from typing import Any, Dict, Optional, List, Tuple # <-- Add Tuple
from pathlib import Path # Import Path

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.generativeai.types import GenerationConfig # Added for JSON output
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# Import the base class
from backend.core.agent_interface import AgentInterface

# --- NEW Import --- 
from backend.agents.action_suggester import get_action_suggestions_for_trial

# --- Configuration ---
# Explicitly load .env from the backend directory
# Assumes this script is in backend/agents/
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"Attempting to load .env from: {dotenv_path}") # Add print statement

# SQLITE_DB_PATH = "backend/db/trials.db" # Old relative path
SQLITE_DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "clinical_trials.db") # Standardized path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # oncology-coPilot-main
# CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")
CHROMA_DB_PATH = str(PROJECT_ROOT / "backend" / "data" / "chroma_data") # Standardized path within backend
CHROMA_COLLECTION_NAME = "clinical_trials_eligibility"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
N_CHROMA_RESULTS = 10 # Number of results to fetch from ChromaDB
# LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = "gemini-1.5-pro"
DEFAULT_LLM_GENERATION_CONFIG = GenerationConfig(
    temperature=0.2, 
    max_output_tokens=8192 # Keep token limit
)

# --- RE-ADD MISSING CONSTANT --- 
SAFETY_SETTINGS = { # Adjust safety settings as needed
    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
}
# --- END RE-ADD --- 

# --- Structured Text Prompt --- 
ELIGIBILITY_AND_NARRATIVE_SUMMARY_PROMPT_TEMPLATE = """
Analyze the patient's eligibility for the following clinical trial based ONLY on the provided information. Provide a concise patient-specific summary, an overall eligibility status, and a breakdown of met, unmet, and unclear criteria.

**Patient Profile:**
{patient_profile_json}

**Clinical Trial Criteria:**
Trial Title: {trial_title}
Trial Status: {trial_status}
Trial Phase: {trial_phase}
Inclusion Criteria:
{inclusion_criteria}
Exclusion Criteria:
{exclusion_criteria}

**Instructions & Output Format (Plain Text ONLY):**
1.  Carefully compare the patient profile against *each* inclusion and exclusion criterion.
2.  Generate a concise patient-specific narrative summary (2-3 sentences).
3.  Determine an overall eligibility status string ('Likely Eligible', 'Likely Ineligible', 'Eligibility Unclear due to missing info').
4.  List the criteria under the appropriate headers below. For each criterion listed, you MUST include the original snippet from the trial's Inclusion or Exclusion criteria text.
5.  **Respond ONLY with plain text** following this structure precisely. Use the exact markers (e.g., `== SUMMARY ==`) and bullet points (`* `).
6.  Do NOT include any JSON or markdown formatting like ```.

== SUMMARY ==
[Your 2-3 sentence narrative summary here]

== ELIGIBILITY ==
[Your overall eligibility assessment string here]

== MET CRITERIA ==
* [Met Criterion 1 Text] - TRIAL_SNIPPET: "[Exact snippet from Inclusion/Exclusion for Met Criterion 1]"
* [Met Criterion 2 Text] - TRIAL_SNIPPET: "[Exact snippet from Inclusion/Exclusion for Met Criterion 2]"
... (Use "None" on a single line if no criteria met)

== UNMET CRITERIA ==
* [Unmet Criterion 1 Text] - TRIAL_SNIPPET: "[Exact snippet for Unmet Criterion 1]" - Reasoning: [Reasoning for unmet criterion 1]
* [Unmet Criterion 2 Text] - TRIAL_SNIPPET: "[Exact snippet for Unmet Criterion 2]" - Reasoning: [Reasoning for unmet criterion 2]
... (Use "None" on a single line if no criteria unmet)

== UNCLEAR CRITERIA ==
* [Unclear Criterion 1 Text] - TRIAL_SNIPPET: "[Exact snippet for Unclear Criterion 1]" - Reasoning: [Reasoning for unclear criterion 1, e.g., missing info]
* [Unclear Criterion 2 Text] - TRIAL_SNIPPET: "[Exact snippet for Unclear Criterion 2]" - Reasoning: [Reasoning for unclear criterion 2]
... (Use "None" on a single line if no criteria unclear)

**Important:**
*   For MET, UNMET, and UNCLEAR criteria, you MUST include the ` - TRIAL_SNIPPET: "[...snippet...]"` part. The snippet MUST be enclosed in double quotes.
*   Ensure reasoning is provided after ` - Reasoning: ` for UNMET and UNCLEAR criteria (this comes AFTER the TRIAL_SNIPPET).
*   If a category has no criteria, write exactly `None` on the line below the header.
*   Focus solely on the provided text. Do not infer information not present.
*   Be concise and specific in your reasoning.
"""
# --- End Structured Text Prompt --- 

# --- MockTrialDatabase Class (Commented out as it's being replaced) ---
# class MockTrialDatabase:
#     \"\"\" Simulates querying a clinical trial database. \"\"\"
#     def search_trials(self, condition: str, status: Optional[str] = None, phase: Optional[int] = None) -> list:
#         \"\"\" Simulates searching for trials based on condition. \"\"\"
#         print(f\"[MockTrialDatabase] Searching trials for condition: \'{condition}\', Status: {status}, Phase: {phase}\")
#         # ... (rest of mock logic) ...
#         return mock_results

class ClinicalTrialAgent(AgentInterface):
    """ Finds clinical trials relevant to a patient's condition using local DBs and LLM assessment. """

    def __init__(self):
        """
        Initialize the agent, including ChromaDB client and LLM client.
        Uses Google's embedding API instead of local models for better deployment compatibility.
        """
        self.chroma_client = None
        self.chroma_collection = None
        self.llm_client = None
        self.embedding_function = None

        # --- Initialize Google Generative AI Client ---
        if not GOOGLE_API_KEY:
            logging.error("GOOGLE_API_KEY not found in environment variables. LLM and embedding features will be disabled.")
            self.llm_client = None
            self.embedding_function = None
        else:
            try:
                logging.info("Configuring Google Generative AI...")
                genai.configure(api_key=GOOGLE_API_KEY)
                logging.info(f"Initializing Google Generative Model: {LLM_MODEL_NAME}")
                # Apply JSON config during initialization if possible, or during generate_content call
                self.llm_client = genai.GenerativeModel(
                    LLM_MODEL_NAME,
                    generation_config=DEFAULT_LLM_GENERATION_CONFIG # Apply config here
                )
                
                # Initialize Google embedding function for ChromaDB
                self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                    api_key=GOOGLE_API_KEY,
                    model_name="models/embedding-001"
                )
                logging.info("Google Generative AI client and embedding function initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize Google Generative AI client: {e}", exc_info=True)
                self.llm_client = None
                self.embedding_function = None

        # --- Initialize ChromaDB ---
        try:
            logging.info(f"Initializing ChromaDB client at: {CHROMA_DB_PATH}")
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            
            if self.embedding_function: # Only attempt to use ChromaDB if Google API key and EF are available
                collections = self.chroma_client.list_collections()
                col_names = [col.name for col in collections]
                if CHROMA_COLLECTION_NAME in col_names:
                    logging.info(f"Attempting to get ChromaDB collection: {CHROMA_COLLECTION_NAME} WITH Google Embedding Function.")
                    try:
                        self.chroma_collection = self.chroma_client.get_collection(
                            name=CHROMA_COLLECTION_NAME,
                            embedding_function=self.embedding_function # Crucial: provide the intended EF
                        )
                        logging.info(f"ChromaDB Collection '{CHROMA_COLLECTION_NAME}' loaded successfully WITH Google Embedding Function. Count: {self.chroma_collection.count()}")
                    except Exception as e:
                        logging.error(f"Failed to get ChromaDB collection '{CHROMA_COLLECTION_NAME}' WITH Google Embedding Function: {e}. This might be due to an embedding function mismatch if the collection was created with a different one or if there's an issue with the stored metadata. Will disable ChromaDB for this session.", exc_info=True)
                        self.chroma_collection = None # Disable Chroma if it can't be loaded with the correct EF
                else:
                    logging.warning(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' not found at path '{CHROMA_DB_PATH}'. Available collections: {col_names}. ChromaDB will be disabled.")
                    self.chroma_collection = None
            else:
                logging.warning("Google Embedding Function not available (likely missing GOOGLE_API_KEY). ChromaDB will be disabled.")
                self.chroma_collection = None

        except Exception as e: # Catches errors like PersistentClient creation failure
            logging.error(f"General failure during ChromaDB client initialization or collection listing: {e}", exc_info=True)
            self.chroma_client = None
            self.chroma_collection = None
        
        logging.info("ClinicalTrialAgent Initialized.")

    @property
    def name(self) -> str:
        return "clinical_trial_finder"

    @property
    def description(self) -> str:
        return "Searches for relevant clinical trials based on patient diagnosis, eligibility context, stage, biomarkers, etc. using local vector and relational databases."

    def _get_db_connection(self):
        """ Establishes a connection to the SQLite database. """
        try:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
            logging.info(f"Connected to SQLite DB: {SQLITE_DB_PATH}")
            return conn
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to SQLite DB at {SQLITE_DB_PATH}: {e}")
            return None

    def _build_query_text(self, context: Dict[str, Any], entities: Dict[str, Any], prompt: str) -> str:
        """ Constructs the text to be embedded for searching based on available info. """
        patient_data = context.get("patient_data", {})
        primary_diagnosis = patient_data.get("diagnosis", {}).get("primary")
        stage = patient_data.get("diagnosis", {}).get("stage")
        biomarkers = patient_data.get("biomarkers", []) # Assuming biomarkers is a list
        prior_treatments = patient_data.get("prior_treatments", []) # Assuming treatments is a list

        # Use specific entities if available
        condition = entities.get("condition", entities.get("specific_condition"))
        phase = entities.get("trial_phase")
        status = entities.get("recruitment_status")

        # Construct query string - prioritize explicit query terms
        parts = []
        if condition:
            parts.append(f"Condition: {condition}")
        elif primary_diagnosis:
             parts.append(f"Condition: {primary_diagnosis}")

        if stage: parts.append(f"Stage: {stage}")
        if phase: parts.append(f"Phase: {phase}")
        if status: parts.append(f"Status: {status}")
        if biomarkers: parts.append(f"Biomarkers: {', '.join(biomarkers)}")
        if prior_treatments: parts.append(f"Prior Treatments: {', '.join(pt.get('name', '') for pt in prior_treatments if pt.get('name'))}")

        # If specific parts identified, use them primarily
        if parts:
             query_text = ". ".join(parts)
             logging.info(f"Using constructed query text: {query_text}")
             return query_text
        # Fallback to using the original prompt if no structured data found
        elif prompt:
             logging.info(f"Using original prompt for query text: {prompt}")
             return prompt
        # Final fallback if prompt is also empty
        else:
             logging.warning("No suitable query text could be constructed.")
             return ""

    # --- Refined LLM Helper - Calls NEW Text Parser --- 
    async def _get_llm_assessment_for_trial(self, patient_context: Dict[str, Any], trial_detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generates prompt, calls LLM for STRUCTURED TEXT, parses text response for a single trial."""
        nct_id = trial_detail.get("nct_id", "UNKNOWN_ID")
        logging.info(f"Starting LLM assessment (structured text) for trial {nct_id}...")
        
        inclusion_criteria = trial_detail.get('inclusion_criteria_text', None) 
        exclusion_criteria = trial_detail.get('exclusion_criteria_text', None) 

        if not inclusion_criteria and not exclusion_criteria:
            logging.warning(f"No criteria text found for trial {nct_id}, skipping LLM assessment.")
            # Return structure indicating skip
            return {
                "llm_eligibility_analysis": None,
                "overall_assessment": "Not Assessed (No Criteria Text)", 
                "narrative_summary": "Eligibility criteria text was missing or could not be retrieved for this trial."
            }
            
        if not self.llm_client:
            logging.error("LLM client not initialized. Cannot perform assessment.")
            return { 
                "llm_eligibility_analysis": None,
                "overall_assessment": "Assessment Failed (Setup Issue)",
                "narrative_summary": "The AI assessment client is not configured."
            }

        try:
            prompt = self._create_eligibility_prompt(
                patient_context, 
                trial_detail.get('brief_title','N/A'), 
                trial_detail.get('overall_status','N/A'), 
                trial_detail.get('phase','N/A'), 
                inclusion_criteria, 
                exclusion_criteria
            ) 
            
            # Use the default config (expects plain text now)
            response = await asyncio.to_thread(
                self.llm_client.generate_content,
                prompt,
                generation_config=DEFAULT_LLM_GENERATION_CONFIG, 
                safety_settings=SAFETY_SETTINGS
            )
            
            # --- Get raw response text --- 
            raw_response_text = ""
            try: 
                if response.parts:
                    raw_response_text = response.parts[0].text
                else:
                    raw_response_text = response.text
            except Exception as e:
                 # ... (keep robust text retrieval error handling) ...
                 logging.warning(f"Could not access response parts/text directly for {nct_id}: {e}")
                 try:
                      raw_response_text = response.text 
                 except AttributeError:
                       logging.error(f"Response object for {nct_id} has no 'text' or 'parts' attribute.", exc_info=True)
                       raw_response_text = "Error: Response object structure invalid."
                 except Exception as e2:
                      logging.error(f"Failed even getting response.text for {nct_id}: {e2}")
                      raw_response_text = "Error retrieving response text."
            # --- End response text extraction ---
            
            logging.debug(f"Raw LLM TEXT response for {nct_id}:\n{raw_response_text}")

            # --- Call NEW Structured Text Parser --- 
            parsed_assessment_dict = self._parse_structured_text_response(raw_response_text)

            if parsed_assessment_dict:
                logging.info(f"Successfully parsed structured text assessment for trial {nct_id}.")
                # The parser should return the dict in the expected nested format
                return {"llm_eligibility_analysis": parsed_assessment_dict} 
            else:
                logging.warning(f"Failed to parse structured text assessment for trial {nct_id}. Raw text logged.")
                return { # Return specific structure for parsing failure
                    "llm_eligibility_analysis": None,
                    "overall_assessment": "Assessment Failed (Text Parsing Error)",
                    "narrative_summary": f"The AI assessment could not be processed from text. Raw response logged."
                }

        except Exception as e:
            logging.error(f"Error during LLM API call for trial {nct_id}: {e}", exc_info=True)
            return { # Return specific structure for API call failure
                "llm_eligibility_analysis": None,
                "overall_assessment": "Assessment Failed (API Error)",
                "narrative_summary": f"An error occurred communicating with the AI: {e}"
            }
    # --- End Refined LLM Helper --- 

    def _fetch_trial_details(self, conn: sqlite3.Connection, nct_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetches full trial details from SQLite for given NCT IDs."""
        if not nct_ids:
            return []
        try:
            conn.row_factory = sqlite3.Row # Return rows as dict-like objects
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(nct_ids))
            # Select all columns needed by the frontend/LLM
            query = f"SELECT * FROM clinical_trials WHERE nct_id IN ({placeholders})"
            cursor.execute(query, nct_ids)
            rows = cursor.fetchall()
            # Convert rows to dictionaries
            results = [dict(row) for row in rows]
            
            # Reorder results to match the input nct_ids order if needed (or handle later)
            # For simplicity now, return as fetched
            return results
        except sqlite3.Error as e:
            logging.error(f"SQLite error fetching trial details: {e}", exc_info=True)
            return []
        except Exception as e:
            logging.error(f"Unexpected error fetching trial details: {e}", exc_info=True)
            return []

    def _fallback_search_trials(self, query: str, limit: int = 10) -> List[str]:
        """
        Fallback search method that searches SQLite directly when ChromaDB is not available.
        Uses basic text matching on trial titles and eligibility/description text.
        """
        try:
            conn = self._get_db_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            
            # Search in title, eligibility_text, and description_text for the query term
            search_query = """
            SELECT nct_id, title, eligibility_text 
            FROM clinical_trials 
            WHERE title LIKE ? OR eligibility_text LIKE ? OR description_text LIKE ?
            LIMIT ?
            """
            
            search_term = f"%{query}%"
            logging.info(f"Fallback SQLite search using query: '{query}' on columns: title, eligibility_text, description_text") # Updated log
            cursor.execute(search_query, (search_term, search_term, search_term, limit))
            results = cursor.fetchall()
            
            nct_ids = [row[0] for row in results]
            logging.info(f"Fallback search found {len(nct_ids)} trials matching '{query}'")
            conn.close()
            return nct_ids
            
        except Exception as e:
            logging.error(f"Fallback search failed: {e}", exc_info=True)
            return []

    # --- NEW: Prompt Generation Method --- 
    def _create_eligibility_prompt(self, patient_context: Dict[str, Any], trial_title: str, trial_status: str, trial_phase: str, inclusion_criteria: Optional[str], exclusion_criteria: Optional[str]) -> str:
        """Creates the prompt for the LLM to assess eligibility and summarize using structured text, handling potentially missing criteria text."""
        # Basic formatting for patient context
        # --- FIX: Use json.dumps for reliable formatting --- 
        try:
            patient_profile_json = json.dumps(patient_context, indent=2)
        except TypeError as e:
            logging.error(f"Patient context is not JSON serializable: {e}. Using basic string representation.")
            patient_profile_json = str(patient_context)
        # --- END FIX --- 
            
        # --- FIX: Format with all arguments for the structured text prompt --- 
        prompt = ELIGIBILITY_AND_NARRATIVE_SUMMARY_PROMPT_TEMPLATE.format(
             patient_profile_json=patient_profile_json,
             trial_title=trial_title,             # Use argument
             trial_status=trial_status,           # Use argument
             trial_phase=trial_phase,             # Use argument
             inclusion_criteria=inclusion_criteria or "(Not provided or not found in source document)",
             exclusion_criteria=exclusion_criteria or "(Not provided or not found in source document)"
         )
        # --- END FIX ---
        return prompt
    # --- END NEW: Prompt Generation Method ---

    # --- NEW: Manual Structured Text Parser --- 
    def _parse_structured_text_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parses the structured plain text response from the LLM."""
        if not response_text:
            logging.warning("LLM structured text response is empty.")
            return None

        try:
            summary = ""
            eligibility_summary = ""
            met_criteria = []
            unmet_criteria = []
            unclear_criteria = []

            # Define markers
            markers = {
                "SUMMARY": "== SUMMARY ==",
                "ELIGIBILITY": "== ELIGIBILITY ==",
                "MET": "== MET CRITERIA ==",
                "UNMET": "== UNMET CRITERIA ==",
                "UNCLEAR": "== UNCLEAR CRITERIA =="
            }
            
            # --- Helper to extract text between markers --- 
            def extract_section(text, start_marker, all_markers):
                start_idx = text.find(start_marker)
                if start_idx == -1:
                    return "" # Marker not found
                
                start_idx += len(start_marker) # Move past the marker itself
                
                # Find the start of the *next* marker
                end_idx = len(text) # Default to end of text
                for marker_value in all_markers.values():
                    next_marker_idx = text.find(marker_value, start_idx)
                    if next_marker_idx != -1:
                         end_idx = min(end_idx, next_marker_idx)
                         
                return text[start_idx:end_idx].strip()
            # --- End Helper --- 

            # Extract sections
            summary_text = extract_section(response_text, markers["SUMMARY"], markers)
            eligibility_text = extract_section(response_text, markers["ELIGIBILITY"], markers)
            met_text = extract_section(response_text, markers["MET"], markers)
            unmet_text = extract_section(response_text, markers["UNMET"], markers)
            unclear_text = extract_section(response_text, markers["UNCLEAR"], markers)
            
            # Assign simple text sections
            summary = summary_text
            eligibility_summary = eligibility_text
            
            # --- Helper to parse bulleted list section --- 
            def parse_criteria_list(section_text, has_reasoning=False):
                items = []
                if not section_text or section_text.lower().strip() == 'none':
                     return items
                 
                lines = section_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('* '):
                         content_after_bullet = line[2:].strip()
                         
                         criterion_text = content_after_bullet
                         trial_snippet_to_assign = None 
                         reasoning_text = None

                         # Regex to find snippet like: - TRIAL_SNIPPET: "..." or - TRIAL_SNIPPET: '...'
                         # Simpler regex to avoid escaping issues with the toolchain
                         snippet_pattern = r' - TRIAL_SNIPPET: (["](?:\\.|[^"])*["]|\'(?:\\.|[^\'])*\')'
                         snippet_match = re.search(snippet_pattern, content_after_bullet)

                         if snippet_match:
                             trial_snippet_with_quotes = snippet_match.group(1)
                             trial_snippet_to_assign = trial_snippet_with_quotes.strip("\'\"") # Strip both single and double quotes
                            
                             criterion_text = content_after_bullet[:snippet_match.start()].strip()
                             remaining_text = content_after_bullet[snippet_match.end():].strip()
                            
                             if has_reasoning and remaining_text.startswith('- Reasoning:'):
                                 reasoning_text = remaining_text[len('- Reasoning:'):].strip()
                             elif has_reasoning and remaining_text:
                                 logging.warning(f"Expected reasoning after TRIAL_SNIPPET but marker not found: {remaining_text} in line: {line}")
                                 reasoning_text = remaining_text 
                        
                         elif has_reasoning:
                             parts = re.split(r'\\s+-\\s+Reasoning:\\s+', content_after_bullet, maxsplit=1)
                             if len(parts) == 2:
                                  criterion_text = parts[0].strip()
                                  reasoning_text = parts[1].strip()
                             else:
                                  criterion_text = content_after_bullet
                                  logging.warning(f"Could not parse reasoning (TRIAL_SNIPPET not found) from: {line}")
                         else: 
                             criterion_text = content_after_bullet

                         items.append({
                             "criterion": criterion_text,
                             "trial_snippet": trial_snippet_to_assign,
                             "reasoning": reasoning_text
                         })
                return items
            # --- End Helper --- 

            # Parse criteria lists
            met_criteria = parse_criteria_list(met_text, has_reasoning=False)
            unmet_criteria = parse_criteria_list(unmet_text, has_reasoning=True)
            unclear_criteria = parse_criteria_list(unclear_text, has_reasoning=True)
            
            # --- Re-categorize MET criteria if snippet is "N/A" or missing --- 
            newly_unclear_from_met = []
            remaining_met_criteria = []
            for criterion_obj in met_criteria:
                snippet = criterion_obj.get("trial_snippet")
                
                # Alternative snippet cleaning
                cleaned_snippet = None
                if snippet is not None:
                    cleaned_snippet = str(snippet).replace('"', '').replace("'", "")
                
                if not cleaned_snippet or cleaned_snippet.upper() == "N/A": # Case-insensitive check for N/A
                    # Preserve existing reasoning if any, otherwise add default
                    existing_reasoning = criterion_obj.get("reasoning")
                    criterion_obj["reasoning"] = existing_reasoning if existing_reasoning else "LLM reported N/A for trial snippet or snippet was missing/empty, making status unclear."
                    newly_unclear_from_met.append(criterion_obj)
                else:
                    # If snippet is not "N/A" and not empty, put the cleaned version back
                    criterion_obj["trial_snippet"] = cleaned_snippet 
                    remaining_met_criteria.append(criterion_obj)
            
            met_criteria = remaining_met_criteria
            unclear_criteria.extend(newly_unclear_from_met)
            # --- End re-categorization --- 

            # --- Construct the final dictionary in the expected nested format --- 
            result_dict = {
                "patient_specific_summary": summary,
                "eligibility_assessment": {
                    "eligibility_summary": eligibility_summary,
                    "met_criteria": met_criteria,
                    "unmet_criteria": unmet_criteria,
                    "unclear_criteria": unclear_criteria
                }
            }
            
            # Basic validation: Check if essential parts were extracted
            if not summary or not eligibility_summary:
                 logging.warning("Manual text parsing failed to extract summary or eligibility status.")
                 # Optionally return None or the partial dict depending on desired strictness
                 # return None 
                 
            # --- Log the final constructed dictionary --- 
            logging.debug(f"Constructed dict from text parser: {json.dumps(result_dict, indent=2)}")
            # --- End Log --- 
            return result_dict

        except Exception as e:
            logging.error(f"Error parsing structured text response: {e}\nRaw text was:\n{response_text[:1000]}...", exc_info=True)
            return None # Or raise, or return a specific error structure
    # --- End Manual Structured Text Parser --- 

    # --- NEW: Method to run analysis on a SINGLE trial object --- 
    async def run_single_trial_analysis(self, trial_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the LLM eligibility assessment for a single, already fetched trial data dictionary.
        This is used by the /api/trial-details/{trial_id} endpoint.

        Args:
            trial_data: A dictionary containing the details of the clinical trial.
            patient_data: A dictionary containing the relevant patient context.

        Returns:
            A dictionary containing the LLM's assessment results (or error/skip indicators),
            structured similarly to the output of _get_llm_assessment_for_trial.
        """
        nct_id = trial_data.get("nct_id", "UNKNOWN_ID")
        logging.info(f"Running single trial analysis for {nct_id}")
        
        if not trial_data:
            logging.warning("run_single_trial_analysis called with empty trial_data.")
            return {
                "llm_eligibility_analysis": None,
                "overall_assessment": "Error (Missing Trial Data)",
                "narrative_summary": "Trial data was not provided for analysis."
            }
        
        if not patient_data:
             logging.warning(f"run_single_trial_analysis called with empty patient_data for {nct_id}.")
             # Decide if this is fatal or just continue without patient context
             # For now, let's return an error as assessment is patient-specific
             return {
                "llm_eligibility_analysis": None,
                "overall_assessment": "Error (Missing Patient Data)",
                "narrative_summary": "Patient data was not provided for analysis."
             }
             
        try:
            # Call the existing LLM assessment helper method
            assessment_result = await self._get_llm_assessment_for_trial(
                patient_context=patient_data, 
                trial_detail=trial_data
            )
            # The helper already formats the output correctly
            return assessment_result if assessment_result else {
                "llm_eligibility_analysis": None,
                "overall_assessment": "Error (Assessment Failed)",
                "narrative_summary": "The LLM assessment process failed unexpectedly."
            }
            
        except Exception as e:
            logging.error(f"Unexpected error in run_single_trial_analysis for {nct_id}: {e}", exc_info=True)
            return {
                "llm_eligibility_analysis": None,
                "overall_assessment": "Error (Internal Agent Error)",
                "narrative_summary": f"An unexpected error occurred within the agent: {e}"
            }
    # --- END NEW: Single Trial Analysis Method --- 

    async def run(self, patient_data: Dict[str, Any] = None, prompt_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """ Executes the agent's logic: search trials, assess eligibility using TEXT LLM output. """
        query = prompt_details.get("prompt", "") if prompt_details else ""
        entities = prompt_details.get("entities", {}) if prompt_details else {}
        
        if not isinstance(patient_data, dict): 
             logging.warning(f"Received non-dict patient_data: {type(patient_data)}. Using empty dict.")
             patient_data = {}
        logging.info(f"ClinicalTrialAgent running. Query: '{query}'. Patient data provided: {bool(patient_data)}")

        # ChromaDB is optional - we can use fallback search if not available
        if not self.chroma_collection:
            logging.info("ChromaDB not available, using fallback search method.")

        conn = None
        try:
            # --- 1. Search for Trials --- 
            found_nct_ids = []
            
            if self.chroma_collection:
                # Use ChromaDB vector search
                logging.info(f"Using ChromaDB vector search for query: {query[:50]}...")
                results = self.chroma_collection.query(
                    query_texts=[query],  # ChromaDB will handle embedding with the function
                    n_results=N_CHROMA_RESULTS,
                    include=["metadatas", "documents", "distances"]
                )
                
                if results and results.get('ids') and results['ids'][0]:
                    # Extract NCT IDs correctly from metadata
                    if results.get('metadatas') and results['metadatas'][0]:
                         found_nct_ids = [meta.get('nct_id') for meta in results['metadatas'][0] if meta.get('nct_id')]
                    logging.info(f"ChromaDB found potential trial IDs: {found_nct_ids}")
                else:
                    logging.info("No relevant trials found in ChromaDB vector search.")
            else:
                # Use fallback SQLite search
                logging.info(f"Using fallback SQLite search for query: {query[:50]}...")
                query_text = self._build_query_text({"patient_data": patient_data}, entities, query)
                found_nct_ids = self._fallback_search_trials(query_text or query, N_CHROMA_RESULTS)
                logging.info(f"Fallback search found trial IDs: {found_nct_ids}")
                
            if not found_nct_ids:
                logging.info("No relevant trials found in search.")
                return {"status": "success", "output": { "found_trials": [] }, "summary": "No relevant trials found in search."}

            # --- 2. Fetch Details from SQLite --- 
            conn = self._get_db_connection()
            if not conn:
                 return {"status": "failure", "output": None, "summary": "Database connection failed."}
            
            found_trials_details = self._fetch_trial_details(conn, found_nct_ids)
            logging.info(f"Fetched details for {len(found_trials_details)} trials from SQLite.")
            
            if not found_trials_details: # Handle case where IDs were found but no details in SQLite
                logging.warning(f"NCT IDs {found_nct_ids} found in search but no details found in SQLite.")
                return {"status": "success", "output": { "found_trials": [] }, "summary": "Trial details not found in database for search results."}

            # --- 3. Perform LLM Eligibility Assessment (Concurrent) --- 
            llm_assessment_tasks = []
            if self.llm_client and patient_data: 
                logging.info(f"Starting concurrent LLM assessments for {len(found_trials_details)} trials...")
                for trial_detail_row in found_trials_details: # Iterate through rows
                    plain_trial_detail = dict(trial_detail_row) # Convert row to dict
                    task = asyncio.create_task(
                        self._get_llm_assessment_for_trial(patient_data, plain_trial_detail) # Pass dict
                    )
                    llm_assessment_tasks.append(task)
            else:
                logging.warning("LLM client not initialized or no patient context provided. Skipping LLM assessment.")

            # --- 4. Gather LLM Results and Update Trial Details --- 
            llm_results = []
            if llm_assessment_tasks:
                llm_results = await asyncio.gather(*llm_assessment_tasks)
                logging.info(f"Completed LLM assessments processing for {len(llm_results)} tasks.")
            
            # Process results and add to trial details
            final_trials_output = [] # Build a new list of plain dicts
            for i, trial_detail_row in enumerate(found_trials_details):
                # Start with a fresh copy of the original trial data from SQLite
                current_trial_output = dict(trial_detail_row) 

                # Initialize interpreted_result structure for internal processing
                interpreted_result_internal = {}
                
                llm_result_dict = llm_results[i] if i < len(llm_results) and llm_results[i] else None
                
                # Temporary storage for what will become llm_assessment
                llm_assessment_data = {
                    "eligibility_status": "Not Assessed (No LLM Task)",
                    "summary": "Assessment not performed.",
                    "met_criteria": [],
                    "unmet_criteria": [],
                    "unclear_criteria": ["Review criteria manually."],
                }
                action_suggestions_data = []

                if llm_result_dict and llm_result_dict.get("llm_eligibility_analysis"):
                    parsed_analysis = llm_result_dict["llm_eligibility_analysis"]
                    eligibility_assessment_nested = parsed_analysis.get("eligibility_assessment", {})

                    # Populate llm_assessment_data
                    llm_assessment_data["eligibility_status"] = eligibility_assessment_nested.get("eligibility_summary", "Assessment Incomplete")
                    llm_assessment_data["summary"] = parsed_analysis.get("patient_specific_summary", "Summary not generated.")
                    
                    def get_full_criteria_objects(key_name):
                        criteria_obj_list = []
                        items = eligibility_assessment_nested.get(key_name, [])
                        if isinstance(items, list):
                            for item_obj in items:
                                if isinstance(item_obj, dict) and "criterion" in item_obj:
                                    # Ensure trial_snippet is present, even if None
                                    if "trial_snippet" not in item_obj:
                                        item_obj["trial_snippet"] = item_obj.pop("trial_document_snippet", None) # Handle rename if old key exists
                                    criteria_obj_list.append(item_obj)
                                elif isinstance(item_obj, str):
                                    logging.warning(f"Criterion item for {key_name} is a string, expected dict: {item_obj}")
                                    criteria_obj_list.append({"criterion": item_obj, "trial_snippet": None, "reasoning": None})
                        return criteria_obj_list

                    llm_assessment_data["met_criteria"] = get_full_criteria_objects("met_criteria")
                    llm_assessment_data["unmet_criteria"] = get_full_criteria_objects("unmet_criteria")
                    llm_assessment_data["unclear_criteria"] = get_full_criteria_objects("unclear_criteria")
                    
                    try:
                         logging.debug(f"Passing TEXT-PARSED eligibility assessment to suggester for {current_trial_output.get('nct_id')}: {json.dumps(eligibility_assessment_nested, indent=2)}")
                         suggestions = get_action_suggestions_for_trial(
                             eligibility_assessment=eligibility_assessment_nested,
                             patient_context=patient_data
                         )
                         logging.debug(f"Received suggestions from suggester for {current_trial_output.get('nct_id')}: {suggestions}")
                         action_suggestions_data = suggestions
                    except Exception as suggester_err:
                         logging.error(f"Error generating action suggestions for trial {current_trial_output.get('nct_id', 'UNKNOWN')}: {suggester_err}", exc_info=True)
                         action_suggestions_data = []
                
                elif llm_result_dict: # LLM task ran but parsing might have failed or assessment was negative
                    llm_assessment_data["eligibility_status"] = llm_result_dict.get("overall_assessment", "Assessment Status Unknown")
                    llm_assessment_data["summary"] = llm_result_dict.get("narrative_summary", "Assessment not performed or failed.")
                    if "failed" in llm_assessment_data["eligibility_status"].lower() or "unknown" in llm_assessment_data["eligibility_status"].lower() :
                         llm_assessment_data["unclear_criteria"] = [{"criterion": "Assessment failed or status unknown, review criteria manually.", "trial_snippet": None, "reasoning": None}]
                    else:
                         llm_assessment_data["unclear_criteria"] = [{"criterion": "Review criteria manually.", "trial_snippet": None, "reasoning": None}]
                    action_suggestions_data = []

                # Add the structured llm_assessment and action_suggestions to the current_trial_output
                current_trial_output['llm_assessment'] = llm_assessment_data
                current_trial_output['action_suggestions'] = action_suggestions_data
                
                # Remove the old 'interpreted_result' if it was added implicitly or explicitly before
                if 'interpreted_result' in current_trial_output:
                    del current_trial_output['interpreted_result']

                final_trials_output.append(current_trial_output)

            # --- 5. Return Results --- 
            logging.info(f"Agent run completed successfully. Returning {len(final_trials_output)} trials.")
            return {
                "status": "success", 
                "output": { "trials_with_assessment": final_trials_output } 
            }
            
        except Exception as e:
            logging.error(f"Error in ClinicalTrialAgent run: {e}", exc_info=True)
            # Ensure a consistent error structure is returned
            return {"status": "error", "summary": f"An internal error occurred in the agent: {str(e)}", "output": {}}
        finally:
            if conn:
                try:
                    conn.close()
                    logging.info("SQLite connection closed in agent run.")
                except Exception as db_close_err:
                     logging.error(f"Error closing SQLite connection: {db_close_err}")
    # --- End Run Method --- 

# Example Usage (for testing) - Keep commented out unless needed for direct testing
# if __name__ == '__main__':
#     import asyncio
#     import json
#
#     async def main():
#         agent = ClinicalTrialAgent()
#
#         # Ensure model and DB are loaded
#         if not agent.embedding_function or not agent.chroma_collection:
#              print("Agent initialization failed. Exiting.")
#              return
#
#         # Example 1: Using patient context (requires relevant data in DB)
#         ctx1 = {"patient_data": {
#                    "diagnosis": {"primary": "Advanced Follicular Lymphoma", "stage": "IV"},
#                    "biomarkers": ["High Tumor Burden", "FLIPI 4"],
#                    "prior_treatments": []
#                 }}
#         kw1 = {"prompt": "Find trials for this follicular lymphoma patient"}
#         print("\\n--- Running Test 1: Patient Context ---")
#         res1 = await agent.run(ctx1, **kw1)
#         print("Result 1:")
#         pprint.pprint(res1)
#         
#         # Example 2: Specifying criteria in prompt/entities
#         ctx2 = {"patient_data": {}}
#         kw2 = {
#             "prompt": "Find phase 1 AKT mutation trials",
#             "entities": {"condition": "solid tumors with AKT mutation", "trial_phase": "1"}
#         }
#         print("\\n--- Running Test 2: Entities/Prompt ---")
#         res2 = await agent.run(ctx2, **kw2)
#         print("Result 2:")
#         pprint.pprint(res2)
#         
#     asyncio.run(main()) 