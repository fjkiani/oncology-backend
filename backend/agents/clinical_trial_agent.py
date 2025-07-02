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

# --- NEW Imports for AstraDB and SentenceTransformer ---
from backend.database_connections import DatabaseConnections
# --- END NEW Imports ---

# --- NEW Import --- 
from backend.agents.action_suggester import get_action_suggestions_for_trial

# --- Configuration ---
# Explicitly load .env from the backend directory
# Assumes this script is in backend/agents/
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"Attempting to load .env from: {dotenv_path}") # Add print statement

# Define paths assuming the application root in the container is /app
# and this agent is at /app/backend/agents/clinical_trial_agent.py

# Path to the root of the deployed application (/app)
APP_ROOT_IN_CONTAINER = Path(__file__).resolve().parent.parent.parent

SQLITE_DB_PATH = str(APP_ROOT_IN_CONTAINER / "backend" / "data" / "clinical_trials.db")
CHROMA_DB_PATH = str(APP_ROOT_IN_CONTAINER / "backend" / "data" / "chroma_data")
CHROMA_COLLECTION_NAME = "clinical_trials_eligibility"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # This seems unused if Google Embeddings are primary
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
        Initialize the agent, including DB connections, embedding model, and LLM client.
        """
        self.db_manager = DatabaseConnections()
        self.embedding_model = None
        self.llm_client = None
        self.astra_collection = None

        # --- Initialize Sentence Transformer for Embeddings ---
        try:
            logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logging.info("Embedding model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize SentenceTransformer model: {e}", exc_info=True)

        # --- Initialize Google Generative AI Client for Assessments ---
        if not GOOGLE_API_KEY:
            logging.error("GOOGLE_API_KEY not found in environment variables. LLM assessment features will be disabled.")
        else:
            try:
                logging.info("Configuring Google Generative AI...")
                genai.configure(api_key=GOOGLE_API_KEY)
                self.llm_client = genai.GenerativeModel(
                    LLM_MODEL_NAME,
                    generation_config=DEFAULT_LLM_GENERATION_CONFIG
                )
                logging.info("Google Generative AI client initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize Google Generative AI client: {e}", exc_info=True)

        # --- Initialize AstraDB Connection ---
        try:
            logging.info("Initializing AstraDB connection via DBManager...")
            # The collection name here should match the one in the loading script
            self.astra_collection = self.db_manager.get_vector_db_collection("clinical_trials")
            if self.astra_collection:
                logging.info("AstraDB collection 'clinical_trials' loaded successfully.")
            else:
                logging.warning("Failed to load AstraDB collection. Vector search will be unavailable.")
        except Exception as e:
            logging.error(f"General failure during AstraDB connection initialization: {e}", exc_info=True)
            self.astra_collection = None
        
        logging.info("ClinicalTrialAgent Initialized.")

    @property
    def name(self) -> str:
        return "clinical_trial_finder"

    @property
    def description(self) -> str:
        return "Searches for relevant clinical trials based on patient diagnosis, eligibility context, stage, biomarkers, etc. using local vector and relational databases."

    def _get_db_connection(self):
        """ Establishes a connection to the SQLite database using the DBManager. """
        try:
            conn = self.db_manager.get_sqlite_connection()
            if conn:
                logging.info(f"Connected to SQLite DB via DBManager.")
            else:
                logging.error("Failed to get SQLite DB connection from DBManager.")
            return conn
        except Exception as e:
            logging.error(f"Error getting SQLite connection from DBManager: {e}")
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
        nct_id = trial_detail.get("id", "UNKNOWN_ID")
        logging.info(f"Starting LLM assessment (structured text) for trial {nct_id}...")
        
        inclusion_criteria = trial_detail.get('inclusion_criteria', None) 
        exclusion_criteria = trial_detail.get('exclusion_criteria', None) 

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
                trial_detail.get('title','N/A'), 
                trial_detail.get('status','N/A'), 
                trial_detail.get('phases','N/A'), 
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
            query = f"SELECT * FROM trials WHERE id IN ({placeholders})"
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

    def _fallback_search_trials(self, conn: sqlite3.Connection, query: str, limit: int = 10) -> List[str]:
        """
        Fallback search method that searches SQLite directly.
        Uses the provided database connection.
        """
        try:
            if not conn:
                logging.error("Fallback search received no database connection.")
                return []
            
            conn.row_factory = sqlite3.Row # Set row_factory for dict-like access
            cursor = conn.cursor()
            
            # Search in title, summary, and criteria for the query term
            search_query = """
            SELECT id FROM trials 
            WHERE title LIKE ? OR summary LIKE ? OR inclusion_criteria LIKE ? OR exclusion_criteria LIKE ?
            LIMIT ?
            """
            
            search_term = f"%{query}%"
            logging.info(f"Fallback SQLite search using query: '{query}' on columns: title, summary, inclusion_criteria, exclusion_criteria")
            cursor.execute(search_query, (search_term, search_term, search_term, search_term, limit))
            results = cursor.fetchall()
            
            nct_ids = [row['id'] for row in results]
            logging.info(f"Fallback search found {len(nct_ids)} trials matching '{query}'")
            return nct_ids
            
        except Exception as e:
            logging.error(f"Fallback search failed: {e}", exc_info=True)
            return []

    def _vector_search_trials(self, query_text: str, n_results: int = 15) -> List[str]:
        """
        Performs a vector search in AstraDB to find relevant trial NCT IDs.
        """
        if not self.astra_collection:
            logging.error("AstraDB collection is not available. Cannot perform vector search.")
            return []
        
        if not self.embedding_model:
            logging.error("Embedding model is not available. Cannot perform vector search.")
            return []
            
        if not query_text:
            logging.warning("Query text is empty. Skipping vector search.")
            return []

        try:
            logging.info(f"Performing vector search in AstraDB with query: '{query_text}'")
            
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            results = self.astra_collection.find(
                sort={"$vector": query_embedding},
                limit=n_results,
                projection={"nct_id": 1} # Only need the ID
            )

            documents = list(results)
            if not documents:
                logging.info("AstraDB vector search returned no documents.")
                return []
            
            nct_ids = list(set([doc['nct_id'] for doc in documents if 'nct_id' in doc]))
            
            logging.info(f"AstraDB vector search found {len(nct_ids)} unique trials: {nct_ids}")
            return nct_ids

        except Exception as e:
            logging.error(f"AstraDB vector search failed: {e}", exc_info=True)
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
             trial_phase=json.dumps(trial_phase),             # Use argument, ensure it's a string
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
        Runs the full analysis pipeline for a single trial: LLM assessment and action suggestions.
        Returns a dictionary combining original trial data with analysis results,
        formatted for the frontend.
        """
        nct_id = trial_data.get("id", "UNKNOWN_ID")
        logging.info(f"Running single trial analysis for {nct_id}")
        
        # 1. Get LLM Assessment
        llm_assessment_result = await self._get_llm_assessment_for_trial(patient_data, trial_data)
        
        # 2. Get Action Suggestions and Format LLM Assessment
        parsed_analysis = {}
        if llm_assessment_result and llm_assessment_result.get("llm_eligibility_analysis"):
            parsed_analysis = llm_assessment_result["llm_eligibility_analysis"]

        eligibility_assessment = parsed_analysis.get("eligibility_assessment", {})
        action_suggestions = get_action_suggestions_for_trial(
            eligibility_assessment=eligibility_assessment,
            patient_context=patient_data
        )
        
        # Create the llm_assessment object in the structure the frontend expects
        llm_assessment_for_frontend = {
            "summary": parsed_analysis.get("patient_specific_summary", "Summary not available."),
            "eligibility_status": eligibility_assessment.get("eligibility_summary", "Not Assessed"),
            "met_criteria": eligibility_assessment.get("met_criteria", []),
            "unmet_criteria": eligibility_assessment.get("unmet_criteria", []),
            "unclear_criteria": eligibility_assessment.get("unclear_criteria", [])
        }

        # 3. Transform and combine results for the frontend
        phases_list = json.loads(trial_data.get("phases", "[]"))
        phase_display = phases_list[0] if phases_list else "N/A"

        final_trial_output = {
            "nct_id": trial_data.get("id"),
            "title": trial_data.get("title"),
            "status": trial_data.get("status"),
            "phase": phase_display,
            "source_url": trial_data.get("source_url"),
            "llm_assessment": llm_assessment_for_frontend,
            "action_suggestions": action_suggestions
        }

        return final_trial_output

    async def run(self, patient_data: Dict[str, Any] = None, prompt_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for the agent.
        1. Takes a search query from `prompt_details`.
        2. Performs a vector search to find relevant trials.
        3. Fetches trial details from SQLite.
        4. If patient data is provided, performs an LLM assessment for each trial.
        5. Returns the found trials, with or without assessments.
        """
        logging.info("ClinicalTrialAgent started.")
        query = prompt_details.get("query", "")
        if not query:
            return {"status": "error", "message": "No search query provided."}

        conn = None # Initialize connection variable
        try:
            # 1. Search for trials
            nct_ids = self._vector_search_trials(query_text=query, n_results=15)
            
            # Get DB connection here, as it's needed for fallback or details fetching
            conn = self._get_db_connection()
            if not conn:
                return {"status": "error", "message": "Could not connect to the trial database."}

            # Use fallback search if vector search is unavailable or returns no results
            if not nct_ids:
                logging.warning("Vector search returned no results or failed. Attempting fallback search.")
                nct_ids = self._fallback_search_trials(conn, query=query, limit=10)
            
            if not nct_ids:
                return {
                    "status": "success", 
                    "message": "No trials found matching your query.", 
                    "found_trials": []
                }

            # 2. Fetch trial details from SQLite
            found_trials_details = self._fetch_trial_details(conn, nct_ids)

            if not found_trials_details:
                return {
                    "status": "success", 
                    "message": f"Could not retrieve details for found trial IDs: {nct_ids}", 
                    "found_trials": []
                }
            
            # 3. Perform LLM Assessment if patient context is provided
            if patient_data:
                logging.info(f"Patient context provided. Performing LLM assessment for {len(found_trials_details)} trials.")
                
                assessment_tasks = [self.run_single_trial_analysis(trial, patient_data) for trial in found_trials_details]
                assessed_trials = await asyncio.gather(*assessment_tasks)
                
                return {
                    "status": "success",
                    "message": f"Found and assessed {len(assessed_trials)} trials.",
                    "trials_with_assessment": assessed_trials 
                }
            else:
                # If no patient context, just return the found trials
                logging.info("No patient context provided. Returning trial details without assessment.")
                return {
                    "status": "success", 
                    "message": f"Found {len(found_trials_details)} trials.",
                    "found_trials": found_trials_details
                }
        finally:
            # Ensure the connection is closed at the end of the request
            if conn:
                conn.close()
                logging.info("SQLite connection closed at the end of agent run.")

# Example Usage (for testing) - Keep commented out unless needed for direct testing
# if __name__ == '__main__':
#     import asyncio
#     import json
#
#     async def main():
#         agent = ClinicalTrialAgent()
#
#         # Ensure model and DB are loaded
#         if not agent.embedding_model or not agent.astra_collection:
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