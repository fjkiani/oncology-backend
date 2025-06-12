"""
Agent responsible for finding and assessing clinical trials from the database.
"""
import json
import os
import sqlite3
import logging
import re
import asyncio
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import sys
import pprint

# Add project root to Python path to allow for absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from backend.core.agent_interface import AgentInterface
from backend.utils.database_connections import DatabaseConnections
from backend.agents.action_suggester import get_action_suggestions_for_trial

# --- Configuration ---
# Load .env from the project root
dotenv_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=dotenv_path)

APP_ROOT_IN_CONTAINER = Path(__file__).resolve().parent.parent.parent
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
N_VECTOR_SEARCH_RESULTS = 500  # Number of results to fetch from vector search
ELIGIBILITY_AND_NARRATIVE_SUMMARY_PROMPT_TEMPLATE = """
You are a clinical trial expert. Your task is to analyze a patient's medical information against a clinical trial's eligibility criteria and provide a structured summary.

**PATIENT PROFILE:**
```json
{patient_profile_json}
```

**CLINICAL TRIAL INFORMATION:**
- **Title:** {trial_title}
- **Status:** {trial_status}
- **Phase:** {trial_phase}

**TRIAL ELIGIBILITY CRITERIA:**
**Inclusion:**
```
{inclusion_criteria}
```
**Exclusion:**
```
{exclusion_criteria}
```

**YOUR TASK:**
Based on all the information, provide the following output in a structured format. Do NOT add any introductory or concluding text.

== SUMMARY ==
Provide a 2-3 sentence narrative summary of the trial, explaining its purpose and what it's investigating.

== ELIGIBILITY ==
Based on the patient's profile, determine if they are: `Likely Eligible`, `Potentially Eligible`, or `Likely Ineligible`.

== MET CRITERIA ==
List the specific inclusion criteria from the trial that the patient appears to meet. For each, provide the snippet from the trial criteria. Format as: `* [Criterion Text] - TRIAL_SNIPPET: "[Snippet]"`
If none, write `None`.

== UNMET CRITERIA ==
List the specific inclusion or exclusion criteria the patient does not meet. For each, provide the snippet from the trial criteria and a brief reasoning based on the patient's profile. Format as: `* [Criterion Text] - TRIAL_SNIPPET: "[Snippet]" - Reasoning: [Your Reasoning]`
If none, write `None`.

== UNCLEAR CRITERIA ==
List criteria where eligibility is uncertain due to missing patient information. For each, provide the snippet from the trial criteria and state what information is needed. Format as: `* [Criterion Text] - TRIAL_SNIPPET: "[Snippet]" - Reasoning: [Information Needed]`
If none, write `None`.
"""

# --- Constants ---
# The threshold for Jaccard similarity to consider a patient's criteria "met".
JACCARD_SIMILARITY_THRESHOLD = 0.5

class ClinicalTrialAgent(AgentInterface):
    """
    Agent responsible for finding and assessing clinical trials from the database.
    """
    def __init__(self):
        """Initializes the agent, database connections, and embedding model."""
        logging.info("Initializing ClinicalTrialAgent...")
        self.db_connections = DatabaseConnections()
        self.trials_collection = self.db_connections.get_vector_db_collection("clinical_trials")
        self.sqlite_conn = self.db_connections.get_sqlite_connection()
        logging.info("AstraDB and SQLite connections established successfully.")

        try:
            logging.info(f"Loading sentence transformer model '{EMBEDDING_MODEL_NAME}'...")
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME, cache_folder=str(APP_ROOT_IN_CONTAINER / "embedding_models")
            )
            logging.info(f"Sentence transformer model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model: {e}", exc_info=True)
            self.embedding_model = None

        # --- Re-enable the LLM Client ---
        try:
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                logging.warning("GOOGLE_API_KEY not found in .env file. LLM features will be disabled.")
                self.llm_client = None
            else:
                logging.info("Configuring Gemini client...")
                genai.configure(api_key=gemini_api_key)
                self.llm_client = genai.GenerativeModel('gemini-1.5-flash')
                logging.info("Gemini client configured successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            self.llm_client = None
            
        logging.info("ClinicalTrialAgent Initialized.")

    @property
    def name(self) -> str:
        return "ClinicalTrialAgent"

    @property
    def description(self) -> str:
        return "Finds relevant clinical trials based on patient data and medical context."

    def _get_db_connection(self):
        """Helper to get a new SQLite connection."""
        try:
            return self.db_connections.get_sqlite_connection()
        except sqlite3.Error as e:
            logging.error(f"Failed to get SQLite connection: {e}", exc_info=True)
            return None
    
    def _build_query_text(self, patient_data: Dict[str, Any], entities: Dict[str, Any], prompt: str) -> str:
        """
        Constructs a detailed, human-readable query string for vector search
        to improve the quality of embeddings.
        """
        description_parts = []
        
        # Start with a clear statement of intent
        base_sentence = "Clinical trials for a patient with"
        
        # Add diagnosis and stage
        if 'diagnosis' in patient_data:
            diag = patient_data['diagnosis']
            condition = diag.get('primary', '')
            stage = diag.get('stage', '')
            if stage:
                base_sentence += f" {stage}"
            if condition:
                base_sentence += f" {condition}"
        
        description_parts.append(base_sentence + ".")

        # Add biomarkers
        if 'biomarkers' in patient_data and patient_data['biomarkers']:
            biomarker_str = ', '.join(patient_data['biomarkers'])
            description_parts.append(f"The patient has the following biomarkers: {biomarker_str}.")

        # Add prior treatments
        if 'prior_treatments' in patient_data and patient_data['prior_treatments']:
            treatments = [t.get('name', '') for t in patient_data['prior_treatments']]
            treatment_str = ', '.join(filter(None, treatments))
            description_parts.append(f"They have received prior treatments including: {treatment_str}.")
            
        # Add the specific user query/prompt last for focus
        if prompt:
            description_parts.append(f"The search is focused on trials involving {prompt}.")
            
        return " ".join(description_parts)

    async def _get_llm_assessment_for_trial(self, patient_context: Dict[str, Any], trial_detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Uses an LLM to assess patient eligibility for a single trial and parse the structured response.
        """
        if not self.llm_client:
            logging.warning(f"LLM client not available. Skipping assessment for trial {trial_detail.get('id')}.")
            trial_detail['eligibility'] = {"status": "Assessment Skipped", "summary": "LLM client not available."}
            return trial_detail

        try:
            inclusion_criteria = trial_detail.get('inclusion_criteria')
            exclusion_criteria = trial_detail.get('exclusion_criteria')

            prompt = self._create_eligibility_prompt(
                patient_context=patient_context,
                trial_title=trial_detail.get('title', 'N/A'),
                trial_status=trial_detail.get('status', 'N/A'),
                trial_phase=trial_detail.get('phases', 'N/A'),
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria
            )
            
            response = await self.llm_client.generate_content_async(
                contents=[prompt],
            )
            
            parsed_assessment = self._parse_structured_text_response(response.text)
            trial_detail['eligibility'] = parsed_assessment
            return trial_detail
            
        except Exception as e:
            logging.error(f"Error during LLM assessment for trial {trial_detail.get('id')}: {e}", exc_info=True)
            trial_detail['eligibility'] = {"status": "Assessment Failed", "summary": f"An error occurred: {e}"}
            return trial_detail

    def _fetch_trial_details(self, nct_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches full trial details from SQLite for a list of NCT IDs.
        """
        if not nct_ids:
            return []
        
        conn = self._get_db_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in nct_ids)
            query = f"SELECT * FROM trials WHERE id IN ({placeholders})"
            cursor.execute(query, nct_ids)
            
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # The frontend expects specific key names. Let's transform the data.
            transformed_results = []
            for row in results:
                # The 'phases' from DB is a JSON string like '["Phase 2"]'. Let's parse it.
                phases_list = json.loads(row.get("phases", "[]"))
                phase_str = ", ".join(phases_list) if phases_list else "N/A"

                transformed_row = {
                    "id": row.get("id"),  # Use "id" as the key for NCT ID
                    "nctId": row.get("id"), # Keep nctId for good measure
                    "trial_id": row.get("id"), # Add trial_id as another possibility
                    "title": row.get("title"),
                    "status": row.get("status"),
                    "phase": phase_str, # Send a clean string
                    "summary": row.get("summary"),
                    "conditions": row.get("conditions"),
                    "interventions": row.get("interventions"),
                    "source": row.get("source"),
                    "inclusion_criteria": row.get("inclusion_criteria"),
                    "exclusion_criteria": row.get("exclusion_criteria"),
                    "eligibility": {"status": "Not Assessed"} # Add placeholder for initial view
                }
                transformed_results.append(transformed_row)

            results_dict = {row['id']: row for row in transformed_results}
            ordered_results = [results_dict[nct_id] for nct_id in nct_ids if nct_id in results_dict]
            
            return ordered_results
        except sqlite3.Error as e:
            logging.error(f"SQLite query failed: {e}", exc_info=True)
            return []
        finally:
            conn.close()

    def _fallback_search_trials(self, query: str, limit: int = 10) -> List[str]:
        """
        A fallback search method that uses simple SQL LIKE queries if vector search fails.
        """
        conn = self._get_db_connection()
        if not conn:
            return []
        try:
            cursor = conn.cursor()
            
            search_query = """
            SELECT id, title, summary
            FROM trials 
            WHERE title LIKE ? OR summary LIKE ?
            LIMIT ?
            """
            
            search_term = f"%{query}%"
            logging.info(f"Fallback SQLite search using query: '{query}' on columns: title, summary")
            cursor.execute(search_query, (search_term, search_term, limit))
            results = cursor.fetchall()
            
            nct_ids = [row[0] for row in results]
            logging.info(f"Fallback search found {len(nct_ids)} trials matching '{query}'")
            return nct_ids
            
        except Exception as e:
            logging.error(f"Fallback search failed: {e}", exc_info=True)
            return []
        finally:
            if conn:
                conn.close()

    def _create_eligibility_prompt(self, patient_context: Dict[str, Any], trial_title: str, trial_status: str, trial_phase: str, inclusion_criteria: Optional[str], exclusion_criteria: Optional[str]) -> str:
        """Creates the prompt for the LLM to assess eligibility and summarize using structured text."""
        try:
            patient_profile_json = json.dumps(patient_context, indent=2)
        except TypeError as e:
            logging.error(f"Patient context is not JSON serializable: {e}. Using basic string representation.")
            patient_profile_json = str(patient_context)
            
        prompt = ELIGIBILITY_AND_NARRATIVE_SUMMARY_PROMPT_TEMPLATE.format(
             patient_profile_json=patient_profile_json,
             trial_title=trial_title,
             trial_status=trial_status,
             trial_phase=trial_phase,
             inclusion_criteria=inclusion_criteria or "(Not provided)",
             exclusion_criteria=exclusion_criteria or "(Not provided)"
         )
        return prompt

    def _parse_structured_text_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """ Parses the custom structured text format from the LLM response. """
        try:
            all_markers = ["== SUMMARY ==", "== ELIGIBILITY ==", "== MET CRITERIA ==", "== UNMET CRITERIA ==", "== UNCLEAR CRITERIA =="]
            
            def extract_section(text, start_marker, all_markers):
                start_index = text.find(start_marker)
                if start_index == -1: return ""
                start_index += len(start_marker)
                
                end_index = len(text)
                for marker in all_markers:
                    next_marker_pos = text.find(marker, start_index)
                    if next_marker_pos != -1:
                        end_index = min(end_index, next_marker_pos)
                return text[start_index:end_index].strip()

            def parse_criteria_list(section_text, has_reasoning=False):
                if section_text.lower() == 'none': return []
                items = []
                for line in section_text.split('* '):
                    line = line.strip()
                    if not line: continue
                    
                    item = {}
                    snippet_marker = " - TRIAL_SNIPPET: \""
                    reasoning_marker = "\" - Reasoning: "
                    
                    snippet_start = line.find(snippet_marker)
                    if snippet_start == -1:
                        item['text'] = line
                        item['snippet'] = "N/A"
                        item['reasoning'] = "N/A"
                        items.append(item)
                        continue

                    item['text'] = line[:snippet_start].strip()
                    
                    if has_reasoning:
                        reasoning_start = line.find(reasoning_marker, snippet_start)
                        if reasoning_start != -1:
                            snippet_end = reasoning_start
                            item['reasoning'] = line[reasoning_start + len(reasoning_marker):].strip()
                        else:
                            snippet_end = line.rfind("\"")
                            item['reasoning'] = "N/A"
                    else:
                        snippet_end = line.rfind("\"")
                        item['reasoning'] = "N/A"
                        
                    item['snippet'] = line[snippet_start + len(snippet_marker):snippet_end].strip()
                    items.append(item)
                return items

            summary = extract_section(response_text, "== SUMMARY ==", all_markers)
            eligibility = extract_section(response_text, "== ELIGIBILITY ==", all_markers)
            met_text = extract_section(response_text, "== MET CRITERIA ==", all_markers)
            unmet_text = extract_section(response_text, "== UNMET CRITERIA ==", all_markers)
            unclear_text = extract_section(response_text, "== UNCLEAR CRITERIA ==", all_markers)

            return {
                "summary": summary,
                "status": eligibility,
                "met_criteria": parse_criteria_list(met_text),
                "unmet_criteria": parse_criteria_list(unmet_text, has_reasoning=True),
                "unclear_criteria": parse_criteria_list(unclear_text, has_reasoning=True)
            }
        except Exception as e:
            logging.error(f"Error parsing structured text response: {e}\nRaw text was:\n{response_text[:500]}...", exc_info=True)
            return None

    async def run_single_trial_analysis(self, trial_data: Dict[str, Any], patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the eligibility assessment for a single, specific trial.
        """
        logging.info(f"Running single trial analysis for {trial_data.get('id', 'N/A')}...")
        
        trial_data_with_assessment = await self._get_llm_assessment_for_trial(
            patient_context=patient_data, 
            trial_detail=trial_data
        )
        return trial_data_with_assessment

    async def run(self, query: str, patient_context: Optional[Dict[str, Any]] = None, page_state: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Finds relevant clinical trials based on a search query, supporting pagination.
        """
        logging.info(f"Running trial search for query: '{query}' with page_state: '{page_state}'")
        if not self.embedding_model:
            logging.error("Embedding model is not available. Cannot perform vector search.")
            return {"status": "error", "message": "Embedding model not loaded."}

        try:
            # 1. Build a rich query by combining the user's prompt and patient context
            if patient_context:
                # The _build_query_text method is not async, so we call it directly.
                # We pass an empty dict for `entities` as we are not doing separate entity extraction here.
                rich_query_text = self._build_query_text(
                    patient_data=patient_context, 
                    entities={}, 
                    prompt=query
                )
            else:
                rich_query_text = query

            # 2. Create embedding for the rich query text
            logging.info(f"Generating embedding for rich query: '{rich_query_text}'")
            query_embedding = self.embedding_model.encode(rich_query_text).tolist()
            logging.info(f"Generated vector starts with: {query_embedding[:5]}")

            # 3. Perform vector search in AstraDB
            logging.info(f"Performing vector search with limit: {N_VECTOR_SEARCH_RESULTS}.")
            if len(query_embedding) != 384:
                logging.error(f"Query embedding dimension {len(query_embedding)} does not match AstraDB dimension 384")
                return {"status": "error", "message": "Vector dimension mismatch"}
            
            # Prepare find parameters
            find_params = {
                "filter": {},  # Add empty filter to enable pagination
                "sort": {"$vector": query_embedding},
                "limit": N_VECTOR_SEARCH_RESULTS,
                "include_similarity": True
            }
            if page_state:
                find_params["page_state"] = page_state
                
            # Correctly call find with keyword arguments
            similar_trials_cursor = self.trials_collection.find(**find_params)
            
            # Correctly iterate to get data and populate the cursor for metadata
            similar_trials = list(similar_trials_cursor)
            
            # Correctly get the next_page_state after iteration
            next_page_state = similar_trials_cursor._next_page_state
            
            logging.info(f"Found {len(similar_trials)} trials from vector search. Next page state: {next_page_state}")

            # Extract unique trial IDs
            nct_ids = list(set(trial['nct_id'] for trial in similar_trials))

            # 4. Fetch full trial details from SQLite
            trial_details = self._fetch_trial_details(nct_ids)
            logging.info(f"Fetched details for {len(trial_details)} trials from SQLite.")

            return {
                "status": "success",
                "message": f"Found {len(trial_details)} trials.",
                "results": trial_details,
                "next_page_state": next_page_state
            }

        except Exception as e:
            logging.error(f"An error occurred during trial search: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


if __name__ == '__main__':
    import asyncio
    import pprint

    async def main():
        """ Main function for local testing of the agent. """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        agent = ClinicalTrialAgent()

        if not agent.embedding_model or not agent.trials_collection:
            print("Agent initialization failed. Exiting.")
            return

        patient_data = {
            "diagnosis": {"primary": "Melanoma", "stage": "Stage III"},
            "biomarkers": ["BRAF V600E-positive"],
            "prior_treatments": [{"name": "chemotherapy", "status": "completed"}]
        }
        prompt_details = {"prompt": "Find trials for a patient with Stage III Melanoma"}

        print("\n--- Running Full Agent Test ---")
        results = await agent.run(patient_data, prompt_details)
        print(f"Agent returned {len(results.get('results', []))} trials.")
        if results.get('results'):
            print("\n--- Top Trial Result ---")
            pprint.pprint(results['results'][0])

        print("\n--- Running Single Trial Analysis Test ---")
        single_trial_example = {
            "id": "NCT04511011",
            "title": "A Study of Relatlimab Plus Nivolumab in Participants With Melanoma That Has Spread",
            "status": "RECRUITING",
            "phases": '["Phase 3"]',
            "summary": "The purpose of this study is to determine whether the combination of relatlimab plus nivolumab is more effective than nivolumab alone in treating participants with metastatic melanoma.",
            "inclusion_criteria": "Key Inclusion Criteria:\n\n*   Participant must have a histologically confirmed diagnosis of Stage III (unresectable) or Stage IV melanoma...\n*   No prior systemic therapy for unresectable or metastatic melanoma.",
            "exclusion_criteria": "Key Exclusion Criteria:\n\n*   Uveal melanoma.\n*   Active brain metastases or leptomeningeal metastases.\n*   Prior treatment with an anti-PD-1, anti-PD-L1, or anti-CTLA-4 antibody."
        }
        single_result = await agent.run_single_trial_analysis(single_trial_example, patient_data)
        print("\n--- Single Trial Assessment ---")
        pprint.pprint(single_result.get('eligibility'))
        
    asyncio.run(main()) 