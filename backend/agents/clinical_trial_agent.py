"""
Agent responsible for finding relevant clinical trials.
Refactored for Task 9: Use new database architecture with SQLite + AstraDB
"""

import json
import os
import sqlite3
import pprint
import logging
import re
import asyncio
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path

# Remove ChromaDB imports - no longer needed
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

# Import the base class
from backend.core.agent_interface import AgentInterface

# Updated import for new database connections
from backend.utils.database_connections import (
    get_sqlite_connection, 
    get_astradb_connection, 
    close_connection
)

# Action suggester import
from backend.agents.action_suggester import get_action_suggestions_for_trial

# Configuration - load environment variables
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configuration constants
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
N_VECTOR_RESULTS = 15  # Number of results from vector search
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = "gemini-1.5-pro"
DEFAULT_LLM_GENERATION_CONFIG = GenerationConfig(
    temperature=0.2, 
    max_output_tokens=8192
)

# Safety settings for LLM
SAFETY_SETTINGS = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
}

# LLM prompt template for eligibility assessment
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


class ClinicalTrialAgent(AgentInterface):
    """Finds clinical trials relevant to a patient's condition using SQLite + AstraDB architecture."""

    def __init__(self):
        """Initialize the agent with new database connections and embedding model."""
        self.embedding_model = None
        self.llm_client = None
        
        # Initialize Sentence Transformer for embeddings
        try:
            logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logging.info("Embedding model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize SentenceTransformer model: {e}", exc_info=True)

        # Initialize Google Generative AI client
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
        
        logging.info("ClinicalTrialAgent initialized with new database architecture.")

    @property
    def name(self) -> str:
        return "clinical_trial_finder"

    @property
    def description(self) -> str:
        return "Searches for relevant clinical trials based on patient diagnosis, eligibility context, stage, biomarkers, etc. using SQLite and AstraDB databases."

    def _build_query_text(self, context: Dict[str, Any], entities: Dict[str, Any], prompt: str) -> str:
        """Constructs the text to be embedded for searching based on available info."""
        patient_data = context.get("patient_data") or {}
        primary_diagnosis = patient_data.get("diagnosis", {}).get("primary") if patient_data else None
        stage = patient_data.get("diagnosis", {}).get("stage") if patient_data else None
        biomarkers = patient_data.get("biomarkers", []) if patient_data else []
        prior_treatments = patient_data.get("prior_treatments", []) if patient_data else []

        # Use specific entities if available
        condition = entities.get("condition", entities.get("specific_condition"))
        phase = entities.get("trial_phase")
        status = entities.get("recruitment_status")

        # Construct query string
        parts = []
        if condition:
            parts.append(f"Condition: {condition}")
        elif primary_diagnosis:
            parts.append(f"Condition: {primary_diagnosis}")

        if stage: 
            parts.append(f"Stage: {stage}")
        if phase: 
            parts.append(f"Phase: {phase}")
        if status: 
            parts.append(f"Status: {status}")
        if biomarkers: 
            parts.append(f"Biomarkers: {', '.join(biomarkers)}")
        if prior_treatments: 
            parts.append(f"Prior Treatments: {', '.join(pt.get('name', '') for pt in prior_treatments if pt.get('name'))}")

        if parts:
            query_text = ". ".join(parts)
            logging.info(f"Using constructed query text: {query_text}")
            return query_text
        elif prompt:
            logging.info(f"Using original prompt for query text: {prompt}")
            return prompt
        else:
            logging.warning("No suitable query text could be constructed.")
            return ""

    def _vector_search_trials(self, query_text: str, n_results: int = N_VECTOR_RESULTS) -> List[str]:
        """
        Performs vector search in AstraDB to find relevant trial NCT IDs.
        Uses the new database architecture.
        """
        if not self.embedding_model:
            logging.error("Embedding model is not available. Cannot perform vector search.")
            return []
            
        if not query_text:
            logging.warning("Query text is empty. Skipping vector search.")
            return []

        try:
            logging.info(f"Performing vector search in AstraDB with query: '{query_text}'")
            
            # Get AstraDB connection using new database connections
            astra_db = get_astradb_connection()
            collection = astra_db.get_collection("trial_vectors")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query_text).tolist()
            
            # Perform vector similarity search
            results = collection.find(
                sort={"$vector": query_embedding},
                limit=n_results,
                projection={"_id": 1}  # Only need the NCT ID
            )

            documents = list(results)
            if not documents:
                logging.info("AstraDB vector search returned no documents.")
                return []
            
            # Extract NCT IDs (stored in _id field)
            nct_ids = [doc["_id"] for doc in documents if "_id" in doc]
            
            logging.info(f"AstraDB vector search found {len(nct_ids)} trials: {nct_ids[:5]}...")
            return nct_ids

        except Exception as e:
            logging.error(f"AstraDB vector search failed: {e}", exc_info=True)
            return []

    def _fallback_search_trials(self, query: str, limit: int = 10) -> List[str]:
        """
        Fallback search method that searches SQLite directly using new schema.
        """
        try:
            connection = get_sqlite_connection()
            cursor = connection.cursor()
            
            # Search in title, brief_summary, and eligibility_criteria using new schema
            search_query = """
            SELECT nct_id FROM trials 
            WHERE title LIKE ? OR brief_summary LIKE ? OR eligibility_criteria LIKE ?
            LIMIT ?
            """
            
            search_term = f"%{query}%"
            logging.info(f"Fallback SQLite search using query: '{query}' on new schema columns")
            cursor.execute(search_query, (search_term, search_term, search_term, limit))
            results = cursor.fetchall()
            
            nct_ids = [row[0] for row in results]
            logging.info(f"Fallback search found {len(nct_ids)} trials matching '{query}'")
            
            close_connection(connection)
            return nct_ids
            
        except Exception as e:
            logging.error(f"Fallback search failed: {e}", exc_info=True)
            return []

    def _fetch_trial_details(self, nct_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches full trial details from SQLite for given NCT IDs using new schema.
        """
        if not nct_ids:
            return []
            
        try:
            connection = get_sqlite_connection()
            cursor = connection.cursor()
            
            placeholders = ','.join('?' * len(nct_ids))
            # Use new schema column names
            query = f"""
            SELECT nct_id, title, status, phase, study_type, conditions, locations,
                   brief_summary, detailed_description, eligibility_criteria,
                   sponsor, enrollment_count, last_updated_date
            FROM trials 
            WHERE nct_id IN ({placeholders})
            """
            
            cursor.execute(query, nct_ids)
            rows = cursor.fetchall()
            
            # Convert to dictionaries with proper column names
            results = []
            for row in rows:
                trial_dict = {
                    "id": row[0],  # nct_id
                    "nct_id": row[0],  # Keep both for compatibility
                    "title": row[1],
                    "status": row[2],
                    "phase": row[3],
                    "study_type": row[4],
                    "conditions": row[5],
                    "locations": row[6],
                    "brief_summary": row[7],
                    "detailed_description": row[8],
                    "eligibility_criteria": row[9],
                    "sponsor": row[10],
                    "enrollment_count": row[11],
                    "last_updated_date": row[12],
                    # Parse phases for compatibility with existing code
                    "phases": [row[3]] if row[3] else [],
                    # Add inclusion/exclusion criteria split for LLM assessment
                    "inclusion_criteria": self._extract_inclusion_criteria(row[9]),
                    "exclusion_criteria": self._extract_exclusion_criteria(row[9])
                }
                results.append(trial_dict)
            
            close_connection(connection)
            return results
            
        except Exception as e:
            logging.error(f"Error fetching trial details: {e}", exc_info=True)
            return []

    def _extract_inclusion_criteria(self, eligibility_criteria: str) -> str:
        """
        Extract inclusion criteria from the full eligibility criteria text.
        This is a simple implementation - could be enhanced with more sophisticated parsing.
        """
        if not eligibility_criteria:
            return ""
        
        # Look for inclusion/exclusion section markers
        text = eligibility_criteria.lower()
        
        # Find inclusion section
        inclusion_markers = ["inclusion criteria:", "inclusion:", "included:"]
        exclusion_markers = ["exclusion criteria:", "exclusion:", "excluded:"]
        
        inclusion_start = -1
        for marker in inclusion_markers:
            pos = text.find(marker)
            if pos != -1:
                inclusion_start = pos + len(marker)
                break
        
        if inclusion_start == -1:
            # If no explicit inclusion section, return first half
            return eligibility_criteria[:len(eligibility_criteria)//2]
        
        # Find where inclusion section ends (exclusion starts)
        inclusion_end = len(eligibility_criteria)
        for marker in exclusion_markers:
            pos = text.find(marker, inclusion_start)
            if pos != -1:
                inclusion_end = pos
                break
        
        return eligibility_criteria[inclusion_start:inclusion_end].strip()

    def _extract_exclusion_criteria(self, eligibility_criteria: str) -> str:
        """
        Extract exclusion criteria from the full eligibility criteria text.
        """
        if not eligibility_criteria:
            return ""
        
        text = eligibility_criteria.lower()
        exclusion_markers = ["exclusion criteria:", "exclusion:", "excluded:"]
        
        for marker in exclusion_markers:
            pos = text.find(marker)
            if pos != -1:
                return eligibility_criteria[pos + len(marker):].strip()
        
        # If no explicit exclusion section, return second half
        return eligibility_criteria[len(eligibility_criteria)//2:]

    def _create_eligibility_prompt(self, patient_context: Dict[str, Any], trial_title: str, 
                                 trial_status: str, trial_phase: str, inclusion_criteria: Optional[str], 
                                 exclusion_criteria: Optional[str]) -> str:
        """Creates the prompt for LLM eligibility assessment."""
        try:
            patient_profile_json = json.dumps(patient_context, indent=2)
        except TypeError as e:
            logging.error(f"Patient context is not JSON serializable: {e}. Using string representation.")
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
        """Parses the structured plain text response from the LLM."""
        if not response_text:
            logging.warning("LLM structured text response is empty.")
            return None

        try:
            # Define section markers
            markers = {
                "SUMMARY": "== SUMMARY ==",
                "ELIGIBILITY": "== ELIGIBILITY ==",
                "MET": "== MET CRITERIA ==",
                "UNMET": "== UNMET CRITERIA ==",
                "UNCLEAR": "== UNCLEAR CRITERIA =="
            }
            
            def extract_section(text, start_marker, all_markers):
                start_idx = text.find(start_marker)
                if start_idx == -1:
                    return ""
                
                start_idx += len(start_marker)
                
                # Find the start of the next marker
                end_idx = len(text)
                for marker_value in all_markers.values():
                    next_marker_idx = text.find(marker_value, start_idx)
                    if next_marker_idx != -1:
                        end_idx = min(end_idx, next_marker_idx)
                         
                return text[start_idx:end_idx].strip()

            # Extract sections
            summary_text = extract_section(response_text, markers["SUMMARY"], markers)
            eligibility_text = extract_section(response_text, markers["ELIGIBILITY"], markers)
            met_text = extract_section(response_text, markers["MET"], markers)
            unmet_text = extract_section(response_text, markers["UNMET"], markers)
            unclear_text = extract_section(response_text, markers["UNCLEAR"], markers)
            
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
                        trial_snippet = None 
                        reasoning_text = None

                        # Extract trial snippet
                        snippet_pattern = r' - TRIAL_SNIPPET: (["](?:\\.|[^"])*["]|\'(?:\\.|[^\'])*\')'
                        snippet_match = re.search(snippet_pattern, content_after_bullet)

                        if snippet_match:
                            trial_snippet_with_quotes = snippet_match.group(1)
                            trial_snippet = trial_snippet_with_quotes.strip("\'\"")
                            
                            criterion_text = content_after_bullet[:snippet_match.start()].strip()
                            remaining_text = content_after_bullet[snippet_match.end():].strip()
                            
                            if has_reasoning and remaining_text.startswith('- Reasoning:'):
                                reasoning_text = remaining_text[len('- Reasoning:'):].strip()

                        items.append({
                            "criterion": criterion_text,
                            "trial_snippet": trial_snippet,
                            "reasoning": reasoning_text
                        })
                return items

            # Parse criteria lists
            met_criteria = parse_criteria_list(met_text, has_reasoning=False)
            unmet_criteria = parse_criteria_list(unmet_text, has_reasoning=True)
            unclear_criteria = parse_criteria_list(unclear_text, has_reasoning=True)
            
            # Construct result dictionary
            result_dict = {
                "patient_specific_summary": summary_text,
                "eligibility_assessment": {
                    "eligibility_summary": eligibility_text,
                    "met_criteria": met_criteria,
                    "unmet_criteria": unmet_criteria,
                    "unclear_criteria": unclear_criteria
                }
            }
            
            return result_dict

        except Exception as e:
            logging.error(f"Error parsing structured text response: {e}", exc_info=True)
            return None

    async def _get_llm_assessment_for_trial(self, patient_context: Dict[str, Any], 
                                          trial_detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generates LLM assessment for a single trial."""
        nct_id = trial_detail.get("id", "UNKNOWN_ID")
        logging.info(f"Starting LLM assessment for trial {nct_id}...")
        
        inclusion_criteria = trial_detail.get('inclusion_criteria')
        exclusion_criteria = trial_detail.get('exclusion_criteria')

        if not inclusion_criteria and not exclusion_criteria:
            logging.warning(f"No criteria text found for trial {nct_id}, skipping LLM assessment.")
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
                trial_detail.get('title', 'N/A'), 
                trial_detail.get('status', 'N/A'), 
                trial_detail.get('phase', 'N/A'), 
                inclusion_criteria, 
                exclusion_criteria
            ) 
            
            response = await asyncio.to_thread(
                self.llm_client.generate_content,
                prompt,
                generation_config=DEFAULT_LLM_GENERATION_CONFIG, 
                safety_settings=SAFETY_SETTINGS
            )
            
            # Extract response text
            raw_response_text = ""
            try: 
                if response.parts:
                    raw_response_text = response.parts[0].text
                else:
                    raw_response_text = response.text
            except Exception as e:
                logging.warning(f"Could not access response parts/text for {nct_id}: {e}")
                try:
                    raw_response_text = response.text 
                except:
                    raw_response_text = "Error retrieving response text."
            
            logging.debug(f"Raw LLM response for {nct_id}:\n{raw_response_text}")

            # Parse structured text response
            parsed_assessment_dict = self._parse_structured_text_response(raw_response_text)

            if parsed_assessment_dict:
                logging.info(f"Successfully parsed assessment for trial {nct_id}.")
                return {"llm_eligibility_analysis": parsed_assessment_dict} 
            else:
                logging.warning(f"Failed to parse assessment for trial {nct_id}.")
                return { 
                    "llm_eligibility_analysis": None,
                    "overall_assessment": "Assessment Failed (Text Parsing Error)",
                    "narrative_summary": "The AI assessment could not be processed."
                }

        except Exception as e:
            logging.error(f"Error during LLM assessment for trial {nct_id}: {e}", exc_info=True)
            return { 
                "llm_eligibility_analysis": None,
                "overall_assessment": "Assessment Failed (API Error)",
                "narrative_summary": f"An error occurred communicating with the AI: {e}"
            }

    async def run_single_trial_analysis(self, trial_data: Dict[str, Any], 
                                      patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline for a single trial: LLM assessment and action suggestions.
        """
        nct_id = trial_data.get("id", "UNKNOWN_ID")
        logging.info(f"Running single trial analysis for {nct_id}")
        
        # Get LLM Assessment
        llm_assessment_result = await self._get_llm_assessment_for_trial(patient_data, trial_data)
        
        # Get Action Suggestions
        parsed_analysis = {}
        if llm_assessment_result and llm_assessment_result.get("llm_eligibility_analysis"):
            parsed_analysis = llm_assessment_result["llm_eligibility_analysis"]

        eligibility_assessment = parsed_analysis.get("eligibility_assessment", {})
        action_suggestions = get_action_suggestions_for_trial(
            eligibility_assessment=eligibility_assessment,
            patient_context=patient_data
        )
        
        # Format for frontend
        llm_assessment_for_frontend = {
            "summary": parsed_analysis.get("patient_specific_summary", "Summary not available."),
            "eligibility_status": eligibility_assessment.get("eligibility_summary", "Not Assessed"),
            "met_criteria": eligibility_assessment.get("met_criteria", []),
            "unmet_criteria": eligibility_assessment.get("unmet_criteria", []),
            "unclear_criteria": eligibility_assessment.get("unclear_criteria", [])
        }

        # Format final output
        phases_list = trial_data.get("phases", [])
        phase_display = phases_list[0] if phases_list else trial_data.get("phase", "N/A")

        final_trial_output = {
            "nct_id": trial_data.get("id"),
            "title": trial_data.get("title"),
            "status": trial_data.get("status"),
            "phase": phase_display,
            "source_url": f"https://clinicaltrials.gov/ct2/show/{trial_data.get('id')}",
            "llm_assessment": llm_assessment_for_frontend,
            "action_suggestions": action_suggestions
        }

        return final_trial_output

    async def run(self, patient_data: Dict[str, Any] = None, 
                 prompt_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for the agent using new database architecture.
        """
        logging.info("ClinicalTrialAgent started with new database architecture.")
        
        # Extract query from prompt_details
        query = ""
        entities = {}
        if prompt_details:
            query = prompt_details.get("query", "")
            entities = prompt_details.get("entities", {})
        
        if not query:
            return {"status": "error", "message": "No search query provided."}

        try:
            # Build search query text
            context = {"patient_data": patient_data} if patient_data else {}
            query_text = self._build_query_text(context, entities, query)
            
            # Perform vector search
            nct_ids = self._vector_search_trials(query_text=query_text)
            
            # Use fallback search if vector search returns no results
            if not nct_ids:
                logging.warning("Vector search returned no results. Attempting fallback search.")
                nct_ids = self._fallback_search_trials(query=query, limit=10)
            
            if not nct_ids:
                return {
                    "status": "success", 
                    "message": "No trials found matching your query.", 
                    "found_trials": []
                }

            # Fetch trial details
            found_trials_details = self._fetch_trial_details(nct_ids)

            if not found_trials_details:
                return {
                    "status": "success", 
                    "message": f"Could not retrieve details for found trial IDs: {nct_ids}", 
                    "found_trials": []
                }
            
            # Perform LLM Assessment if patient context is provided
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
                # Return trials without assessment
                logging.info("No patient context provided. Returning trial details without assessment.")
                return {
                    "status": "success", 
                    "message": f"Found {len(found_trials_details)} trials.",
                    "found_trials": found_trials_details
                }

        except Exception as e:
            logging.error(f"Error in ClinicalTrialAgent.run: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"An error occurred while searching for trials: {str(e)}"
            }