import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import json
import google.generativeai as genai
import os
import asyncio

# Attempt to import the interface, handle if not found for now
try:
    from ..core.agent_interface import AgentInterface
except ImportError:
    logging.warning("AgentInterface not found. Using dummy class.")
    class AgentInterface:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description
        async def run(self, **kwargs) -> Dict[str, Any]:
            raise NotImplementedError

# --- Search Logic Configuration ---

# Mock knowledge base (can be expanded or moved to config)
KNOWN_PGP_INHIBITORS = ['nelfinavir', 'indinavir', 'saquinavir', 'ritonavir', 'ketoconazole', 'itraconazole']
KNOWN_AZOLE_ANTIFUNGALS = ['itraconazole', 'ketoconazole', 'voriconazole', 'fluconazole', 'posaconazole']

# Define Search Targets for Internal Data Lookup
SEARCH_TARGETS = [
    {
        "id": "ecog_ps", # Target identifier
        "criterion_patterns": [r"ecog", r"performance status", r"karnofsky", r"kps"], # Regex patterns to match criterion text
        "search_fields": ["notes"], # Where to look in patient_data
        "search_type": "keyword_sentence", # How to search (e.g., find sentences with keywords)
        "keywords": ["ecog", "ps", "performance status", "kps", "karnofsky", "ambulatory", "bedridden", r"ecog\s*\d", r"kps\s*\d+"] # Keywords/patterns to find *within* the field
    },
    {
        "id": "platelet",
        "criterion_patterns": [r"platelet", r"plt", r"thrombocyte"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component", # Special type for labs
        "lab_test_keywords": ["platelet", "plt"] # Keywords to match lab component 'test' name
    },
    {
        "id": "anc",
        "criterion_patterns": [r"anc", r"absolute neutrophil"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["anc", "absolute neutrophil", "neutrophils abs"]
    },
    {
        "id": "hemoglobin",
        "criterion_patterns": [r"hemoglobin", r"hgb", r"hb"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["hemoglobin", "hgb", "hb"]
    },
    {
        "id": "creatinine",
        "criterion_patterns": [r"creatinine", r"crcl", r"renal function", r"kidney function"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["creatinine", "crcl", "creat"]
    },
    {
        "id": "bilirubin",
        "criterion_patterns": [r"bilirubin", r"bili"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["bilirubin", "bili", "total bilirubin"]
    },
    {
        "id": "liver_enzymes",
        "criterion_patterns": [r"alt", r"sgpt", r"ast", r"sgot", r"liver function", r"hepatic function"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["alt", "sgpt", "ast", "sgot", "alanine aminotransferase", "aspartate aminotransferase"]
    },
    {
        "id": "specific_mutation",
        "criterion_patterns": [
            # More specific patterns first
            r"(?:(?:presence|absence)\s+of\s+)?(?:activating|pathogenic|deleterious)?\s+([a-zA-Z0-9]+)\s+(?:mutation|variant|alteration)", # e.g., "KRAS mutation", "activating BRAF mutation"
            r"([a-zA-Z0-9]+)\s+(?:positive|negative|mutated|wild-type|wt)" # e.g., "EGFR positive", "TP53 mutated"
        ],
        "search_fields": ["mutations"],
        "search_type": "mutation_lookup", # Special type for mutations list
        "capture_group": 1 # Which regex group captures the gene name
    },
    {
        "id": "weight_loss_history",
        "criterion_patterns": [
            r"weight loss", r"lost weight", r"gained weight", 
            r"unintentional weight loss", r"significant weight change",
            r"cachexia", r"nutritional status", r"bmi change"
        ],
        "search_fields": ["notes", "diagnosis"], # Check notes and diagnosis fields
        "search_type": "keyword_sentence",
        "keywords": [
            "weight loss", "lost", "gained", "lbs", "kg", "pounds", "kilograms",
            "cachectic", "malnourished", "bmi", "unintentional change"
        ]
    },
    {
        "id": "pgp_inhibitor",
        "criterion_patterns": [r"p-gp inhibitor", r"p-glycoprotein inhibitor"],
        "search_fields": ["currentMedications"],
        "search_type": "medication_check", # Special type
        "known_list": KNOWN_PGP_INHIBITORS # Use the list defined earlier
    },
    {
        "id": "azole_antifungal",
        "criterion_patterns": [r"azole antifungal"],
        "search_fields": ["currentMedications"],
        "search_type": "medication_check",
        "known_list": KNOWN_AZOLE_ANTIFUNGALS
    },
    # TODO: Add more targets: VTE history, prior therapy, transplant history etc.
]

# --- End Search Logic Configuration ---

# --- Constants (Should match ClinicalTrialAgent or be centralized) ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-latest")
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# Add safety settings and default generation config if needed, mirroring ClinicalTrialAgent
SAFETY_SETTINGS = [{ "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" }, { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" }, { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" }, { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE" }]
DEFAULT_LLM_GENERATION_CONFIG = {
    "temperature": 0.2, 
    "top_p": 0.95, 
    "top_k": 40,
    "max_output_tokens": 4096, 
    # "response_mime_type": "text/plain", # Already default 
}
# --- End Constants ---

# Attempt to import the specific agent, handle if not found
try:
    from .genomic_analyst_agent import GenomicAnalystAgent, GenomicAnalysisResult
except ImportError:
    logging.warning("GenomicAnalystAgent not found. Genomic analysis features will be unavailable.")
    GenomicAnalystAgent = None # Set to None if import fails

class EligibilityDeepDiveAgent(AgentInterface):
    """
    Agent performs a deeper analysis of unmet/unclear eligibility criteria
    by leveraging an LLM to check against available patient data.
    """
    # --- Define Name and Description as Properties to satisfy ABC --- 
    @property
    def name(self) -> str:
        return "EligibilityDeepDiveAgent"

    @property
    def description(self) -> str:
        return "Performs a detailed review of specific eligibility criteria against patient data using an LLM."
    # --- End Properties --- 
    
    def _parse_criteria_from_trial_data(self, trial_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses inclusion/exclusion criteria strings from trial_data into a structured list."""
        parsed_criteria = []
        
        # --- Handle Inclusion Criteria ---
        inclusion_text = trial_data.get("inclusion_criteria", "")
        if inclusion_text and isinstance(inclusion_text, str):
            # Split by newline and filter out empty lines or simple headers
            items = [item.strip() for item in inclusion_text.split('\n') if item.strip()]
            for item in items:
                # Remove leading bullets/asterisks for cleaner text
                clean_item = re.sub(r"^\s*[\*\-•]\s*", "", item)
                if clean_item and len(clean_item) > 10: # Basic filter for meaningful criteria
                    parsed_criteria.append({
                        "criterion": clean_item,
                        "type": "inclusion",
                        "reasoning": "N/A - Full Deep Dive" # Add context
                    })

        # --- Handle Exclusion Criteria ---
        exclusion_text = trial_data.get("exclusion_criteria", "")
        if exclusion_text and isinstance(exclusion_text, str):
            items = [item.strip() for item in exclusion_text.split('\n') if item.strip()]
            for item in items:
                clean_item = re.sub(r"^\s*[\*\-•]\s*", "", item)
                if clean_item and len(clean_item) > 10:
                    parsed_criteria.append({
                        "criterion": clean_item,
                        "type": "exclusion",
                        "reasoning": "N/A - Full Deep Dive"
                    })
        
        logging.info(f"[{self.name}] Parsed {len(parsed_criteria)} criteria for full deep dive from trial_data.")
        return parsed_criteria

    def __init__(self):
        # Call super().__init__() - No need to set name/desc here now
        super().__init__() 
        
        # Initialize only the LLM client here
        self.llm_client = None
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                self.llm_client = genai.GenerativeModel(
                    LLM_MODEL_NAME,
                    # system_instruction="You are an expert clinical research assistant..." # Add if needed
                )
                # Use self.name (the property) in logging
                logging.info(f"[{self.name}] Google GenAI client initialized successfully with model {LLM_MODEL_NAME}.") 
            except Exception as e:
                 # Use self.name (the property) in logging
                logging.error(f"[{self.name}] Failed to initialize Google GenAI client: {e}", exc_info=True)
        else:
             # Use self.name (the property) in logging
            logging.error(f"[{self.name}] GOOGLE_API_KEY environment variable not set. LLM features will be disabled.")

    async def _run_batch_analysis(self, criteria_list: List[Dict[str, Any]], patient_data_snippet: Dict[str, Any], trial_id: str) -> List[Dict[str, Any]]:
        """
        Constructs a prompt, runs a single batch LLM call, and parses the response.
        """
        # 1. Construct the prompt
        formatted_criteria = ""
        for i, criterion_obj in enumerate(criteria_list):
            formatted_criteria += f"{i+1}. {criterion_obj['criterion']}\n"

        prompt = (
            "You are an expert clinical research assistant performing a deep dive analysis on a list of eligibility criteria for a clinical trial. "
            f"Your task is to determine if a patient meets each criterion based *only* on the provided patient data snippet.\n\n"
            "## CONTEXT\n"
            f"**Trial ID:** {trial_id}\n\n"
            f"## PATIENT DATA\n"
            "Review the following patient data carefully. This is the ONLY information you have.\n"
            f"```json\n{json.dumps(patient_data_snippet, indent=2, default=str)}\n```\n\n"
            "## CRITERIA TO ANALYZE\n"
            "Analyze each of the following criteria from the provided list:\n"
            f"```\n{formatted_criteria}\n```\n\n"
            "## YOUR TASK\n"
            "1.  **Go through each criterion one by one.**\n"
            "2.  **Make a definitive decision for each:** Does the patient meet the criterion, not meet it, or is it impossible to determine from the given data?\n"
            "3.  **Provide direct evidence:** Quote or reference the specific lab value, mutation, note, or medication from the patient data that justifies your decision. If no data is available, state that explicitly.\n"
            "4.  **Format your response as a single JSON array.** Each object in the array must correspond to a criterion from the list and have three keys: `criterion`, `status`, and `evidence`.\n\n"
            "   -   `\"criterion\"`: The full, original text of the criterion you are analyzing.\n"
            "   -   `\"status\"`: Must be one of three exact string values: `MET`, `NOT_MET`, or `UNCLEAR`.\n"
            "   -   `\"evidence\"`: A string explaining your reasoning, citing specific data points from the patient record.\n\n"
            "**Analysis Output (JSON Array):**\n"
        )

        # 2. Make the single API call
        try:
            logging.info(f"[{self.name}:{trial_id}] Sending single batch request to LLM for {len(criteria_list)} criteria.")
            response = await self.llm_client.generate_content_async(
                prompt,
                generation_config=DEFAULT_LLM_GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS
            )
            response_text = response.text
        except Exception as e:
            logging.error(f"[{self.name}:{trial_id}] Batched LLM call failed: {e}", exc_info=True)
            # Return an error object for each criterion
            return [
                {"criterion": c.get("criterion"), "status": "ERROR_ANALYSIS_FAILED", "evidence": f"LLM call failed: {e}", "analysis_source": "System Error"}
                for c in criteria_list
            ]

        # 3. Parse the response
        try:
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not match:
                raise json.JSONDecodeError("No JSON array found in response.", response_text, 0)
            
            json_str = match.group(0)
            parsed_results = json.loads(json_str)

            if not isinstance(parsed_results, list):
                raise TypeError("LLM response was not a list.")

            # Map results back to original criteria to ensure order and completeness
            final_results = []
            result_map = {res.get("criterion"): res for res in parsed_results}
            
            for original_criterion in criteria_list:
                oc_text = original_criterion["criterion"]
                # Find the result from the LLM, case-insensitive and ignoring whitespace
                found_res = None
                for llm_criterion_text, llm_result in result_map.items():
                    if llm_criterion_text and oc_text.strip().lower() == llm_criterion_text.strip().lower():
                        found_res = llm_result
                        break
                
                if found_res:
                    final_results.append(found_res)
                else:
                    # LLM didn't return a result for this specific criterion
                    final_results.append({
                        "criterion": oc_text,
                        "status": "ERROR_MISSING_FROM_LLM_RESPONSE",
                        "evidence": "The LLM did not provide an analysis for this specific criterion in its batched response."
                    })
            
            return final_results

        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Failed to decode or validate JSON array from LLM response: {e}", exc_info=True)
            return [
                {"criterion": c.get("criterion"), "status": "ERROR_PARSING_FAILED", "evidence": f"Failed to parse batched LLM response: {e}. Raw response: {response_text[:200]}...", "analysis_source": "System Error"}
                for c in criteria_list
            ]

    # --- MAIN AGENT EXECUTION METHOD ---
    async def run(self, unmet_criteria: List[Dict[str, Any]], unclear_criteria: List[Dict[str, Any]], patient_data: Dict[str, Any], trial_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyzes a list of unmet and unclear criteria against patient data.
        Enhances the analysis by considering any original reasoning provided.
        """
        trial_id = trial_data.get("id", "UNKNOWN_TRIAL") # Get trial_id for logging
        logging.info(f"[{self.name}:{trial_id}] Starting deep dive. Unmet: {len(unmet_criteria)}, Unclear: {len(unclear_criteria)} criteria.")

        if not self.llm_client:
            logging.error(f"[{self.name}:{trial_id}] LLM client not initialized. Cannot perform deep dive.")
            # Return a structure indicating an error, perhaps mirroring the expected success output
            return {
                "trial_id": trial_id,
                "summary": "Deep dive could not be performed due to LLM initialization error.",
                "analyzed_criteria": [],
                "remaining_gaps": [], # No gaps identified if agent can't run
                "strategic_next_steps": [],
                "error": "LLM client not initialized"
            }

        # --- Prepare Patient Data Snippet (Same as before) ---
        patient_data_snippet = {
            "patientId": patient_data.get("id"),
            "diagnosis": patient_data.get("diagnosis"),
            "treatments": patient_data.get("treatments", [])[:5], # Limit for brevity
            "currentMedications": patient_data.get("currentMedications", [])[:5],
            "labs": patient_data.get("labs", [])[:3], # Limit recent labs
            "notes": [note for note in patient_data.get("notes", [])[:2]], # Only text of first 2 notes
            "mutations": patient_data.get("mutations", []) # Include all mutations for this patient
        }
        patient_data_snippet = {k: v for k, v in patient_data_snippet.items() if v}
        # --- End Snippet Prep ---

        all_criteria_to_analyze = unmet_criteria + unclear_criteria
        is_full_dive = False

        # --- NEW: Full Deep Dive Fallback Logic ---
        if not all_criteria_to_analyze:
            logging.info(f"[{self.name}:{trial_id}] No specific criteria provided. Attempting a full deep dive from trial data.")
            all_criteria_to_analyze = self._parse_criteria_from_trial_data(trial_data)
            is_full_dive = True
            if not all_criteria_to_analyze:
                 logging.warning(f"[{self.name}:{trial_id}] Could not parse any criteria from trial_data for a full deep dive.")
        
        analyzed_results = []

        if not all_criteria_to_analyze:
            logging.info(f"[{self.name}:{trial_id}] No criteria provided for deep dive.")
            return {
                "trial_id": trial_id,
                "summary": "No criteria were submitted for deep dive analysis.",
                "analyzed_criteria": [],
                "remaining_gaps": [],
                "strategic_next_steps": []
            }

        # --- BATCHED LLM ANALYSIS ---
        if all_criteria_to_analyze:
             analyzed_results = await self._run_batch_analysis(
                criteria_list=all_criteria_to_analyze,
                patient_data_snippet=patient_data_snippet,
                    trial_id=trial_id
             )
        # --- END BATCHED LLM ANALYSIS ---

        # --- Task 5.1.2: Internal Data Search Logic --- 
        internal_search_findings = {}
        processed_gap_indices = set() # To avoid searching the same gap multiple times if criteria overlap targets
        logging.info(f"[{self.name}:{trial_id}] Starting internal data search for {len(analyzed_results)} analyzed criteria...")

        for i, result in enumerate(analyzed_results):
            if i in processed_gap_indices:
                continue

            criterion_text = result.get("criterion")
            if not criterion_text or not isinstance(criterion_text, str):
                continue # Skip if no valid criterion text
                
            findings_for_this_criterion = []
            target_matched = None

            for target in SEARCH_TARGETS:
                match_found = False
                captured_value = None
                try:
                    patterns = target.get("criterion_patterns", [])
                    if not patterns: continue
                    
                    # Combine patterns into a single regex for matching this target
                    # Ensure patterns are strings before joining
                    string_patterns = [p for p in patterns if isinstance(p, str)]
                    if not string_patterns: continue
                    target_regex = re.compile(r"|".join(string_patterns), re.IGNORECASE)
                    
                    match = target_regex.search(criterion_text)
                    if match:
                        match_found = True
                        target_matched = target # Store the matched target config
                        target_id = target.get("id", "UNKNOWN") # Evaluate outside f-string
                        capture_group_index = target.get("capture_group")
                        if capture_group_index is not None and len(match.groups()) >= capture_group_index:
                            captured_value = match.group(capture_group_index)
                            logging.debug(f"[{self.name}:{trial_id}] Criterion '{criterion_text[:50]}...' matched target '{target_id}'. Captured: '{captured_value}'")
                        else:
                            logging.debug(f"[{self.name}:{trial_id}] Criterion '{criterion_text[:50]}...' matched target '{target_id}'. No capture group needed/found.")
                            
                except re.error as re_err:
                    target_id_err = target.get('id', 'UNKNOWN') # Evaluate outside f-string
                    logging.error(f"[{self.name}:{trial_id}] Invalid regex in target '{target_id_err}': {patterns}. Error: {re_err}")
                    continue # Skip this target if regex fails
                except Exception as e:
                    target_id_ex = target.get('id', 'UNKNOWN') # Evaluate outside f-string
                    logging.error(f"[{self.name}:{trial_id}] Error matching criterion to target '{target_id_ex}': {e}")
                    continue

                if match_found and target_matched:
                    # Perform the search based on target config
                    search_type = target_matched.get("search_type")
                    search_fields = target_matched.get("search_fields", [])
                    search_findings_for_target = []

                    for field_name in search_fields:
                        field_data = patient_data.get(field_name) # Use the FULL patient_data here
                        if field_data is None: 
                            target_id_nf = target_matched.get("id", "UNKNOWN") # Evaluate outside f-string
                            logging.debug(f"[{self.name}:{trial_id}] Search field '{field_name}' not found in patient data for target '{target_id_nf}'")
                            continue # Skip if the necessary field doesn't exist in patient_data
                        
                        try:
                            if search_type == "lab_component":
                                lab_keywords = target_matched.get("lab_test_keywords", [])
                                search_findings_for_target.extend(self._search_lab_component(field_data, lab_keywords))
                            elif search_type == "keyword_sentence":
                                keywords = target_matched.get("keywords", [])
                                search_findings_for_target.extend(self._search_keyword_sentence(field_data, keywords))
                            elif search_type == "mutation_lookup":
                                gene_symbol_to_find = captured_value # Use the captured gene symbol
                                if gene_symbol_to_find:
                                     search_findings_for_target.extend(self._search_mutation_list(field_data, gene_symbol_to_find))
                                else:
                                    target_id_unk = target_matched.get("id", "UNKNOWN") # Evaluate outside f-string
                                    logging.warning(f"[{self.name}:{trial_id}] Mutation lookup target matched for criterion '{criterion_text[:50]}...' but no gene symbol was captured.")
                            elif search_type == "medication_check":
                                known_list = target_matched.get("known_list", [])
                                search_findings_for_target.extend(self._search_medication_check(field_data, known_list))
                            else:
                                target_id_unk = target_matched.get("id", "UNKNOWN") # Evaluate outside f-string
                                logging.warning(f"[{self.name}:{trial_id}] Unknown search_type '{search_type}' for target '{target_id_unk}'")
                        except Exception as search_ex:
                            target_id_s_err = target_matched.get("id", "UNKNOWN") # Evaluate outside f-string
                            logging.error(f"[{self.name}:{trial_id}] Error during internal search type '{search_type}' for target '{target_id_s_err}': {search_ex}", exc_info=True)
                    
                    if search_findings_for_target:
                        findings_for_this_criterion.extend(search_findings_for_target)
                        
                    # Mark this criterion as processed by this target and break from target loop
                    processed_gap_indices.add(i)
                    break 
            
            # Store findings associated with the criterion text
            if findings_for_this_criterion:
                internal_search_findings[criterion_text] = findings_for_this_criterion
                logging.info(f"[{self.name}:{trial_id}] Internal search found {len(findings_for_this_criterion)} potential context items for criterion: '{criterion_text[:50]}...'")
            elif target_matched: # Search was performed but nothing found
                 internal_search_findings[criterion_text] = [] # Indicate search happened but found nothing
                 target_id_nf2 = target_matched.get("id", "UNKNOWN") # Evaluate outside f-string
                 logging.info(f"[{self.name}:{trial_id}] Internal search performed for criterion '{criterion_text[:50]}...' based on target '{target_id_nf2}', but found no specific context.")
                 
        logging.info(f"[{self.name}:{trial_id}] Internal search completed for {len(processed_gap_indices)} unique criteria.")
        # --- End Task 5.1.2 --- 

        # --- Post-Analysis: Combine results and findings ---
        clarified_items = []
        remaining_gaps = [] # Items still UNCLEAR or now NOT_MET based on deep dive
        
        for result in analyzed_results:
            criterion_text = result.get("criterion")
            # Attach internal search results to the result object for easier processing later
            result["internal_search_findings"] = internal_search_findings.get(criterion_text) # Will be None if search wasn't applicable/run
            
            if result.get("status") == "MET":
                clarified_items.append(result)
            else: # NOT_MET, UNCLEAR, or ERROR from deep dive
                remaining_gaps.append(result)
        # --- End Post-Analysis Combination ---

        # --- Generate Summary (Can be enhanced later) ---
        if is_full_dive:
            summary_text = f"Full deep dive completed for {len(analyzed_results)} criteria. "
        else:
            summary_text = f"Deep dive completed for {len(analyzed_results)} criteria. "
        summary_text += f"{len(clarified_items)} criteria were potentially clarified as MET. "
        summary_text += f"{len(remaining_gaps)} criteria remain as NOT_MET, UNCLEAR, or encountered errors."
        # --- End Summary --- 

        # --- Task 5.1.4: Generate Strategic Next Steps ---
        strategic_next_steps = []
        if remaining_gaps and self.llm_client:
            logging.info(f"[{self.name}:{trial_id}] Generating strategic next steps based on {len(remaining_gaps)} remaining gaps and internal search results...")
            try:
                # Prepare context for the prompt, including internal search outcomes
                prompt_context_gaps = []
                for gap in remaining_gaps:
                    criterion_text = gap.get("criterion")
                    # Access findings attached earlier during post-analysis combination
                    findings = gap.get("internal_search_findings")
                    search_outcome = "Not Searched Internally"
                    if findings is not None: # Search was attempted
                        search_outcome = f"Internal Search Found {len(findings)} item(s)" if findings else "Internal Search Found Nothing"

                    prompt_context_gaps.append({
                        "criterion": criterion_text,
                        "status_after_llm_dive": gap.get("status"),
                        "reason_unresolved": gap.get("evidence"),
                        "internal_search_outcome": search_outcome,
                        # Optionally include a snippet of findings if present and not too long
                        "internal_finding_snippet": findings[0].get("context") if findings else None
                    })

                # Construct the refined prompt
                next_steps_prompt = f"""
                Context: You are an expert clinical research coordinator AI assistant. A deep dive analysis was performed on clinical trial eligibility criteria. Some criteria remain unresolved (NOT_MET, UNCLEAR, or ERROR). An internal search within available patient data (labs, notes, mutations) was also performed for some criteria.

                Remaining Unresolved Criteria & Internal Search Outcomes:
                ```json
                {json.dumps(prompt_context_gaps, indent=2)}
                ```

                Task: Generate a **prioritized and concise list (max 5-7 actions)** of the MOST IMPORTANT concrete next steps to resolve the remaining ambiguities. Focus on actions that address critical eligibility factors first (like key labs, performance status, major exclusions, core genomic tests, measurable disease).

                Instructions:
                - **Prioritize:** Address gaps related to critical factors first.
                - **Be Specific:** Suggest concrete actions (ORDER_LABS [specify], SCHEDULE_ASSESSMENT [specify], REVIEW_INTERNAL_FINDING [point to source/context], CLARIFY_WITH_PATIENT [specific topic], VERIFY_MEDICATION [specific class], CHECK_EXTERNAL_RECORD [specify system/source], OTHER [specify]).
                - **Use Internal Findings:** If `internal_search_outcome` indicates items were found, the primary action should be `REVIEW_INTERNAL_FINDING`. Include the `internal_finding_snippet` in the rationale or details if provided.
                - **Target External Gaps:** If `internal_search_outcome` is "Internal Search Found Nothing" or "Not Searched Internally", suggest the most direct *external* action needed (e.g., ORDER_LABS, SCHEDULE_ASSESSMENT, CHECK_EXTERNAL_RECORD).
                - **Combine Actions:** Group similar actions (e.g., order multiple labs in one step).
                - **Limit Quantity:** Provide NO MORE than 7 distinct action items in total, focusing on the highest impact steps.
                - **Output Format:** Respond ONLY with a valid JSON list of objects. Each object must have keys: "action_type" (String), "description" (String), "rationale" (String), "details" (Optional[String]).

                Example Action Object (Internal Finding):
                {{ "action_type": "REVIEW_INTERNAL_FINDING", "description": "Review ECOG mention in recent Oncology note", "rationale": "Addresses critical Performance Status gap. Internal search found potential context.", "details": "Finding: 'Patient presents... ECOG 1 today.'" }}
                
                Example Action Object (External Needed):
                {{ "action_type": "ORDER_LABS", "description": "Order CBC w/ Diff & CMP", "rationale": "Addresses critical gaps in ANC, Platelets, Hemoglobin, etc. Internal search found no recent values.", "details": "CBC with differential, Comprehensive Metabolic Panel" }}

                Generate the prioritized JSON list (max 7 items) now:
                ```json
                [
                    // Generate JSON objects here 
                ]
                ```
                """
                
                # --- Make LLM call (using existing safe extraction logic) ---
                logging.debug(f"[{self.name}:{trial_id}] Sending refined prompt to LLM for strategic next steps...")
                # Assume generate_content_async is preferred/available
                next_steps_response = await self.llm_client.generate_content_async(
                    next_steps_prompt,
                    generation_config=DEFAULT_LLM_GENERATION_CONFIG,
                    safety_settings=SAFETY_SETTINGS
                )
                
                raw_next_steps_text = ""
                try:
                    # Standard way to get text from Gemini response
                    if next_steps_response.parts:
                        raw_next_steps_text = next_steps_response.parts[0].text
                    elif hasattr(next_steps_response, 'text'):
                        raw_next_steps_text = next_steps_response.text # Fallback for simpler text responses
                    else: 
                        logging.warning(f"[{self.name}:{trial_id}] Next steps LLM Response structure unexpected or no text content found. Blocked? Resp: {next_steps_response}")
                        raw_next_steps_text = "Error: LLM response structure invalid or missing text."
                except AttributeError:
                     logging.warning(f"[{self.name}:{trial_id}] Next steps LLM Response object missing .text/.parts attribute. Blocked? Resp: {next_steps_response}")
                     raw_next_steps_text = "Error: LLM response blocked or attribute missing."
                except Exception as text_ex:
                    logging.error(f"[{self.name}:{trial_id}] Error extracting text from next steps LLM response: {text_ex}", exc_info=True)
                    raw_next_steps_text = f"Error extracting text: {text_ex}"

                logging.debug(f"[{self.name}:{trial_id}] Raw next steps response: {raw_next_steps_text[:500]}...")

                # --- Parse JSON response (using existing safe parsing logic) ---
                try:
                    raw_next_steps_text = raw_next_steps_text.strip() # Strip raw text first
                    logging.debug(f"[{self.name}:{trial_id}] Raw next steps response (stripped, first 600 chars): {raw_next_steps_text[:600]}")

                    json_string = None
                    # Try to extract from markdown block first
                    if raw_next_steps_text.startswith("```json") and raw_next_steps_text.endswith("```"):
                        json_string = raw_next_steps_text[len("```json"): -len("```")].strip()
                        logging.info(f"[{self.name}:{trial_id}] Extracted from markdown block by start/end check.")
                    else:
                        # Fallback regex if start/end check fails (e.g. extra text after closing ```)
                        match = re.search(r"```json\\s*(.*?)\\s*```", raw_next_steps_text, re.DOTALL)
                        if match:
                            json_string = match.group(1).strip()
                            logging.info(f"[{self.name}:{trial_id}] Extracted from markdown block by regex.")
                        else:
                            # If no markdown, assume the whole string might be JSON (e.g., LLM forgot markdown)
                            logging.warning(f"[{self.name}:{trial_id}] No JSON markdown block detected. Assuming raw response is JSON.")
                            json_string = raw_next_steps_text # Already stripped

                    if not json_string:
                        logging.error(f"[{self.name}:{trial_id}] JSON string for next steps is empty after extraction attempts. Raw text: {raw_next_steps_text[:500]}")
                        raise json.JSONDecodeError("Extracted JSON string is empty", raw_next_steps_text, 0)

                    logging.info(f"[{self.name}:{trial_id}] Attempting to parse for next steps (first 500 chars): '''{json_string[:500]}...'''")
                    strategic_next_steps = json.loads(json_string)

                    if not isinstance(strategic_next_steps, list):
                        logging.error(f"[{self.name}:{trial_id}] Parsed JSON for next steps is not a list. Type: {type(strategic_next_steps)}. Content: {json_string[:200]}")
                        raise json.JSONDecodeError("Parsed JSON is not a list as expected by prompt", json_string, 0)

                    logging.info(f"[{self.name}:{trial_id}] Successfully parsed {len(strategic_next_steps)} strategic next steps.")

                except json.JSONDecodeError as json_err:
                    problematic_string = json_string if 'json_string' in locals() and json_string is not None else raw_next_steps_text
                    logging.error(f"[{self.name}:{trial_id}] Failed to parse JSON for strategic next steps: {json_err}. String attempted (first 500 chars): '{problematic_string[:500]}'")
                    strategic_next_steps = [{"action_type": "ERROR", "description": "Failed to parse next steps from LLM.", "rationale": f"JSON Error: {json_err}", "details": raw_next_steps_text[:500]}]
                
            except Exception as next_step_ex:
                logging.error(f"[{self.name}:{trial_id}] Error generating strategic next steps: {next_step_ex}", exc_info=True)
                strategic_next_steps = [{"action_type": "ERROR", "description": "Failed to generate strategic actions.", "rationale": str(next_step_ex), "details": None}]
                
        elif not remaining_gaps:
             strategic_next_steps = [{"action_type": "INFO", "description": "All initially unmet/unclear criteria were clarified.", "rationale": "No further action needed based on deep dive.", "details": None}]
        else: # LLM client not available
             strategic_next_steps = [{"action_type": "ERROR", "description": "LLM client needed to generate strategic actions.", "rationale": "LLM client not initialized.", "details": None}]
        # --- End Task 5.1.4 ---

        logging.info(f"[{self.name}:{trial_id}] Deep dive finished. Summary: {summary_text}")
        return {
            "trial_id": trial_id,
            "summary": summary_text,
            "analyzed_criteria": analyzed_results, # Detailed results for each criterion analyzed
            "clarified_items": clarified_items, # Subset of analyzed_criteria that became MET
            "remaining_gaps": remaining_gaps,   # Subset that are still NOT_MET/UNCLEAR/ERROR
            "strategic_next_steps": strategic_next_steps,
            "internal_search_results": internal_search_findings # Add search results to final report
        } 
    # --- Internal Search Helper Methods (Task 5.1.2) ---
    def _search_lab_component(self, labs_data: List[Dict[str, Any]], lab_keywords: List[str]) -> List[Dict[str, Any]]:
        """Searches lab data for specific components."""
        findings = []
        if not labs_data or not isinstance(labs_data, list):
            return findings
            
        keywords_lower = [k.lower() for k in lab_keywords]
        
        for lab_panel in labs_data:
            if not isinstance(lab_panel, dict): continue
            panel_name = lab_panel.get('panelName', 'Unknown Panel')
            lab_date = lab_panel.get('resultDate', lab_panel.get('orderDate', '?'))
            for component in lab_panel.get('components', []):
                if not isinstance(component, dict): continue
                test_name = component.get('test', '')
                if not test_name or not isinstance(test_name, str): continue
                
                test_name_lower = test_name.lower()
                for keyword in keywords_lower:
                    if keyword in test_name_lower:
                        findings.append({
                            "source": f"Lab Panel '{panel_name}' ({lab_date})",
                            "context": f"{test_name}: {component.get('value', 'N/A')} {component.get('unit', '')} (Ref: {component.get('refRange', '?')})",
                            "match": keyword
                        })
                        break # Found match for this component
        return findings

    def _search_keyword_sentence(self, notes_data: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """Searches notes for sentences containing specific keywords or regex patterns."""
        findings = []
        if not notes_data or not isinstance(notes_data, list):
            return findings

        # Compile keywords into a single regex pattern for efficiency if needed, 
        # or just do simple lowercased substring checks.
        # Using simple checks for now.
        keywords_lower = [k.lower() for k in keywords if isinstance(k, str)] # Basic keywords
        regex_patterns = [re.compile(p, re.IGNORECASE) for p in keywords if not isinstance(p, str)] # If patterns are included

        for note in notes_data:
            if not isinstance(note, dict): continue
            note_text = note.get('text', '')
            if not note_text or not isinstance(note_text, str): continue
            note_source = f"Note ({note.get('date', '?')} by {note.get('provider', '?')})"
            
            # Simple sentence splitting
            sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', note_text) if s and s.strip()]
            for sentence in sentences:
                sentence_lower = sentence.lower()
                found_match = None
                # Check basic keywords
                for keyword in keywords_lower:
                    if keyword in sentence_lower:
                        found_match = keyword
                        break
                # Check regex patterns if keyword not found
                if not found_match:
                    for pattern in regex_patterns:
                        if pattern.search(sentence): # Use search on original sentence for regex
                            found_match = pattern.pattern # Store the pattern as the match
                            break
                            
                if found_match:
                    findings.append({
                        "source": note_source,
                        "context": sentence,
                        "match": found_match
                    })
                    # Don't break here, a sentence might match multiple keywords/patterns
                    # If we only want the first match per sentence, add a break here.
        return findings

    def _search_mutation_list(self, mutations_data: List[Dict[str, Any]], gene_symbol: str) -> List[Dict[str, Any]]:
        """Searches the mutations list for a specific gene symbol."""
        findings = []
        if not mutations_data or not isinstance(mutations_data, list) or not gene_symbol:
            return findings
            
        gene_symbol_upper = gene_symbol.upper()
        
        for mutation in mutations_data:
             # Access hugo_gene_symbol directly now it's top level in DB result
            hugo_symbol = mutation.get('hugo_gene_symbol')
            if hugo_symbol and isinstance(hugo_symbol, str) and hugo_symbol.upper() == gene_symbol_upper:
                findings.append({
                    "source": "Patient Mutations List (DB)",
                    "context": f"Gene: {hugo_symbol}, Change: {mutation.get('protein_change', 'N/A')}, Type: {mutation.get('variant_type', 'N/A')}, Status: {mutation.get('mutation_status', 'N/A')}",
                    "match": gene_symbol, # The gene symbol we searched for
                    "raw_mutation_data": mutation # Include the whole record for context
                })
                # Continue searching, patient might have multiple mutations in the same gene
                
        return findings

    def _search_medication_check(self, meds_data: List[Dict[str, Any]], known_list: List[str]) -> List[Dict[str, Any]]:
        """Checks current medications against a known list of drugs."""
        findings = []
        if not meds_data or not isinstance(meds_data, list):
            return findings
            
        known_list_lower = [k.lower() for k in known_list]
        
        for med in meds_data:
            if not isinstance(med, dict): continue
            med_name = med.get('name', '')
            if not med_name or not isinstance(med_name, str): continue
            
            med_name_lower = med_name.lower()
            for known_med in known_list_lower:
                if known_med in med_name_lower: # Simple substring check
                    findings.append({
                        "source": "Current Medications List",
                        "context": f"{med_name} {med.get('dosage', '')} {med.get('frequency', '')}",
                        "match": known_med
                    })
                    break # Found match for this medication
        return findings
        
    # --- End Internal Search Helpers --- 
    
# This allows the script to be run directly for testing if needed
if __name__ == '__main__':
    # Example of how to run the agent
    async def main():
        # You would need to mock patient_data, trial_data, etc.
        # This is just a placeholder for a potential test execution
        print("EligibilityDeepDiveAgent can be tested here.")
        # agent = EligibilityDeepDiveAgent()
        # result = await agent.run(...)
        # print(result)
    
    asyncio.run(main())
        
