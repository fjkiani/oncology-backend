"""
Core Orchestrator for managing AI agents and handling user prompts.
"""

import json
import os
from typing import Any, Dict, Optional, TYPE_CHECKING
import google.generativeai as genai
import time

# Import Agents
from backend.agents.data_analysis_agent import DataAnalysisAgent
from backend.agents.notification_agent import NotificationAgent
from backend.agents.scheduling_agent import SchedulingAgent
from backend.agents.referral_agent import ReferralAgent
from backend.agents.side_effect_agent import SideEffectAgent
from backend.agents.clinical_trial_agent import ClinicalTrialAgent
from backend.agents.comparative_therapy_agent import ComparativeTherapyAgent
from backend.agents.patient_education_draft_agent import PatientEducationDraftAgent
from backend.agents.genomic_analyst_agent import GenomicAnalystAgent

# Placeholder imports - will need LangChain components later
# from langchain.chat_models import ChatGoogleGenerativeAI # Or other LLM
# from typing import Dict, Any 

# Placeholder for LLM API call
async def call_llm_for_intent(prompt: str) -> dict:
    # ... (existing mock implementation) ...
    if "trial" in prompt or "study" in prompt:
        condition = None
        if "for" in prompt:
             # Basic extraction - Needs improvement!
            try: 
                condition = prompt.split(" for ")[1].split(" involving ")[0].split(" related to ")[0].strip('.?')
            except Exception:
                pass
        entities = {"condition": condition} if condition else {}
        return {"intent": "find_clinical_trials", "entities": entities}
    # ... (other mock intents) ...
    return {"intent": "unknown_intent", "entities": {}}

# Define constants for agent names
DATA_ANALYZER = "data_analyzer"
NOTIFIER = "notifier"
SCHEDULER = "scheduler"
REFERRAL_DRAFTER = "referral_drafter"
CLINICAL_TRIAL_FINDER = "clinical_trial_finder"
SIDE_EFFECT_MANAGER = "side_effect_manager"
COMPARATIVE_THERAPIST = "comparative_therapist"
PATIENT_EDUCATOR = "patient_educator"
GENOMIC_ANALYST = "genomic_analyst"

# Define constants for intent names
SUMMARIZE = "summarize"
SCHEDULE = "schedule"
NOTIFY = "notify"
REFERRAL = "referral"
ANSWER_QUESTION = "answer_question"
FIND_TRIALS = "find_trials"
MANAGE_SIDE_EFFECTS = "manage_side_effects"
UNKNOWN_INTENT = "unknown_intent"
SUMMARIZE_DEEP_DIVE = "summarize_deep_dive"
ANALYZE_GENOMIC_PROFILE = "analyze_genomic_profile"

# Forward reference for type hinting AgentsOrchestrator
if TYPE_CHECKING:
    from backend.agents.agent_orchestrator import AgentOrchestrator as ActivityTrackingOrchestratorType

class AgentOrchestrator:
    """ Coordinates AI agents to handle user prompts and workflows using LLM for intent parsing. """

    # Define supported intents using constants
    SUPPORTED_INTENTS = [
        SUMMARIZE, SCHEDULE, NOTIFY, REFERRAL, 
        ANSWER_QUESTION, FIND_TRIALS, MANAGE_SIDE_EFFECTS,
        SUMMARIZE_DEEP_DIVE,
        ANALYZE_GENOMIC_PROFILE
    ]

    def __init__(self, activity_tracking_orchestrator: Optional['ActivityTrackingOrchestratorType'] = None):
        """ Initializes the orchestrator, loads configuration, and registers agents. """
        # LLM for Intent Parsing/Planning
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            # If the agent initialization didn't already raise an error, this will.
            raise ValueError("Orchestrator requires GOOGLE_API_KEY environment variable.")
        
        try:
            # Assuming genai was configured in main.py or agent init
            # If not, uncomment: genai.configure(api_key=self.api_key)
            self.intent_parser_model = genai.GenerativeModel('gemini-1.5-flash')
            print("Orchestrator Initialized with Intent Parser Model.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize intent parser model: {e}")
        
        # Instantiate and Register Agents using constants
        self.agents = {}
        agent_classes = {
            DATA_ANALYZER: DataAnalysisAgent,
            NOTIFIER: NotificationAgent,
            SCHEDULER: SchedulingAgent,
            REFERRAL_DRAFTER: ReferralAgent,
            CLINICAL_TRIAL_FINDER: ClinicalTrialAgent,
            SIDE_EFFECT_MANAGER: SideEffectAgent,
            COMPARATIVE_THERAPIST: ComparativeTherapyAgent,
            PATIENT_EDUCATOR: PatientEducationDraftAgent,
            GENOMIC_ANALYST: GenomicAnalystAgent
        }
        
        for agent_name, agent_class in agent_classes.items():
            try:
                agent_instance = agent_class()
                self.agents[agent_name] = agent_instance
                print(f"Registered agent: {agent_name}")
            except Exception as e:
                print(f"Error initializing {agent_class.__name__}: {e}. It will be unavailable.")
        
        print("Agent Orchestrator Initialized.")
        self.activity_tracker = activity_tracking_orchestrator
        if self.activity_tracker:
            print(f"CoreOrchestrator (AgentOrchestrator) received activity_tracker: {type(self.activity_tracker)}")
        else:
            print("CoreOrchestrator (AgentOrchestrator) did NOT receive an activity_tracker.")

    async def handle_prompt(self, prompt: str, patient_id: str, patient_data: dict) -> dict:
        """ Receives a prompt, parses intent, routes to the appropriate agent, and returns the result. """
        print(f"Orchestrator received prompt for patient {patient_id}: '{prompt}'")
        context = {"patient_data": patient_data}

        # --- Step 1: Intent Parsing (Using LLM) --- 
        parsed_details = await self._parse_intent_with_llm(prompt)
        print(f"Parsed intent result: {parsed_details}")

        if parsed_details["status"] != "success":
            return parsed_details # Return the parsing error

        intent = parsed_details.get("intent", UNKNOWN_INTENT)
        entities = parsed_details.get("entities", {})
        agent_kwargs = {"prompt": prompt, "intent": intent, "entities": entities, "patient_id": patient_id}

        if intent not in self.SUPPORTED_INTENTS:
            return {"status": "unknown_intent", "message": f"Could not process the recognized intent: {intent}"}

        # Determine agent to use based on intent
        agent_name_to_use = None
        if intent in [SUMMARIZE, ANSWER_QUESTION]:
            agent_name_to_use = DATA_ANALYZER
        elif intent == SUMMARIZE_DEEP_DIVE:
            agent_name_to_use = DATA_ANALYZER
        elif intent == NOTIFY:
            agent_name_to_use = NOTIFIER
        elif intent == SCHEDULE:
            agent_name_to_use = SCHEDULER
        elif intent == REFERRAL:
            agent_name_to_use = REFERRAL_DRAFTER
        elif intent == FIND_TRIALS:
             agent_name_to_use = CLINICAL_TRIAL_FINDER
        elif intent == MANAGE_SIDE_EFFECTS:
             agent_name_to_use = SIDE_EFFECT_MANAGER
        elif intent == ANALYZE_GENOMIC_PROFILE:
            agent_name_to_use = GENOMIC_ANALYST

        if not agent_name_to_use or agent_name_to_use not in self.agents:
            return {"status": "agent_not_implemented", "intent": intent, "message": f"Agent for '{intent}' is not available in this version."}

        # Execute the chosen agent
        agent_result = None
        agent_key_for_tracking = agent_name_to_use
        start_time = time.time() # Default start time
        action_description = "" # Initialize
        patient_id_str_for_tracking = str(patient_id) if patient_id else "N/A" # Define here for broader scope

        # --- Activity Tracking START ---
        if self.activity_tracker and agent_key_for_tracking:
            action_description = f"Prompt: '{prompt[:50]}...' (Intent: {intent})" # Set action_description here
            print(f"[CoreOrchestrator] Calling activity_tracker._update_agent_activity_start for {agent_key_for_tracking}")
            # Capture the returned start_time
            current_start_time = self.activity_tracker._update_agent_activity_start(
                agent_key_for_tracking,
                action_description=action_description,
                patient_id_str=patient_id_str_for_tracking
            )
            if current_start_time: # Ensure it's not None if the method could return that
                start_time = current_start_time
        # --- End Activity Tracking START ---

        try:
            print(f"Routing to agent: {agent_name_to_use} for intent: {intent}")
            agent_to_run = self.agents[agent_name_to_use]
            
            # --- Adapt call signature --- 
            if agent_name_to_use in [CLINICAL_TRIAL_FINDER, DATA_ANALYZER, GENOMIC_ANALYST]:
                agent_result = await agent_to_run.run(patient_data=patient_data, prompt_details=agent_kwargs)
            else:
                 agent_result = await agent_to_run.run(patient_data=patient_data, prompt_details=agent_kwargs)
            # --- End Adaptation --- 

            # Add parsed info to the final result for transparency/debugging
            if agent_result:
                agent_result["parsed_intent"] = intent
                agent_result["parsed_entities"] = entities
            
            # Successfully completed
            if self.activity_tracker and agent_key_for_tracking:
                print(f"[CoreOrchestrator] Calling activity_tracker._update_agent_activity_end (success) for {agent_key_for_tracking}")
                self.activity_tracker._update_agent_activity_end(
                    agent_key_for_tracking,
                    start_time=start_time,
                    patient_id_str=patient_id_str_for_tracking,
                    action_description=action_description, # Use the one defined earlier
                    agent_result=agent_result,
                    error=None
                )
            
            return agent_result if agent_result else {
                 "status": "failure", 
                 "parsed_intent": intent,
                 "parsed_entities": entities,
                 "error_message": f"Agent '{agent_name_to_use}' did not return a result.",
                 "output": None
             }

        except Exception as e:
            print(f"Error during agent execution ({agent_name_to_use}): {e}")
            # Error occurred
            if self.activity_tracker and agent_key_for_tracking:
                print(f"[CoreOrchestrator] Calling activity_tracker._update_agent_activity_end (error) for {agent_key_for_tracking}")
                self.activity_tracker._update_agent_activity_end(
                    agent_key_for_tracking,
                    start_time=start_time,
                    patient_id_str=patient_id_str_for_tracking,
                    action_description=action_description, # Use the one defined earlier
                    agent_result=None,
                    error=e
                )
            return {"status": "failure", "output": None, "summary": f"Agent {agent_name_to_use} failed.", "error_message": str(e)}

    async def _parse_intent_with_llm(self, prompt: str) -> Dict[str, Any]:
        """ Uses the configured LLM to parse intent and extract entities. """
        instruction = f"""
You are an expert clinical assistant responsible for understanding requests about patient records.
Analyze the user's request and determine their primary intent. Also extract any relevant entities.

The possible intents are: {json.dumps(self.SUPPORTED_INTENTS)}

Relevant entities might include:
- timeframe (e.g., "last week", "most recent")
- data_type (e.g., "labs", "CT scan", "notes")
- recipient (e.g., "PCP", "patient", "Dr. Smith")
- specific_condition (e.g., "glucose levels", "nausea", "breast cancer")
- date_details (e.g., "next Tuesday", "in 3 days")
- time_preference (e.g., "morning", "afternoon")
- reason (e.g., "follow-up", "consultation", "evaluation")
- recipient_specialty (e.g., "Cardiology", "Oncology")
- trial_phase (e.g., "2", "3")
- recruitment_status (e.g., "Recruiting", "Active")
- biomarkers (e.g., "EGFR positive", "HER2 negative")
- disease_stage (e.g., "Stage III", "advanced")
- medication_name (e.g., "Letrozole", "Chemotherapy")
- symptom (e.g., "nausea", "fatigue", "rash")
- treatment_type (e.g., "chemotherapy", "immunotherapy")

Analyze the following user request:
{prompt}

Respond ONLY with a JSON object containing:
1. "intent": One of the possible intents listed above, or "unknown_intent" if none fit well.
2. "entities": An object containing any extracted entities as key-value pairs. If no relevant entities are found, return an empty object.

Example Response for "Summarize the latest labs":
{{
  "intent": "summarize",
  "entities": {{
    "timeframe": "latest",
    "data_type": "labs"
  }}
}}

Example Response for "Summarize patient record and provide detailed analysis":
{{
  "intent": "summarize_deep_dive",
  "entities": {{}}
}}

Example Response for "Notify Dr. Baker about the patient's high glucose from yesterday's labs":
{{
  "intent": "notify",
  "entities": {{
    "recipient": "Dr. Baker",
    "specific_condition": "high glucose",
    "timeframe": "yesterday",
    "data_type": "labs"
  }}
}}

Example Response for "What was the blood pressure last week?":
{{
  "intent": "answer_question",
  "entities": {{
    "data_type": "blood pressure",
    "timeframe": "last week"
  }}
}}

Example Response for "Find phase 3 recruiting trials for Stage IV breast cancer":
```json
{{
  "intent": "find_trials",
  "entities": {{
    "trial_phase": 3,
    "recruitment_status": "Recruiting",
    "specific_condition": "breast cancer",
    "disease_stage": "Stage IV"
  }}
}}
```

Example Response for "How do I manage nausea from chemo?":
```json
{{
  "intent": "manage_side_effects",
  "entities": {{
    "symptom": "nausea",
    "treatment_type": "chemo"
  }}
}}
```

Example Response for "What are side effects of Letrozole?":
```json
{{
  "intent": "manage_side_effects",
  "entities": {{
    "medication_name": "Letrozole"
  }}
}}
```

Example Response for "Analyze genomic profile":
```json
{{
  "intent": "analyze_genomic_profile",
  "entities": {{}}
}}
```

JSON Response:
"""

        print("Sending prompt to Intent Parser Model...")
        response_text = "" # Initialize for error handling
        try:
            response = await self.intent_parser_model.generate_content_async(instruction)
            response_text = response.text
            print(f"Raw Intent Parser Response Text: {response_text}") # Log raw response
            
            # Attempt to extract JSON block more robustly
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end+1]
                print(f"Extracted JSON String: {json_str}")
                parsed_json = json.loads(json_str)
            else:
                # Fallback: Try parsing the whole cleaned string if block finding fails
                cleaned_text = response_text.strip().replace("`json", "").replace("```", "")
                if not cleaned_text:
                     raise ValueError("LLM returned an empty response.")
                print(f"Could not find JSON block, attempting to parse cleaned text: {cleaned_text}")
                parsed_json = json.loads(cleaned_text)

            # Basic validation
            if "intent" not in parsed_json or "entities" not in parsed_json:
                raise ValueError("LLM response missing required keys 'intent' or 'entities'.")
            if parsed_json["intent"] not in self.SUPPORTED_INTENTS and parsed_json["intent"] != "unknown":
                 print(f"Warning: LLM returned an intent '{parsed_json['intent']}' not in SUPPORTED_INTENTS list.")
                 # Optionally force to 'unknown' or handle as needed
                 # parsed_json['intent'] = "unknown"
            
            parsed_json["status"] = "success"
            return parsed_json

        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response JSON: {e}\nRaw Response Text: {response_text}")
            return {"status": "failure", "message": "Failed to parse intent JSON from LLM response.", "error_details": str(e)}
        except ValueError as e:
             print(f"Validation Error or Empty LLM Response: {e}\nRaw Response Text: {response_text}")
             return {"status": "failure", "message": f"LLM response invalid: {e}", "error_details": str(e)}
        except Exception as e:
            print(f"Error during intent parsing LLM call: {e}")
            return {"status": "failure", "message": "Error communicating with intent parsing model.", "error_details": str(e)}
       
    # Remove or comment out the old keyword-based parsing method
    # def _parse_intent(self, prompt: str) -> str:
    #     # ... old code ...

# Example Usage (for testing purposes, not directly used by FastAPI yet)
# if __name__ == '__main__':
#     orchestrator = AgentOrchestrator()
#     async def run_test():
#         test_prompt = "Give me a summary for patient PAT12345"
#         # Assuming mock_patient_data_dict is available or loaded here
#         # result = await orchestrator.handle_prompt(test_prompt, "PAT12345", mock_patient_data_dict["PAT12345"])
#         # print(result)
#     import asyncio
#     # asyncio.run(run_test()) 