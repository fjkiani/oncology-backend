import logging
from typing import Dict, Any, Optional
import time # Added
from datetime import datetime # Added

from .data_analysis_agent import DataAnalysisAgent
# from .interaction_check_agent import InteractionCheckAgent # <--- COMMENTED OUT TEMPORARILY
from .genomic_analyst_agent import GenomicAnalystAgent
# from .consult_note_analyzer_agent import ConsultNoteAnalyzerAgent # <--- COMMENTED OUT TEMPORARILY
from .consultation_synthesizer_agent import ConsultationSynthesizerAgent # <-- Import new agent
from .notification_agent import NotificationAgent
from .scheduling_agent import SchedulingAgent
from .referral_agent import ReferralAgent
from .clinical_trial_agent import ClinicalTrialAgent
from .side_effect_agent import SideEffectAgent
from .comparative_therapy_agent import ComparativeTherapyAgent
from .patient_education_draft_agent import PatientEducationDraftAgent
from .integrative_medicine_agent import IntegrativeMedicineAgent
from .crispr_agent import CRISPRAgent
from .lab_order_agent import LabOrderAgent
from ..config import constants
# from ..utils.intent_parser import IntentParser # <--- COMMENTED OUT TEMPORARILY
# from ..llm.gemini_client import GeminiClient # <--- COMMENTED OUT TEMPORARILY

logger = logging.getLogger(__name__)

# Define user-friendly names and descriptions, can be moved to constants or derived
AGENT_METADATA = {
    constants.DATA_ANALYZER: {
        "name": "Data Analysis Agent",
        "description": "Analyzes patient data for summaries, deep dives, and answers questions.",
        "supported_intents": [constants.SUMMARIZE, constants.SUMMARIZE_DEEP_DIVE, "answer_question"] # Example
    },
    constants.GENOMIC_ANALYST: {
        "name": "Genomic Analyst Agent",
        "description": "Performs analysis of genomic profiles and individual mutations.",
        "supported_intents": [constants.ANALYZE_GENOMIC_PROFILE]
    },
    constants.CONSULTATION_SYNTHESIZER: {
        "name": "Consultation Synthesizer Agent",
        "description": "Synthesizes consultation notes from various inputs.",
        "supported_intents": [constants.SYNTHESIZE_CONSULTATION] # Assuming command maps to an intent-like concept
    },
    constants.NOTIFIER: {
        "name": "Notification Agent",
        "description": "Drafts and manages notifications to care team members or patients.",
        "supported_intents": [constants.NOTIFY]
    },
    constants.SCHEDULER: {
        "name": "Scheduling Agent",
        "description": "Finds available appointment slots and books appointments.",
        "supported_intents": [constants.SCHEDULE]
    },
    constants.REFERRAL_DRAFTER: {
        "name": "Referral Agent",
        "description": "Drafts referral letters to specialists.",
        "supported_intents": [constants.REFERRAL]
    },
    constants.CLINICAL_TRIAL_FINDER: {
        "name": "Clinical Trial Agent",
        "description": "Finds relevant clinical trials based on patient criteria.",
        "supported_intents": [constants.FIND_TRIALS]
    },
    constants.SIDE_EFFECT_MANAGER: {
        "name": "Side Effect Agent",
        "description": "Provides information and management tips for treatment side effects.",
        "supported_intents": [constants.MANAGE_SIDE_EFFECTS]
    },
    constants.COMPARATIVE_THERAPY_AGENT: {
        "name": "Comparative Therapy Agent",
        "description": "Provides insights on comparative therapy options (conceptual).",
        "supported_intents": [] # Or relevant intents if directly callable
    },
    constants.PATIENT_EDUCATOR: {
        "name": "Patient Education Agent",
        "description": "Drafts educational materials for patients.",
        "supported_intents": [] # Or relevant intents
    },
    constants.INTEGRATIVE_MEDICINE_AGENT: {
        "name": "Integrative Medicine Agent",
        "description": "Provides insights on integrative medicine options (conceptual).",
        "supported_intents": []
    },
    constants.CRISPR_AGENT: {
        "name": "CRISPR Agent (Conceptual)",
        "description": "Provides conceptual insights on CRISPR-related gene editing options.",
        "category": "Research & Development",
        "capabilities": ["Simulate gene editing potential", "Identify CRISPR targets (conceptual)"],
        "associatedIntents": [] 
    },
    constants.LAB_ORDER_AGENT: {
        "name": "Lab Order Agent",
        "description": "Handles drafting and managing lab orders.",
        "category": "Administrative",
        "capabilities": ["Draft lab orders based on clinical context", "Suggest relevant tests"],
        "associatedIntents": [constants.DRAFT_LAB_ORDER_COMMAND] 
    },
    constants.CLINICAL_TRIAL_AGENT: {
        "name": "Clinical Trial Agent",
        "description": "Matches patients to clinical trials using detailed eligibility criteria.",
        "category": "Research",
        "capabilities": ["Search trial databases", "Assess eligibility based on patient data", "Provide LLM-based trial summaries"],
        "associatedIntents": [constants.MATCH_ELIGIBLE_TRIALS_COMMAND]
    }
}


class AgentOrchestrator:
    def __init__(self):
        print("***** Entering backend.agents.agent_orchestrator.AgentOrchestrator.__init__ *****") # DEBUG PRINT
        # Initialize LLM Client (or receive it)
        # This might be shared across agents
        # self.llm_client = GeminiClient() # <--- COMMENTED OUT TEMPORARILY
        # self.intent_parser = IntentParser(llm_client=self.llm_client) # <--- COMMENTED OUT TEMPORARILY
        
        # Register agents
        self.agents = {
            constants.DATA_ANALYZER: DataAnalysisAgent(), # <--- REMOVED llm_client
            # constants.INTERACTION_CHECKER: InteractionCheckAgent(), # <--- COMMENTED OUT TEMPORARILY
            constants.GENOMIC_ANALYST: GenomicAnalystAgent(), # <--- REMOVED llm_client
            # constants.CONSULT_NOTE_ANALYZER: ConsultNoteAnalyzerAgent(llm_client=self.llm_client), # <--- COMMENTED OUT TEMPORARILY
            constants.CONSULTATION_SYNTHESIZER: ConsultationSynthesizerAgent(), # <-- REMOVED llm_client
            constants.NOTIFIER: NotificationAgent(),
            constants.SCHEDULER: SchedulingAgent(),
            constants.REFERRAL_DRAFTER: ReferralAgent(),
            constants.CLINICAL_TRIAL_FINDER: ClinicalTrialAgent(),
            constants.SIDE_EFFECT_MANAGER: SideEffectAgent(),
            constants.COMPARATIVE_THERAPY_AGENT: ComparativeTherapyAgent(),
            constants.PATIENT_EDUCATOR: PatientEducationDraftAgent(),
            constants.INTEGRATIVE_MEDICINE_AGENT: IntegrativeMedicineAgent(),
            constants.CRISPR_AGENT: CRISPRAgent(),
            constants.LAB_ORDER_AGENT: LabOrderAgent(),
            constants.CLINICAL_TRIAL_AGENT: ClinicalTrialAgent(),
        }
        logger.info(f"Agent Orchestrator initialized. Agents registered in self.agents: {list(self.agents.keys())}")

        # Initialize agent activity store
        self.agent_activity_store: Dict[str, Dict[str, Any]] = {}
        for agent_key, agent_instance in self.agents.items():
            metadata = AGENT_METADATA.get(agent_key, {})
            # Try to get description from agent's docstring if not in metadata
            description = metadata.get("description", "")
            if not description and agent_instance.__doc__:
                description = agent_instance.__doc__.strip().split('\n')[0]

            agent_name = metadata.get("name", "")
            if not agent_name and hasattr(agent_instance, 'name') and isinstance(agent_instance.name, str):
                agent_name = agent_instance.name.replace("_", " ").title()
            elif not agent_name:
                agent_name = agent_key.replace("_", " ").title()

            self.agent_activity_store[agent_key] = {
                "id": agent_key,
                "name": agent_name,
                "description": description or f"{agent_key} default description.",
                "status": "Idle",
                "lastActivityTimestamp": None,
                "lastPatientId": None,
                "lastActionSummary": None,
                "invocationCount": 0,
                "averageResponseTimeMs": 0.0,
                "totalResponseTimeMs": 0.0,
                "supportedIntents": metadata.get("supported_intents", 
                                                 getattr(agent_instance, 'SUPPORTED_INTENTS', []))
            }
        logger.info(f"Agent activity store initialized: {list(self.agent_activity_store.keys())}")

    def _update_agent_activity_start(self, agent_key: str, action_description: str, patient_id_str: Optional[str]) -> float:
        """Updates agent activity when a task starts and returns the start time."""
        activity_entry = self.agent_activity_store.get(agent_key)
        if not activity_entry:
            logger.error(f"Attempted to start activity for unknown agent_key: {agent_key}. Available keys: {list(self.agent_activity_store.keys())}")
            return time.time() # Return current time, but log indicates an issue

        start_time = time.time()
        status_message = f"Processing: {action_description}"
        if patient_id_str:
            status_message += f" for Patient {patient_id_str}"
        
        activity_entry["status"] = status_message
        logger.info(f"Agent {agent_key} starting: {action_description} for patient {patient_id_str if patient_id_str else 'N/A'}")
        return start_time

    def _update_agent_activity_end(self, agent_key: str, start_time: float, patient_id_str: Optional[str], action_description: str, agent_result: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
        """Updates agent activity when a task ends."""
        activity_entry = self.agent_activity_store.get(agent_key)
        if not activity_entry:
            logger.error(f"Attempted to end activity for unknown agent_key: {agent_key}. Available keys: {list(self.agent_activity_store.keys())}")
            return

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        if error:
            status_message = f"Error: {action_description}"
            if patient_id_str:
                 status_message += f" for Patient {patient_id_str}"
            status_message += f". Details: {str(error)[:100]}" # Truncate error
            activity_entry["status"] = status_message
            activity_entry["lastActionSummary"] = f"Failed: {action_description}. Error: {str(error)[:100]}"
            logger.error(f"Agent {agent_key} errored for {action_description} (Patient {patient_id_str}): {error}", exc_info=True if isinstance(error, Exception) else False)
        else:
            status_message = f"Completed: {action_description}"
            if patient_id_str:
                 status_message += f" for Patient {patient_id_str}"
            activity_entry["status"] = status_message
            
            summary = f"Successfully executed: {action_description}"
            if isinstance(agent_result, dict):
                if agent_result.get("summary"):
                    summary = agent_result.get("summary")
                elif agent_result.get("message"):
                    summary = agent_result.get("message")
            activity_entry["lastActionSummary"] = summary
            logger.info(f"Agent {agent_key} completed: {action_description} for patient {patient_id_str if patient_id_str else 'N/A'}. Status: {activity_entry['status']}")

        activity_entry["lastActivityTimestamp"] = datetime.utcnow().isoformat() + "Z"
        if patient_id_str: # Only update if a patient ID was involved in this specific call
            activity_entry["lastPatientId"] = patient_id_str
        
        activity_entry["invocationCount"] += 1
        activity_entry["totalResponseTimeMs"] += response_time_ms
        activity_entry["averageResponseTimeMs"] = activity_entry["totalResponseTimeMs"] / activity_entry["invocationCount"]
        
        logger.info(f"Agent {agent_key} activity updated. Response time: {response_time_ms:.2f} ms")

    async def _execute_agent_and_track_activity(self, agent_key: str, agent_instance: Any, run_params: Dict[str, Any], patient_id_str: Optional[str], action_description: str) -> Dict[str, Any]:
        activity_entry = self.agent_activity_store.get(agent_key)
        if not activity_entry:
            logger.error(f"Attempted to track activity for unknown agent_key: {agent_key}")
            # Fallback: run agent without tracking if not in store, though it shouldn't happen
            return await agent_instance.run(**run_params)

        start_time = self._update_agent_activity_start(agent_key, action_description, patient_id_str)
        # original_status = activity_entry["status"] # Store original status in case of nested calls or quick updates, this might still be useful if _update_agent_activity_start doesn't preserve it across calls, but for now let's simplify.

        agent_result_payload = None
        error_occurred = None

        try:
            # agent_result = await agent_instance.run(**run_params) # Original call
            agent_result_payload = await agent_instance.run(**run_params)
            
            # Logic for setting status and lastActionSummary moved to _update_agent_activity_end
            # activity_entry["status"] = f"Completed: {action_description}"
            # if patient_id_str:
            #      activity_entry["status"] += f" for Patient {patient_id_str}"
            # activity_entry["lastActionSummary"] = f"Successfully executed: {action_description}"
            # if isinstance(agent_result, dict) and agent_result.get("summary"):
            #      activity_entry["lastActionSummary"] = agent_result.get("summary")
            # elif isinstance(agent_result, dict) and agent_result.get("message"):
            #      activity_entry["lastActionSummary"] = agent_result.get("message")

            # return agent_result # Return the original result

        except Exception as e:
            # logger.error(f"Error running agent {agent_key} for {action_description} (Patient {patient_id_str}): {e}", exc_info=True) # Logging moved
            # activity_entry["status"] = f"Error: {action_description} for Patient {patient_id_str}. Details: {str(e)[:100]}" # Status update moved
            # activity_entry["lastActionSummary"] = f"Failed: {action_description}. Error: {str(e)[:100]}" # Summary update moved
            error_occurred = e # Store the exception
            # agent_result_payload will remain None, or be the error structure below
            agent_result_payload = {"status": "error", "message": f"Error executing command via agent {agent_key}.", "error_details": str(e)}
        
        finally:
            # Call _update_agent_activity_end in finally to ensure it always runs
            self._update_agent_activity_end(
                agent_key,
                start_time,
                patient_id_str,
                action_description,
                agent_result=agent_result_payload if not error_occurred else None, # Pass agent_result only on success
                error=error_occurred
            )
            # Original logic for timestamps, counts, etc., is now in _update_agent_activity_end
            # end_time = time.time()
            # response_time_ms = (end_time - start_time) * 1000
            # activity_entry["lastActivityTimestamp"] = datetime.utcnow().isoformat() + "Z"
            # if patient_id_str:
            #     activity_entry["lastPatientId"] = patient_id_str
            # activity_entry["invocationCount"] += 1
            # activity_entry["totalResponseTimeMs"] += response_time_ms
            # activity_entry["averageResponseTimeMs"] = activity_entry["totalResponseTimeMs"] / activity_entry["invocationCount"]
            # logger.info(f"Agent {agent_key} finished. Status: {activity_entry['status']}. Response time: {response_time_ms:.2f} ms")

        # Return the original agent result or the error structure
        return agent_result_payload if agent_result_payload is not None else {} # Ensure a dict is returned

    async def handle_prompt(self, prompt: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a free-text prompt by parsing intent and routing to the appropriate agent."""
        patient_id = patient_data.get('patientId', 'N/A')
        logger.info(f"Orchestrator handling prompt for patient {patient_id}: '{prompt[:50]}...'")
        
        # 1. Parse Intent (Using your IntentParser or LLM call)
        # For now, we'll need to hardcode or simplify intent determination for the sake of tracking
        # This part needs to be robust for actual agent routing.
        # intent_result = await self.intent_parser.parse_intent(prompt, constants.SUPPORTED_INTENTS)
        intent = None # Placeholder - THIS NEEDS TO BE REPLACED WITH ACTUAL INTENT PARSING
        
        # --- Temporary intent determination based on prompt keywords for dashboard demonstration ---
        # THIS IS A SIMPLIFICATION AND SHOULD BE REPLACED BY A ROBUST INTENT PARSER
        lower_prompt = prompt.lower()
        if "deep dive" in lower_prompt or "summarize_deep_dive" in lower_prompt: # summarize_deep_dive might be passed directly too
            intent = constants.SUMMARIZE_DEEP_DIVE
        elif "summarize" in lower_prompt or "summary" in lower_prompt : # Check after deep dive
            intent = constants.SUMMARIZE
        elif "genomic profile" in lower_prompt or "analyze_genomic_profile" in lower_prompt:
            intent = constants.ANALYZE_GENOMIC_PROFILE
        # Add more keyword-based intents if necessary for demo
        
        # If an intent is passed directly as the prompt (e.g. from UI button designed for specific intent)
        if prompt in [constants.SUMMARIZE, constants.SUMMARIZE_DEEP_DIVE, constants.ANALYZE_GENOMIC_PROFILE]:
             intent = prompt

        logger.info(f"Determined intent (simplified): {intent} for prompt: {prompt}")

        if intent is None:
            # If no intent determined, maybe send to a general query agent or return error
            # For now, we might try to use DataAnalysisAgent as a fallback for questions
            # Or simply return an error if strict intent matching is required.
            # Let's try a generic "answer_question" if no other intent matches, for DataAnalysisAgent
            # This is a fallback, ideally intent parsing is more accurate.
            logger.warning(f"Could not determine specific intent for prompt: '{prompt}'. Treating as general question for DataAnalysisAgent.")
            # return {"status": "error", "message": "Could not determine intent from prompt.", "details": None}
            # For demo purposes, let's assume it's a question for DataAnalysisAgent if no other intent fits
            agent_key = constants.DATA_ANALYZER
            action_description = f"Answer question (from prompt: '{prompt[:30]}...')"
            run_params = {"patient_data": patient_data, "prompt_details": {"intent": "answer_question", "original_prompt": prompt}}
        else:
            agent_key = None
            run_params = {"patient_data": patient_data, "prompt_details": {"intent": intent, "original_prompt": prompt}}
            action_description = f"Intent: {intent}"

            if intent == constants.SUMMARIZE or intent == constants.SUMMARIZE_DEEP_DIVE:
                agent_key = constants.DATA_ANALYZER
                run_params["task"] = "deep_dive" if intent == constants.SUMMARIZE_DEEP_DIVE else "summarize"
            elif intent == constants.CHECK_INTERACTIONS: # Currently commented out
                # agent_key = constants.INTERACTION_CHECKER 
                pass
            elif intent == constants.ANALYZE_GENOMIC_PROFILE:
                 agent_key = constants.GENOMIC_ANALYST
            
        if agent_key and agent_key in self.agents:
            agent_instance = self.agents[agent_key]
            return await self._execute_agent_and_track_activity(
                agent_key=agent_key,
                agent_instance=agent_instance,
                run_params=run_params,
                patient_id_str=patient_id,
                action_description=action_description
            )
        elif agent_key is None and intent is not None: # Intent parsed but no agent configured
             logger.warning(f"No agent configured for intent: {intent}")
             return {"status": "error", "message": f"Intent '{intent}' is recognized but no agent is configured to handle it."}
        else: # Fallback from above if no specific intent but we decided to route to DataAnalysisAgent for general question
            if agent_key == constants.DATA_ANALYZER and action_description.startswith("Answer question"):
                 agent_instance = self.agents[agent_key]
                 return await self._execute_agent_and_track_activity(
                    agent_key=agent_key,
                    agent_instance=agent_instance,
                    run_params=run_params, # run_params already set for this case
                    patient_id_str=patient_id,
                    action_description=action_description
                )
            logger.error(f"Could not determine agent for prompt: '{prompt}' (Intent: {intent})")
            return {"status": "error", "message": f"Could not determine appropriate agent for prompt: {prompt}", "details": None}


    async def handle_agent_command(self, message: Dict[str, Any], patient_data_cache: Optional[Dict] = None) -> Dict[str, Any]:
        """Handles specific commands often originating from UI buttons (like in ConsultationPanel)."""
        command = message.get("command")
        params = message.get("params", {})
        patient_id = message.get("patientId") # Get patient ID from message
        
        # ---- START DEBUG PRINTS ----
        logger.info(f"[CMD_DEBUG] Received command: '{command}' (type: {type(command)})")
        logger.info(f"[CMD_DEBUG] constants.SCHEDULE: '{constants.SCHEDULE}' (type: {type(constants.SCHEDULE)})")
        logger.info(f"[CMD_DEBUG] constants.REFERRAL: '{constants.REFERRAL}' (type: {type(constants.REFERRAL)})")
        logger.info(f"[CMD_DEBUG] command == constants.SCHEDULE is {command == constants.SCHEDULE}")
        logger.info(f"[CMD_DEBUG] command == constants.REFERRAL is {command == constants.REFERRAL}")
        # ---- END DEBUG PRINTS ----

        logger.info(f"Orchestrator handling command: '{command}' for patient {patient_id}")
        
        agent_key = None
        run_params = {}
        # requires_patient_data = False # We'll handle this per command
        action_description = f"Command: {command}" # Default action description

        if command == constants.SYNTHESIZE_CONSULTATION:
            agent_key = constants.CONSULTATION_SYNTHESIZER
            initial_context = params.get("initial_context")
            initiator_note_analysis = params.get("initiator_note_analysis")
            if not initial_context or not initiator_note_analysis:
                 # No agent call, return error directly
                 return {"type": "error", "status": "error", "error": "Missing context or analysis in params for synthesis command."}
            run_params = {
                "initial_context": initial_context,
                "initiator_note_analysis": initiator_note_analysis
            }
            action_description = "Synthesize Consultation Note" # More concise

        elif command == constants.SUMMARIZE:
            agent_key = constants.DATA_ANALYZER
            action_description = "Summarize patient data (via command)"
            if not patient_id:
                return {"type": "error", "status": "error", "error": "Patient ID is required for summarize command."}
            if not patient_data_cache:
                return {"type": "error", "status": "error", "error": "Patient data cache not provided for summarize command."}
            
            patient_data = patient_data_cache.get(patient_id.upper())
            if not patient_data:
                return {"type": "error", "status": "error", "error": f"Patient data for {patient_id} not found in cache."}
            
            run_params = {
                "patient_data": patient_data, # Pass the actual patient data
                "prompt_details": {"intent": "summarize", "original_prompt": action_description}
            }
        
        elif command == constants.SUMMARIZE_DEEP_DIVE: # Assuming a deep dive command
            agent_key = constants.DATA_ANALYZER
            action_description = "Perform deep dive summarization (via command)"
            if not patient_id:
                return {"type": "error", "status": "error", "error": "Patient ID is required for deep dive command."}
            if not patient_data_cache:
                return {"type": "error", "status": "error", "error": "Patient data cache not provided for deep dive command."}

            patient_data = patient_data_cache # <--- FIX: Use the cache directly
            if not patient_data: # Check if patient_data (which is patient_data_cache) is empty
                return {"type": "error", "status": "error", "error": f"Patient data for {patient_id} not found or empty in cache."}

            run_params = {
                "patient_data": patient_data, # Pass the actual patient data
                "prompt_details": {"intent": "summarize_deep_dive", "original_prompt": action_description},
                "task": "deep_dive" # Specific for DataAnalysisAgent's deep dive
            }

        elif command == constants.ANALYZE_GENOMIC_PROFILE:
            agent_key = constants.GENOMIC_ANALYST
            action_description = "Analyzing Genomic Profile"
            
            # patient_id is already extracted from the message at the top of the function
            if not patient_id:
                logger.error("Patient ID is missing in message for ANALYZE_GENOMIC_PROFILE command.")
                return {"type": "error", "status": "error", "error": "Patient ID is required for genomic analysis."}

            # For this command, main.py passes the specific patient's data via patient_data_cache argument.
            actual_patient_data_for_agent = patient_data_cache
            
            if not actual_patient_data_for_agent: # Check if it's None or empty
                logger.error(f"Patient data (from patient_data_cache) for {patient_id} is missing or empty for ANALYZE_GENOMIC_PROFILE.")
                return {"type": "error", "status": "error", "error": f"Patient data for {patient_id} is missing or empty."}

            # run_params is initialized as {} at the top of the function.
            # We must populate it correctly here.
            run_params["patient_data"] = actual_patient_data_for_agent
            run_params["prompt_details"] = {"intent": constants.ANALYZE_GENOMIC_PROFILE} # Base prompt_details
            run_params["prompt_details"].update(params or {}) # Add any other params from the message
            
            logger.info(f"[GenomicAnalysisInOrchestrator] Populated run_params for {command}. Patient data keys: {list(actual_patient_data_for_agent.keys())}, Details: {run_params['prompt_details']}")

        elif command == constants.SCHEDULE: # New block for Scheduling
            agent_key = constants.SCHEDULER
            action_description = f"Schedule appointment (via command: {params.get('original_prompt', 'No prompt provided')[:50]})"
            if not patient_id:
                return {"type": "error", "status": "error", "error": "Patient ID is required for schedule command."}
            if not patient_data_cache:
                return {"type": "error", "status": "error", "error": "Patient data cache not provided for schedule command."}
            
            patient_data = patient_data_cache.get(patient_id.upper())
            if not patient_data:
                return {"type": "error", "status": "error", "error": f"Patient data for {patient_id} not found in cache."}

            run_params = {
                "patient_data": patient_data,
                "prompt_details": {
                    "intent": constants.SCHEDULE, # Use the command as the intent
                    "original_prompt": params.get("prompt", "Schedule appointment"), # Get prompt from params
                    "details": params # Pass all other params as details
                }
            }
            # Ensure SchedulingAgent's run method expects these params (e.g., prompt in prompt_details.original_prompt)

        elif command == constants.REFERRAL: # New block for Referral
            agent_key = constants.REFERRAL_DRAFTER
            action_description = f"Draft referral (via command: {params.get('original_prompt', 'No prompt provided')[:50]})"
            if not patient_id:
                return {"type": "error", "status": "error", "error": "Patient ID is required for referral command."}
            if not patient_data_cache:
                return {"type": "error", "status": "error", "error": "Patient data cache not provided for referral command."}

            patient_data = patient_data_cache.get(patient_id.upper())
            if not patient_data:
                return {"type": "error", "status": "error", "error": f"Patient data for {patient_id} not found in cache."}

            run_params = {
                "patient_data": patient_data,
                "prompt_details": {
                    "intent": constants.REFERRAL, # Use the command as the intent
                    "original_prompt": params.get("prompt", "Draft referral letter"), # Get prompt from params
                    "details": params # Pass all other params as details
                }
            }
            # Ensure ReferralAgent's run method expects these params

        elif command == constants.DRAFT_LAB_ORDER_COMMAND:
            agent_key = constants.LAB_ORDER_AGENT
            action_description = f"Draft lab order (via command: {params.get('prompt', 'Draft lab order')[:50]})"
            if not patient_id:
                return {"type": "error", "status": "error", "error": "Patient ID is required for draft_lab_order command."}
            if not patient_data_cache:
                return {"type": "error", "status": "error", "error": "Patient data cache not provided for draft_lab_order command."}
            
            current_patient_data = patient_data_cache.get(patient_id.upper())
            if not current_patient_data:
                return {"type": "error", "status": "error", "error": f"Patient data for {patient_id} not found in cache."}

            run_params = {
                "patient_data": current_patient_data,
                "prompt_details": {"prompt": params.get("prompt", ""), "original_action": params.get("original_action")}
            }
            # This command will now use the common execution logic below

        elif command == constants.MATCH_ELIGIBLE_TRIALS_COMMAND:
            agent_key = constants.CLINICAL_TRIAL_AGENT
            patient_id = message.get("patientId")
            if not patient_id:
                return {"type": "error", "status": "error", "error": "Patient ID is required for matching trials."}

            # patient_data_cache IS the patient's data, already fetched in main.py
            current_patient_data = patient_data_cache if patient_data_cache else {}
            
            if not current_patient_data: # Check if it's an empty dict
                logger.error(f"Patient data for {patient_id} (passed as patient_data_cache) is empty for command {command}.")
                return {"type": "error", "status": "error", "error": f"Patient data for {patient_id} not found or empty."}

            run_params = {"patient_data": current_patient_data, "prompt_details": {"prompt": f"Find trials for patient {patient_id}"}}
            action_description = f"Matching eligible trials for patient {patient_id}"

        else:
            logger.warning(f"Unknown or unhandled command in AgentOrchestrator: {command}")
            return {"type": "error", "status": "error", "error": f"Unknown or unhandled command: {command}"}

        # Actual agent execution using the wrapper
        if agent_key and agent_key in self.agents:
            agent_instance = self.agents[agent_key]
            
            # The _execute_agent_and_track_activity method will return the agent's result,
            # potentially with added status/error keys if the agent itself doesn't provide them.
            # The WebSocket handler in main.py expects a certain structure.
            # We need to ensure the final output is suitable.
            
            agent_execution_result = await self._execute_agent_and_track_activity(
                agent_key=agent_key,
                agent_instance=agent_instance,
                run_params=run_params,
                patient_id_str=patient_id, # Pass patient_id, can be None if not relevant
                action_description=action_description
            )

            # Construct a consistent response structure for the WebSocket handler
            if isinstance(agent_execution_result, dict) and agent_execution_result.get("status") == "error":
                # Error already formatted by the tracker
                return {
                    "type": "error", # Or a more specific error type if available
                    "status": "error",
                    "command": command,
                    "error": agent_execution_result.get("message", agent_execution_result.get("error_details", "Agent execution failed.")),
                    "details": agent_execution_result.get("error_details")
                }
            else:
                # Success, ensure it has a 'content' field or similar for the WS handler
                # The tracker itself doesn't add 'content'; it returns the agent's raw result.
                # The specific agent needs to return a dict with 'content', 'summary', or 'result.text' etc.
                # For dashboard purposes, the tracking is done. This return is for the client.
                # Let's return the agent_execution_result directly, the WS handler in main.py
                # will then pick out the relevant parts.
                # It's also important to add a 'type' for the WS handler if the agent doesn't.
                
                response_type = command # Default type to the command name
                if command == constants.SYNTHESIZE_CONSULTATION:
                     response_type = "consultation_synthesis_result" # Match frontend expectations
                # Add other command-to-type mappings if needed

                # If agent_execution_result is already a well-formed response dict from the agent, use it.
                # Otherwise, wrap it.
                if isinstance(agent_execution_result, dict) and "type" in agent_execution_result:
                    return agent_execution_result
                else:
                    return {
                        "type": response_type,
                        "status": "success", # Assume success if not an error from tracker
                        "command": command,
                        "content": agent_execution_result # This might be the raw output or a dict
                    }
        else:
            # This case should ideally be caught by the initial 'else' for unknown command
            logger.error(f"Logic error: Agent key determined as {agent_key}, but not found in self.agents or was None.")
            return {"type": "error", "status": "error", "error": f"Internal configuration error for command {command}."}