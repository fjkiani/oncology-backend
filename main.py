from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Import BaseModel and Field
import json
import os
from dotenv import load_dotenv # Import load_dotenv
import asyncio # Import asyncio if not already present
from typing import Optional, List, Dict, Any # <-- Import Optional, List, Dict, Any for type hinting
import re # <-- Import regex module
import sys # <-- Import sys
import time
import random
import shlex
import argparse
from datetime import datetime
import logging
import sqlite3 # <--- Import sqlite3
import uuid

# --- Explicitly add project root to sys.path --- 
# This helps resolve module imports when running with uvicorn from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    print(f"Adding {PROJECT_ROOT} to sys.path")
    sys.path.insert(0, PROJECT_ROOT)
# --- End sys.path modification --- 

# Placeholder for future Python-based AI utils
# from . import ai_utils 

# Load environment variables from .env file
load_dotenv()

# Import the orchestrator, blockchain utility, and connection manager
from backend.core.orchestrator import AgentOrchestrator as CoreOrchestrator # Alias the core one
from backend.agents.agent_orchestrator import AgentOrchestrator as AgentsOrchestrator # <-- IMPORT the agents one with alias
from backend.core.blockchain_utils import record_contribution
from backend.core.connection_manager import manager
from backend.core.llm_utils import get_llm_text_response

# Import specific agents needed for slash commands
from backend.agents.comparative_therapy_agent import ComparativeTherapyAgent
from backend.agents.patient_education_draft_agent import PatientEducationDraftAgent
from backend.agents.clinical_trial_agent import ClinicalTrialAgent
from backend.agents.eligibility_deep_dive_agent import EligibilityDeepDiveAgent # <-- Import the new agent

# --- IMPORT CONSTANTS --- Needed for WS command checks
from backend.config import constants 
# --- END CONSTANTS IMPORT ---

# Import the new research router
from backend.research.router import router as research_router

# --- Define DB Path (Assuming it's defined elsewhere or in .env) ---
# If SQLITE_DB_PATH is in .env, it should be loaded via load_dotenv()
# Otherwise, define it explicitly here or import from a config file
TRIALS_DB_PATH = os.getenv('SQLITE_DB_PATH', './backend/data/clinical_trials.db') # Existing path for trials
PATIENT_MUTATIONS_DB_PATH = os.path.join(PROJECT_ROOT, 'backend', 'data', 'patient_data.db')
# --- End DB Path Definition ---

logger = logging.getLogger(__name__) # <--- GET LOGGER INSTANCE
logging.basicConfig(level=logging.DEBUG) # <--- SET LOGGING LEVEL TO DEBUG

app = FastAPI()

# Initialize a global store for Kanban tasks if not already present by previous edits
KANBAN_TASKS_STORE: List[Dict[str, Any]] = []
print(f"Initialized KANBAN_TASKS_STORE: {KANBAN_TASKS_STORE}")


# Instantiate BOTH orchestrators
# The AgentsOrchestrator (plural) is the one with the activity store
agents_orchestrator = AgentsOrchestrator()
print(f"Initialized AgentsOrchestrator (for activity tracking): {type(agents_orchestrator)}")

# The CoreOrchestrator (singular, aliased from core.orchestrator.AgentOrchestrator) handles prompts
# Pass the activity tracking orchestrator to the core orchestrator
orchestrator = CoreOrchestrator(activity_tracking_orchestrator=agents_orchestrator)
print(f"Initialized CoreOrchestrator, passed activity_tracking_orchestrator: {hasattr(orchestrator, 'activity_tracking_orchestrator')}")

# --- New Endpoint for Agent Activity ---
@app.get("/api/agent_activity")
async def get_agent_activity():
    """Returns the current activity status of all registered agents."""
    # logger.info("Accessing /api/agent_activity endpoint...") # REMOVED FORCED LOG
    # raise ValueError("Forced test error in get_agent_activity") # REMOVED FORCED ERROR

    if not agents_orchestrator or not hasattr(agents_orchestrator, 'agent_activity_store'):
        logger.error("AgentOrchestrator (agents_orchestrator) or its activity store is not initialized.")
        raise HTTPException(status_code=500, detail="Agent activity service not available.")
    
    # The store keeps agent_key as key, and the activity dict as value.
    # The frontend dashboard likely expects a list of these activity dicts.
    activity_list = list(agents_orchestrator.agent_activity_store.values())
    return activity_list
# --- End New Endpoint --

# --- NEW Endpoint to retrieve all Kanban tasks ---
@app.get("/api/tasks")
async def get_all_tasks():
    """Returns all tasks currently in the KANBAN_TASKS_STORE."""
    logging.info(f"Accessed /api/tasks endpoint. Returning {len(KANBAN_TASKS_STORE)} tasks.")
    return KANBAN_TASKS_STORE
# --- End NEW /api/tasks Endpoint ---


# Configure CORS
origins = [
    "http://localhost:5173",  # Assuming default Vite dev server port
    "http://localhost:3000",  # Common React dev server port
    "https://oncology-co-pilot.vercel.app", # Added Vercel production frontend URL

    # Add any other origins if necessary (e.g., deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mock data from a JSON file (or define it directly)
# MOCK_DATA_PATH = 'mock_patient_data.json' # Original, now we use the Python dict

# Attempt to import the new mock_patient_data_dict
from backend.data.mock_patient_data_dict import mock_patient_data_dict
print("Successfully imported mock_patient_data_dict from backend.data")

# --- Placeholder Authentication --- 
# In a real app, this would involve JWT decoding, session checking, etc.
async def authenticate_websocket_token(token: str) -> Optional[str]:
    """Placeholder function to validate a token from WebSocket."""
    print(f"Attempting to authenticate token: {token[:20]}...") # Log more of the token
    # Dummy validation: Check if token is not empty and has the correct prefix
    prefix = "valid_token_"
    if token and token.startswith(prefix):
        user_id = token[len(prefix):] # Extract the part AFTER the prefix
        if user_id: # Ensure we extracted something
             print(f"Token validated successfully for user: {user_id}")
             return user_id
    print(f"Token validation failed for token: {token[:20]}...")
    return None
# --- End Placeholder --- 

# Define the endpoint to get patient data
@app.get("/api/patients/{patient_id}")
async def get_patient_data(patient_id: str):
    # In a real app, you'd query a database based on patient_id
    # Convert patient_id to uppercase to make the lookup case-insensitive
    patient_info = mock_patient_data_dict.get(patient_id.upper())
    if not patient_info:
        # To ensure patient_info is always a dictionary for .copy() later if needed,
        # or for direct modification, initialize it if not found to prevent NoneType errors.
        # However, the common pattern is to raise HTTPException if not found.
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found in mock_patient_data_dict")

    # Create a mutable copy to potentially add DB mutations
    patient_data_to_return = patient_info.copy()

    # Check if mutations are already provided by mock_patient_data_dict
    # The .get() method on dict is safer as it returns None if "mutations" key doesn't exist
    mutations_from_mock = patient_info.get("mutations")

    if mutations_from_mock is not None and isinstance(mutations_from_mock, list) and mutations_from_mock:
        print(f"Using {len(mutations_from_mock)} mutations directly from mock_patient_data_dict for patient {patient_id}")
        patient_data_to_return["mutations"] = mutations_from_mock # Ensure these are used
    else:
        print(f"Mutations not found or empty in mock_patient_data_dict for {patient_id}. Attempting to fetch from DB.")
        mutations_from_db = []
        conn = None # Initialize conn here
        try:
            print(f"Connecting to mutations DB at: {PATIENT_MUTATIONS_DB_PATH}")
            if not os.path.exists(PATIENT_MUTATIONS_DB_PATH):
                print(f"Warning: Mutations database not found at {PATIENT_MUTATIONS_DB_PATH}. Patient will have no DB mutations.")
                patient_data_to_return["mutations"] = [] # Set to empty if DB file doesn't exist
            else:
                conn = sqlite3.connect(PATIENT_MUTATIONS_DB_PATH)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM mutations WHERE patient_id = ?", (patient_id,))
                rows = cursor.fetchall()
                mutations_from_db = [dict(row) for row in rows]
                print(f"Fetched {len(mutations_from_db)} mutations from DB for patient {patient_id}")
                patient_data_to_return["mutations"] = mutations_from_db
                
        except sqlite3.Error as e:
            print(f"Database error when fetching mutations for patient {patient_id}: {e}")
            patient_data_to_return["mutations"] = [] # Default to empty list on DB error
        except Exception as e:
            print(f"Unexpected error when fetching mutations for patient {patient_id}: {e}")
            patient_data_to_return["mutations"] = [] # Default to empty list on other errors
        finally:
            if conn:
                conn.close()

    # Ensure the mutations key always exists in the returned data, even if empty
    if "mutations" not in patient_data_to_return:
        patient_data_to_return["mutations"] = []
    
    return {"success": True, "data": patient_data_to_return}

# --- New Prompt Endpoint using Orchestrator ---
class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/prompt/{patient_id}")
async def handle_prompt_request(patient_id: str, request: PromptRequest):
    """ Receives a user prompt and routes it through the orchestrator. """
    # 1. Get patient data
    patient_data = mock_patient_data_dict.get(patient_id.upper(), {})
    if not patient_data:
        raise HTTPException(status_code=404, detail="Patient not found for prompt processing")

    # 2. Call the orchestrator's handle_prompt method
    try:
        result = await orchestrator.handle_prompt(
            prompt=request.prompt,
            patient_id=patient_id,
            patient_data=patient_data
        )
        return result
    except Exception as e:
        print(f"Error during prompt handling: {e}")
        # Consider more specific error handling based on orchestrator responses later
        raise HTTPException(status_code=500, detail=f"Failed to process prompt: {e}")

# --- Feedback Endpoint with Blockchain Logging --- 
class FeedbackRequest(BaseModel):
    feedback_text: str
    ai_output_context: str # e.g., ID of the summary, or the summary text itself for context

@app.post("/api/feedback/{patient_id}")
async def handle_feedback(patient_id: str, request: FeedbackRequest):
    """ 
    Receives feedback on AI output, stores it (conceptually), 
    and logs metadata to the blockchain.
    """
    print(f"Received feedback for patient {patient_id}: {request.feedback_text[:100]}...")
    
    # --- 1. (Conceptual) Store Full Feedback Off-Chain --- 
    # In a real app, you would save request.feedback_text, request.ai_output_context,
    # patient_id, timestamp, user_id etc., into a secure database.
    # For POC, we just construct the data string to be hashed.
    data_to_log = f"PATIENT_ID={patient_id};CONTEXT={request.ai_output_context};FEEDBACK={request.feedback_text}"
    print("Conceptual: Storing feedback off-chain.")
    
    # --- 2. Log Metadata to Blockchain --- 
    try:
        success, tx_hash_or_error = await record_contribution(
            contribution_type="AI_Feedback",
            data_to_log=data_to_log
        )
        
        if success:
            print(f"Blockchain transaction successful: {tx_hash_or_error}")
            return {
                "status": "success", 
                "message": "Feedback received and metadata logged to blockchain.",
                "blockchain_tx_hash": tx_hash_or_error
            }
        else:
            # Log the error but return a user-friendly message
            print(f"Blockchain transaction failed: {tx_hash_or_error}")
            # Don't expose detailed blockchain errors to the frontend
            raise HTTPException(status_code=500, detail="Failed to log feedback metadata to blockchain.")

    except Exception as e:
        # Catch unexpected errors during the process
        print(f"Error handling feedback: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing feedback: {e}")

# --- Helper Functions for Consultation Initiation (Revised) ---

async def _gather_included_data(patient_id: str, include_options: Dict[str, bool]) -> Dict[str, Any]:
    """
    Gathers specified sections of patient data based on the include_options dict.
    """
    print(f"[Data Gathering] Starting for {patient_id} with options: {include_options}")
    related_info = {}
    patient_data = mock_patient_data_dict.get(patient_id.upper(), {})
    if not patient_data:
        print(f"[Data Gathering] Patient {patient_id} not found.")
        return related_info

    # Map include_options keys to mock_patient_data keys and desired format
    data_map = {
        "includeLabs": ("recentLabs", "Recent Labs"),
        "includeMeds": ("currentMedications", "Current Medications"),
        "includeHistory": ("medicalHistory", "Medical History"),
        "includeNotes": ("notes", "Recent Notes"), # Maybe limit notes? e.g., [:2]
        "includeDiagnosis": ("diagnosis", "Diagnosis"),
        # Add more mappings as needed (e.g., imaging)
    }

    for option_key, (data_key, display_key) in data_map.items():
        if include_options.get(option_key, False): # Check if the option is True
            data_section = patient_data.get(data_key)
            if data_section:
                # Simple implementation: Add the whole section. 
                # Could refine later (e.g., only recent notes/labs)
                if data_key == "notes":
                     related_info[display_key] = data_section[:2] # Limit notes
                else:
                    related_info[display_key] = data_section
                print(f"[Data Gathering] Included '{display_key}'")
            else:
                 print(f"[Data Gathering] Section '{display_key}' requested but not found/empty.")

    print(f"[Data Gathering] Completed. Included sections: {related_info.keys()}")
    return related_info

async def _generate_consult_focus(patient_id: str, topic: str, related_info: Dict[str, Any], initiator_note: Optional[str]) -> str:
    """Generates the AI focus statement using LLM based on topic, included data, and note."""
    print(f"[Focus Generation] Starting for {patient_id} based on topic: '{topic[:50]}...'")
    patient_name = mock_patient_data_dict.get(patient_id.upper(), {}).get('demographics', {}).get('name', 'the patient')
    
    prompt = f"Patient: {patient_name} ({patient_id})\n"
    prompt += f"Consultation Topic/Reason: {topic}\n"
    
    if initiator_note:
        prompt += f"Initiator Note: {initiator_note}\n"
    
    prompt += "\nSelected Patient Context Provided:\n"
    if not related_info:
        prompt += "- None provided beyond the topic.\n"
    else:
        # Format included data concisely for the prompt
        for key, value in related_info.items():
            # Basic summarization/truncation for prompt clarity
            if isinstance(value, list) and len(value) > 3:
                 prompt += f"- {key}: (Showing first 3 of {len(value)}) {json.dumps(value[:3], indent=1)}\n"
            elif isinstance(value, list) and not value:
                 prompt += f"- {key}: None\n"
            else:
                prompt += f"- {key}: {json.dumps(value, indent=1)}\n" 
            
    # --- Revised Final Instructions ---
    prompt += "\n---\n" # Add separator
    prompt += "Instructions for AI:\n"
    if initiator_note:
        prompt += "1. **Analyze the 'Initiator Note' above.** This note contains the specific reason and context for this consultation request.\n"
        prompt += "2. **Identify the core clinical question, concern, or decision point highlighted in the 'Initiator Note'.**\n"
        prompt += "3. **Synthesize this core point into a concise 'Consult Focus' statement (1-2 sentences).** This statement should directly reflect the essence of the initiator's note.\n"
        prompt += "4. Use the 'Consultation Topic/Reason' and 'Selected Patient Context' only as background information to understand the broader context, but the focus statement itself MUST derive primarily from the 'Initiator Note'.**\n"
    else:
        # Original logic for when no note is provided
        prompt += "1. Review the 'Consultation Topic/Reason' and the 'Selected Patient Context Provided'.**\n"
        prompt += "2. Synthesize this information into a concise 'Consult Focus' statement (1-2 sentences).**\n"
        prompt += "3. This statement should guide the consulting physician on the likely key question or area needing discussion.\n"
    prompt += "---\nConsult Focus:" # Ask for the output directly after instructions
    
    print(f"[Focus Generation] Prompting LLM:\\n{prompt[:500]}...")
    
    try:
        focus_statement = await get_llm_text_response(prompt)
        print(f"[Focus Generation] LLM Response received: {focus_statement[:100]}...")
        return focus_statement if focus_statement else "AI could not generate a focus statement."
    except Exception as e:
        print(f"[Focus Generation] Error calling LLM: {e}")
        return f"Error generating AI focus statement: {e}"

# --- WebSocket Helper for Agent Actions --- 
async def handle_message_for_agent(message_data: dict, websocket: WebSocket, user_id: str, room_id: str) -> Optional[dict]:
    """
    Checks if a message triggers a direct agent action (e.g., command or button press).
    If so, executes the agent and returns the formatted result for broadcasting.
    Otherwise, returns None.
    """
    message_type = message_data.get("type")
    message_text = message_data.get("text", "").strip()
    agent_name = None
    result_text = None
    status = "success"
    error_message = None

    # Placeholder: Get patient_id from message_data or room_id context if needed
    # Assuming patient_id might be part of the room_id or message context
    # For now, let's extract if sent explicitly in the message, otherwise use a placeholder.
    patient_id = message_data.get("patientId", room_id.split('_')[1] if '_patient_' in room_id else "UNKNOWN_PATIENT")

    # Check for direct agent invocation commands or specific types
    if message_type == "agent_action" and message_data.get("action") == "summarize":
        agent_name = "data_analyzer"
        try:
            agent = orchestrator.agents.get(agent_name)
            if agent:
                # Prepare context and kwargs for DataAnalysisAgent
                patient_data = mock_patient_data_dict.get(patient_id.upper(), {})
                context = {"patient_data": patient_data}
                # Extract relevant parts for the prompt if needed, or pass the whole message
                prompt = message_data.get("payload", {}).get("prompt", "Summarize the patient record.")
                entities = message_data.get("payload", {}).get("entities", {})
                kwargs = {"prompt": prompt, "entities": entities, "patient_id": patient_id}
                
                # Run the agent (adjust based on actual run signature)
                agent_result = await agent.run(context=context, **kwargs) 
                result_text = agent_result.get("output") or agent_result.get("summary", "No summary available.")
            else:
                raise ValueError(f"Agent '{agent_name}' not found.")
        except Exception as e:
            print(f"Error running {agent_name}: {e}")
            status = "failure"
            error_message = f"Failed to execute {agent_name}: {e}"
            result_text = f"Error: Could not generate summary."
    
    elif message_text.startswith("/compare-therapy"):
        agent_name = "comparative_therapist"
        print(f"Handling /compare-therapy command: {message_text}")
        try:
            # First check the format of the message to understand what we're parsing
            print(f"Raw command text: {message_text}")
            
            # Extract parameters without using argparse
            # Example: /compare-therapy current="X" vs="Y" focus="Z"
            command_pattern = r'/compare-therapy\s+current="([^"]*)"\s+vs="([^"]*)"\s+focus="([^"]*)"'
            match = re.search(command_pattern, message_text)
            
            if not match:
                raise ValueError("Command format incorrect. Use: /compare-therapy current=\"therapy1\" vs=\"therapy2\" focus=\"criteria1,criteria2\"")
            
            current_therapy = match.group(1)
            comparison_therapy = match.group(2)
            focus_criteria_text = match.group(3)
            
            print(f"Parsed manually: current={current_therapy}, vs={comparison_therapy}, focus={focus_criteria_text}")
            
            # Split focus criteria into a list
            focus_criteria = [c.strip() for c in focus_criteria_text.split(',')]
            
            agent = orchestrator.agents.get(agent_name)
            if agent:
                result_text = await agent.run(
                    patient_id=patient_id.upper(), 
                    therapy_a=current_therapy, 
                    therapy_b=comparison_therapy, 
                    focus_criteria=focus_criteria
                )
            else:
                raise ValueError(f"Agent '{agent_name}' not found.")
                
        except Exception as e:
            print(f"Error processing /compare-therapy: {e}")
            status = "failure"
            error_message = f"Failed to process command: {e}"
            result_text = f"Error: {e}"

    elif message_text.startswith("/draft-patient-info"):
        # Parse the command using regex instead of argparse to match the format topic="..."
        try:
            match = re.search(r'/draft-patient-info\s+topic="(.*?)"\s*$', message_text, re.IGNORECASE)
            if not match:
                raise ValueError("Invalid command format. Use: /draft-patient-info topic=\"Your explanation topic\"")
            
            topic = match.group(1)
            if not topic:
                raise ValueError("Missing required argument (topic).")
                
            print(f"Parsed command: topic='{topic}'")

            agent_name = "PatientEducationDraftAgent"
            agent = PatientEducationDraftAgent()
            
            try:
                # Wrap the agent execution in a try-except to handle LLM errors gracefully
                result = await agent.run(
                    topic=topic,
                    context={"id": patient_id.upper()} # Minimal context
                )
                result_text = result  # The agent returns the formatted string directly
            except Exception as agent_ex:
                print(f"Error during agent execution: {agent_ex}")
                result_text = f"Sorry, I couldn't generate patient education content: {agent_ex}"
                agent_response_type = "error"
                agent_name = "System"
            else:
                agent_response_type = "agent_output" # USE GENERIC SUCCESS TYPE
        except Exception as e:
            print(f"Error in command parsing: {e}")
            result_text = f"Error: {e}"
            agent_response_type = "error"
            agent_name = "System"

    # --- Add other elif blocks here for future commands like /draft-patient-info ---

    if agent_name and result_text:
        # Format the agent response
        response_data = {
            "type": "agent_output", # USE GENERIC SUCCESS TYPE
            "sender": agent_name,
            "status": status,
            "text": result_text,
            "error": error_message,
            "timestamp": message_data.get("timestamp"), # Keep original timestamp if possible
            "id": message_data.get("id"), # Keep original ID if possible
            "userId": "agent", # Identify sender as agent
            "username": agent_name.replace("_", " ").title(),
            "replyingToTimestamp": message_data.get("timestamp") # Agent reply refers to the command timestamp
        }
        return response_data
    else:
        return None # Not an agent message or command

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host
    client_port = websocket.client.port
    print(f"WebSocket connection attempt from: {client_host}:{client_port}")
    await manager.connect(websocket)
    authenticated_user_id = None
    current_room = None # Track the room this socket has joined

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            message_type = data.get("type")
            print(f"Received WS message type: {message_type} from {authenticated_user_id or f'{client_host}:{client_port}'}")

            if message_type == "auth":
                token = data.get("token")
                # --- Add Logging --- 
                print(f"Auth message data received: {data}") 
                # --- End Logging --- 
                user_id = await authenticate_websocket_token(token)
                if user_id:
                    authenticated_user_id = user_id
                    await manager.associate_user(user_id, websocket)
                    await manager.send_personal_message({"type": "auth_success", "message": f"Authenticated as {user_id}"}, websocket)
                    print(f"User {user_id} authenticated for WebSocket connection from {client_host}:{client_port}")
                    
                    # --- Corrected Auto-Join Logic --- 
                    # Use the patientId explicitly passed during auth for the initial room join
                    patient_id_for_auto_join = data.get("patientId") 
                    if patient_id_for_auto_join:
                        # Ensure we join the PATIENT room, not a stale consult ID
                        await manager.join_room(patient_id_for_auto_join.upper(), websocket)
                        current_room = patient_id_for_auto_join.upper() # Track joined PATIENT room
                        print(f"User {user_id} auto-joined room: {patient_id_for_auto_join}")
                    else:
                        print(f"User {user_id} authenticated but no patientId provided for auto-join.")
                        # Decide if connection should remain open without a primary room
                        # For now, we allow it, but prompts might fail later if no room is joined.

                else:
                    await manager.send_personal_message({"type": "auth_fail", "message": "Invalid token"}, websocket)
                    print(f"Authentication failed for WebSocket from {client_host}:{client_port}. Disconnecting.")
                    await websocket.close(code=1008)
                    manager.disconnect(websocket)
                    break

            # --- All subsequent actions require authentication ---
            elif not authenticated_user_id:
                print(f"WS message ignored from unauthenticated connection {client_host}:{client_port}")
                await manager.send_personal_message({"type": "error", "message": "Authentication required"}, websocket)
                continue # Ignore message, wait for auth

            elif message_type == "join":
                room_id = data.get("roomId")
                if room_id:
                    await manager.join_room(room_id.upper(), websocket)
                    current_room = room_id.upper()
                    await manager.send_personal_message({"type": "status", "message": f"Joined room {room_id}"}, websocket)
                    print(f"User {authenticated_user_id} explicitly joined room: {room_id}")
                else:
                     await manager.send_personal_message({"type": "error", "message": "Room ID missing for join request"}, websocket)

            elif message_type == "prompt":
                if not current_room:
                    await manager.send_personal_message({"type": "error", "message": "Cannot process prompt: Not in a room."}, websocket)
                    continue
                    
                prompt_text = data.get("prompt")
                # Assuming room ID often corresponds to patient ID for general prompts
                patient_id_for_prompt = current_room 
                patient_data = mock_patient_data_dict.get(patient_id_for_prompt.upper())

                if not patient_data:
                     await manager.send_personal_message({"type": "error", "message": f"Patient data not found for ID: {patient_id_for_prompt}"}, websocket)
                     continue
                if not prompt_text:
                     await manager.send_personal_message({"type": "error", "message": "Prompt cannot be empty."}, websocket)
                     continue

                try:
                    print(f"Processing prompt '{prompt_text[:50]}...' for patient {patient_id_for_prompt} in room {current_room}")
                    result = await orchestrator.handle_prompt(
                        prompt=prompt_text,
                        patient_id=patient_id_for_prompt.upper(),
                        patient_data=patient_data
                    )
                    # Send the result back, INCLUDING a top-level status
                    # Extract status from the orchestrator result, default to 'success' if agent worked
                    backend_status = result.get('status', 'success') # Default to success if key missing but no exception occurred
                    await manager.send_personal_message({
                        "type": "prompt_result", 
                        "status": backend_status, # Add top-level status 
                        "result": result # Keep the full original result nested
                        }, websocket)
                except Exception as e:
                    print(f"Error processing prompt via WebSocket: {e}")
                    await manager.send_personal_message({"type": "error", "message": f"Error processing prompt: {e}"}, websocket)

            elif message_type == "initiate_consult":
                # --- Updated Initiate Consult Logic --- 
                target_user_id = data.get("targetUserId")
                patient_id = data.get("patientId")
                initiator_info = data.get("initiator")
                room_id = data.get("roomId")
                context_data = data.get("context")

                if not all([target_user_id, patient_id, initiator_info, room_id, context_data]):
                    print("Initiate consult failed: Missing parameters")
                    await manager.send_personal_message({"type": "initiate_fail", "roomId": room_id, "error": "Missing required parameters"}, websocket)
                    continue
                
                # Extract data based on the revised payload structure
                initial_trigger = context_data.get("initialTrigger") # Contains { description: "..." }
                include_options = context_data.get("includeOptions", {}) # Dict of bools
                use_ai = context_data.get("useAI", False) 
                initiator_note = context_data.get("initiatorNote")
                topic_description = initial_trigger.get("description", "General Consultation")

                print(f"Initiating consult from {initiator_info['id']} to {target_user_id} for patient {patient_id} in room {room_id}.")
                print(f"Topic: '{topic_description[:50]}...', Include Options: {include_options}, AI Assist: {use_ai}")
                if initiator_note: print(f"Initiator Note: {initiator_note[:50]}...")

                # --- Prepare context to send to target user --- 
                related_info = None
                focus_statement = None
                
                try:
                    # 1. Gather included data based on checkboxes (always happens)
                    related_info = await _gather_included_data(patient_id.upper(), include_options)
                except Exception as gather_ex:
                     print(f"Error during data gathering: {gather_ex}")
                     related_info = {"error": f"Could not gather context data: {gather_ex}"}

                if use_ai:
                    try:
                        # 2. Generate focus statement using LLM (only if useAI is true)
                        focus_statement = await _generate_consult_focus(patient_id.upper(), topic_description, related_info or {}, initiator_note)
                    except Exception as focus_ex:
                        print(f"Error during AI focus generation: {focus_ex}")
                        focus_statement = f"AI Error: Could not generate focus ({focus_ex})"

                # Construct the final context payload for the recipient
                context_to_send = {
                    "initialTrigger": initial_trigger, # Keep the original trigger/topic info
                    "initiatorNote": initiator_note,
                    "useAI": use_ai, # Let recipient know if AI was involved
                    "relatedInfo": related_info, # The data gathered based on checkboxes
                    "consultFocusStatement": focus_statement if use_ai else None, # Only include if AI was used
                    # Add patient_data from the original context if available
                    "patient_data": context_data.get("patient_data") if context_data else None
                }

                # --- Find target user socket(s) and send --- 
                target_sockets = await manager.get_user_sockets(target_user_id)
                if target_sockets:
                    message_to_target = {
                        "type": "consult_request",
                        "roomId": room_id,
                        "patientId": patient_id,
                        "initiator": initiator_info,
                        "context": context_to_send # Send the processed context
                    }
                    sent_count = 0
                    for target_socket in target_sockets:
                        try:
                            # Use manager's method which now handles serialization
                            await manager.send_personal_message(message_to_target, target_socket)
                            sent_count += 1
                        except Exception as e:
                             print(f"Error sending consult_request to a socket for {target_user_id}: {e}")
                             
                    if sent_count > 0:
                        print(f"Successfully sent consult request for room {room_id} to {sent_count} socket(s) for user {target_user_id}")
                        await manager.send_personal_message({"type": "initiate_ok", "roomId": room_id}, websocket)
                    else:
                         print(f"Failed sending consult_request to any socket for {target_user_id}")
                         await manager.send_personal_message({"type": "initiate_fail", "roomId": room_id, "error": "Failed to send message to colleague's active sessions"}, websocket)
                else:
                    print(f"Target user {target_user_id} not found or not connected.")
                    await manager.send_personal_message({"type": "initiate_fail", "roomId": room_id, "error": "Colleague is not currently connected"}, websocket)

            # === NEW: Handle structured agent commands from UI (e.g., Analyze Genomic Profile button) ===
            elif message_type == "agent_command":
                command_name = data.get("command")
                # Use current_room, which is defined in the broader websocket_endpoint scope
                logger.info(f"User {authenticated_user_id} in room {current_room} sent agent_command: {command_name}") 
                try:
                    # --- FETCH PATIENT DATA FOR THE COMMAND --- 
                    patient_id_for_command = data.get("patientId")
                    current_patient_data = {}
                    if patient_id_for_command:
                        current_patient_data = mock_patient_data_dict.get(patient_id_for_command.upper(), {})
                        if not current_patient_data:
                            logger.warning(f"Patient data not found in mock_patient_data_dict for ID: {patient_id_for_command} during agent_command.")
                    else:
                        logger.warning("No patientId found in agent_command message data.")
                    # --- END FETCH --- 
                        
                    command_result = await agents_orchestrator.handle_agent_command(
                        message=data, # Corrected parameter name from message_data to message
                        patient_data_cache=current_patient_data 
                    )
                    logger.debug(f"[main.py] Agent command '{command_name}' result before sending to WebSocket: {json.dumps(command_result, indent=2)}")
                    await manager.send_personal_message(command_result, websocket)
                except Exception as e:
                    # Use current_room here as well for consistent logging
                    logger.error(f"Unhandled exception processing agent_command '{command_name}' for user {authenticated_user_id} in room {current_room}: {e}", exc_info=True)
                    error_response = {
                        "type": "agent_command_error", 
                        "command": command_name,
                        "status": "failure",
                        "error": f"Server-side error processing command '{command_name}': {str(e)}",
                        "content": {}
                    }
                    await manager.send_personal_message(error_response, websocket)
            # === End NEW agent_command handler ===
            
            elif message_type == "agent_command_text":
                room_id = data.get("roomId")
                message_text = data.get("text", "").strip() # The raw command text
                sender_info = data.get("sender")
                patient_id_for_command = data.get("patientId") # Expect patientId for context

                if not all([room_id, message_text, sender_info, patient_id_for_command]):
                    await manager.send_personal_message({"type": "error", "message": "Missing fields for agent command text"}, websocket)
                    continue # Skip processing

                # Parse the command
                agent_name = None
                result_text = None
                agent_response_type = "agent_output" # Default response type

                try:
                    if message_text.startswith("/compare-therapy"):
                        try:
                            print(f"Processing /compare-therapy command: {message_text}")
                            
                            # First check the format of the message to understand what we're parsing
                            print(f"Raw command text: {message_text}")
                            
                            # Extract parameters without using argparse
                            # Example: /compare-therapy current="X" vs="Y" focus="Z"
                            command_pattern = r'/compare-therapy\s+current="([^"]*)"\s+vs="([^"]*)"\s+focus="([^"]*)"'
                            match = re.search(command_pattern, message_text)
                            
                            if not match:
                                raise ValueError("Command format incorrect. Use: /compare-therapy current=\"therapy1\" vs=\"therapy2\" focus=\"criteria1,criteria2\"")
                            
                            current_therapy = match.group(1)
                            comparison_therapy = match.group(2)
                            focus_criteria_text = match.group(3)
                            
                            print(f"Parsed manually: current={current_therapy}, vs={comparison_therapy}, focus={focus_criteria_text}")
                            
                            # Split focus criteria into a list
                            focus_criteria = [c.strip() for c in focus_criteria_text.split(',')]
                            
                            # Import again at this scope to be sure
                            try:
                                from backend.agents.comparative_therapy_agent import ComparativeTherapyAgent
                                agent_name = "ComparativeTherapyAgent"
                                agent = ComparativeTherapyAgent()
                                print(f"Instantiated {agent_name}")
                                
                                result = await agent.run(
                                    patient_id=patient_id_for_command.upper(),
                                    therapy_a=current_therapy,
                                    therapy_b=comparison_therapy,
                                    focus_criteria=focus_criteria,
                                    context={"id": patient_id_for_command.upper()}
                                )
                                print(f"Agent run complete, result: {type(result)}")
                                
                                # Handle both string and dict return types
                                if isinstance(result, dict):
                                    result_text = result.get("comparison_summary", str(result))
                                else:
                                    # Assume it's a string if not a dict
                                    result_text = str(result)
                            except ImportError as imp_err:
                                print(f"ImportError when loading ComparativeTherapyAgent: {imp_err}")
                                raise
                            except Exception as agent_err:
                                print(f"Error running ComparativeTherapyAgent: {agent_err}")
                                raise
                                
                        except Exception as ex:
                            print(f"Error in /compare-therapy command block: {ex}")
                            result_text = f"Error processing /compare-therapy command: {ex}"
                            agent_name = "System"
                            agent_response_type = "error"

                    elif message_text.startswith("/draft-patient-info"):
                        try:
                            print(f"Processing /draft-patient-info command: {message_text}")
                            
                            # First check the format of the message to understand what we're parsing
                            print(f"Raw command text: {message_text}")
                            
                            # Extract topic without using argparse
                            # Example: /draft-patient-info topic="Managing nausea from chemotherapy"
                            command_pattern = r'/draft-patient-info\s+topic="([^"]*)"'
                            match = re.search(command_pattern, message_text)
                            
                            if not match:
                                raise ValueError("Command format incorrect. Use: /draft-patient-info topic=\"Your topic here\"")
                            
                            topic = match.group(1)
                            print(f"Parsed manually: topic={topic}")
                            
                            # Import again at this scope to be sure
                            try:
                                from backend.agents.patient_education_draft_agent import PatientEducationDraftAgent
                                agent_name = "PatientEducationDraftAgent"
                                agent = PatientEducationDraftAgent()
                                print(f"Instantiated {agent_name}")
                                
                                result = await agent.run(
                                    topic=topic,
                                    context={"id": patient_id_for_command.upper()} # Minimal context
                                )
                                print(f"Agent run complete, result: {type(result)}")
                                
                                # Handle both string and dict return types
                                if isinstance(result, dict):
                                    result_text = result.get("draft_content", str(result))
                                else:
                                    # Assume it's a string if not a dict
                                    result_text = str(result)
                                agent_response_type = "agent_output" # <- USE GENERIC SUCCESS TYPE
                            except ImportError as imp_err:
                                print(f"ImportError when loading PatientEducationDraftAgent: {imp_err}")
                                raise
                            except Exception as agent_err:
                                print(f"Error running PatientEducationDraftAgent: {agent_err}")
                                raise
                                
                        except Exception as ex:
                            print(f"Error in /draft-patient-info command block: {ex}")
                            result_text = f"Error processing /draft-patient-info command: {ex}"
                            agent_name = "System"
                            agent_response_type = "error"

                    else:
                         # Command not recognized
                        result_text = f"Unknown command: {message_text.split()[0]}"
                        agent_name = "System"
                        agent_response_type = "error" # Send as error type

                except (argparse.ArgumentError, Exception) as e:
                    print(f"Error parsing or running agent command '{message_text}': {e}")
                    result_text = f"Error processing command: {e}"
                    agent_name = "System"
                    agent_response_type = "error"

                # Prepare response if command was processed (even if it was an error message)
                if agent_name and result_text is not None:
                    timestamp = asyncio.get_event_loop().time()
                    
                    # Check if it was an error or a successful agent output
                    if agent_response_type == "error":
                        response_data = {
                            "type": "error",
                            "roomId": room_id,
                            "agentName": agent_name, # Could be "System"
                            "sender": sender_info,
                            "timestamp": timestamp,
                            "message": result_text # Error message content
                        }
                    else: # Assume it's a successful "agent_output"
                        response_data = {
                            "type": "agent_output", # USE GENERIC SUCCESS TYPE
                            "roomId": room_id,
                            "agentName": agent_name, # Actual agent name
                            "sender": sender_info, 
                            "timestamp": timestamp,
                            "content": result_text # Agent result content
                        }

                    # Broadcast the agent's result or error
                    print(f"Broadcasting agent ({agent_name}) message (type: {response_data['type']}) to room {room_id}")
                    await manager.broadcast_to_room(room_id, response_data)
                else:
                    # This case handles parsing/execution errors where we only send back a personal message
                    await manager.send_personal_message({
                        "type": "error", 
                        "message": result_text or "Failed to process command.", 
                        "timestamp": asyncio.get_event_loop().time()
                    }, websocket)
                    
                continue # Agent command handled, skip further checks

            elif message_type == "chat_message":
                room_id = data.get("roomId")
                message_content = data.get("content")
                sender_info = data.get("sender") # {id: ..., name: ...}
                if room_id and message_content and sender_info:
                    # Add timestamp on the server
                    timestamp = asyncio.get_event_loop().time() 
                    chat_payload = {
                        "type": "chat_message",
                        "roomId": room_id,
                        "content": message_content,
                        "sender": sender_info,
                        "timestamp": timestamp
                    }
                    print(f"Broadcasting chat message in room {room_id} from {sender_info['id']}")
                    await manager.broadcast_to_room(room_id, chat_payload, exclude_sender=websocket)
                    # Also send back to sender for confirmation/display
                    await manager.send_personal_message(chat_payload, websocket)
                else:
                    await manager.send_personal_message({"type": "error", "message": "Missing fields for chat message"}, websocket)

            elif message_type == "analyze_initiator_note":
                room_id = data.get("roomId")
                note_text = data.get("note_text")
                sender_info = data.get("sender") # Include sender info if available

                if not room_id or not note_text:
                    await manager.send_personal_message({"type": "error", "message": "Missing roomId or note_text for analysis request."}, websocket)
                    continue

                print(f"Received request to analyze initiator note in room {room_id}")
                await manager.send_personal_message({"type": "system_message", "roomId": room_id, "content": "Analyzing initiator's note..."}, websocket)

                analysis_prompt = (
                    "You are an AI assistant helping clinicians collaborate. Below is a detailed note written by an initiating clinician explaining the reason for a consultation request. "
                    "Please analyze this note thoroughly.\n\n"
                    "**Initiator's Note:**\n"
                    f"'''{note_text}'''\n\n"
                    "**Your Task:**\n"
                    "1. Summarize the core clinical reasoning presented in the note.\n"
                    "2. Identify the key decision points or clinical questions (explicit or implicit) raised by the initiator.\n"
                    "3. Extract any specific concerns or nuances highlighted.\n"
                    "4. Present your analysis clearly and concisely, formatted perhaps with bullet points or short paragraphs, suitable for the consulting clinician."
                    "\n\n**Analysis:**\n"
                )

                analysis_result = "Analysis could not be generated." # Default
                error_message = None
                try:
                    analysis_result = await get_llm_text_response(analysis_prompt)
                    print(f"LLM analysis of initiator note completed for room {room_id}.")
                except Exception as e:
                    print(f"Error during LLM call for initiator note analysis: {e}")
                    error_message = f"Failed to analyze note: {e}"

                # Prepare response payload
                analysis_payload = {
                    "type": "initiator_note_analysis",
                    "roomId": room_id,
                    "analysis": analysis_result,
                    "error": error_message,
                    # Optionally include sender info if needed later
                    # "requesting_user": sender_info 
                }
                
                # Broadcast the analysis result to the room
                print(f"Broadcasting initiator note analysis to room {room_id}")
                await manager.broadcast_to_room(room_id, analysis_payload)

            else:
                print(f"Unknown message type received: {message_type}")
                # Optionally send an error back
                await manager.send_personal_message({"type": "error", "message": f"Unsupported message type: {message_type}"}, websocket)

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected from {authenticated_user_id or f'{client_host}:{client_port}'} with code: {e.code}")
        # Optionally log disconnect reason if needed (e.g., e.reason)
    except Exception as e:
        # Catch potential errors during receive/processing
        print(f"Error in WebSocket connection handler for {authenticated_user_id or f'{client_host}:{client_port}'}: {e}")
        # Try to close gracefully if possible
        try:
            await websocket.close(code=1011) # Internal Error
        except RuntimeError:
            pass # Already closed or unable to close
    finally:
        # Ensure the connection is removed from the manager on disconnect/error
        print(f"Cleaning up WebSocket connection for {authenticated_user_id or f'{client_host}:{client_port}'}")
        manager.disconnect(websocket)
        if authenticated_user_id and current_room:
            # Optional: Broadcast a leave message if desired
            leave_message = {"type": "system_message", "roomId": current_room, "content": f"{authenticated_user_id} left."}
            # Don't await this, just fire and forget if it fails
            # Pass websocket as the third positional arg (sender to exclude)
            asyncio.create_task(manager.broadcast_to_room(current_room, leave_message, websocket))
            

# Include the research API router
app.include_router(research_router, prefix="/api/research", tags=["research"])

# Simple root endpoint
@app.get("/")
async def read_root():
    return {"message": "Beat Cancer AI Backend is running"}

# --- Add Request Model ---
class TrialSearchRequest(BaseModel):
    query: str = Field(..., description="The search query text entered by the user.")
    # Use the more specific PatientContext model defined later
    patient_context: Optional['PatientContext'] = Field(default=None, description="Optional patient context data.")

# --- NEW Clinical Trial Search Endpoint --- 
@app.post("/api/search-trials")
async def search_clinical_trials(request: TrialSearchRequest):
    """
    Endpoint to search for clinical trials using the ClinicalTrialAgent.
    Receives search query and optional patient context.
    
    *** DEVELOPMENT NOTE: Mock data removed. Agent call re-enabled. ***
    """
    logging.info(f"Received trial search request. Query: '{request.query}', Patient Context Provided: {request.patient_context is not None}")
    
    # --- MOCK DATA (Commented Out) --- 
    # mock_action_suggestions = [...]
    # mock_found_trial = {...}
    # mock_result_data = {...}
    # return {
    #     "success": True, 
    #     "data": mock_result_data 
    # }
    # --- END MOCK DATA ---

    # --- Original Agent Call (Re-enabled) ---
    agent = ClinicalTrialAgent() # Instantiate the agent
    
    # Prepare context and kwargs for the agent
    # Agent expects patient data under 'patient_data' key in context
    context = {"patient_data": request.patient_context.model_dump() if request.patient_context else {}} 
    kwargs = {"prompt": request.query} # Pass query as prompt
    
    try:
        # Run the agent asynchronously
        result = await agent.run(patient_data=context.get("patient_data"), prompt_details=kwargs)
        logging.info(f"ClinicalTrialAgent Result Status: {result.get('status')}")
        
        # Check agent status and return appropriate response
        if result.get("status") == "success":
            agent_output = result.get("output", {})
            logging.info(f"[/api/search-trials] Agent output received: {json.dumps(agent_output, indent=2)}") # Log the full agent_output

            found_trials = agent_output.get("found_trials", []) 
            logging.info(f"[/api/search-trials] Initial found_trials (from agent_output.get('found_trials')): {len(found_trials)} trials.")

            if "trials_with_assessment" in agent_output and not found_trials:
                 logging.info(f"[/api/search-trials] 'trials_with_assessment' key found and initial found_trials is empty. Adapting. Count: {len(agent_output['trials_with_assessment'])}")
                 found_trials = agent_output["trials_with_assessment"]
            
            logging.info(f"[/api/search-trials] Final found_trials before returning to frontend: {len(found_trials)} trials.")
            if len(found_trials) > 0:
                logging.debug(f"[/api/search-trials] First trial in final list: {json.dumps(found_trials[0], indent=2)}")


            return {
                "success": True, 
                "data": { "found_trials": found_trials } # Return trials under data.found_trials
            }
        elif result.get("status") == "clarification_needed":
             # For now, treat clarification needed as no results found, 
             # could enhance later to pass clarification message back
              return {
                "success": True, 
                "data": {"found_trials": []}, 
                "message": result.get("summary", "Search criteria unclear.")
            }
        else: # error or other status
             logging.error(f"ClinicalTrialAgent run failed: {result.get('summary')}")
             raise HTTPException(
                status_code=500, 
                detail=result.get("summary", "Agent failed to run")
            )
             
    except Exception as e:
        logging.error(f"Error running ClinicalTrialAgent or processing request: {e}", exc_info=True)
        # Return a standard error response
        raise HTTPException(status_code=500, detail=f"Failed to search trials: {str(e)}")
    # --- End Original Agent Call ---

# --- Data Models ---
class PatientContext(BaseModel):
    # Define fields based on what Research.jsx sends
    # Example: Adjust according to actual patientData structure
    diagnosis: Optional[Dict[str, Any]] = None
    demographics: Optional[Dict[str, Any]] = None
    labs: Optional[List[Dict[str, Any]]] = None
    # Add other relevant fields as needed
    # Make fields Optional if they might not always be present

class TrialSearchRequest(BaseModel):
    query: str
    patient_context: Optional[PatientContext] = None # Use the refined model

class ConsultationRequest(BaseModel):
    room_id: str
    prompt: str
    focus: Optional[str] = None # Context for the agent
    current_history: Optional[List[Dict[str, Any]]] = None # For context aware agents
    replying_to_timestamp: Optional[str] = None # Track message lineage

# --- NEW: Pydantic Models for Plan Followups ---
class ActionSuggestion(BaseModel):
    # Structure based on what ClinicalTrialAgent's ActionSuggester likely provides 
    # and what handlePlanFollowups in Research.jsx expects to send
    action_type: str = Field(..., description="Category of suggestion, e.g., TASK, PATIENT_MESSAGE_SUGGESTION") 
    suggestion: str = Field(..., description="Concise text of the suggested action")
    draft_text: Optional[str] = Field(default=None, description="Pre-drafted text for messages, tasks, etc.")
    criterion: Optional[str] = Field(default=None, description="The related trial criterion text") 
    missing_info: Optional[str] = Field(default=None, description="Explanation of what information is missing, if any")

class PlanFollowupsRequest(BaseModel):
    action_suggestions: List[ActionSuggestion]
    patient_id: Optional[str] = None 
    trial_id: Optional[str] = None   # Added: ID of the trial context for association
    trial_title: Optional[str] = None # Added: Title of the trial context for display

# --- End NEW Pydantic Models ---

# --- Placeholder Planning Agent Logic ---

async def plan_followups_logic(
    suggestions: List[ActionSuggestion], 
    patient_id: Optional[str], 
    trial_id: Optional[str],      # Added parameter
    trial_title: Optional[str]    # Added parameter
) -> List[Dict[str, Any]]: 
    """
    Placeholder logic for the Planning Agent.
    Takes action suggestions, patient ID, and trial context, returns structured Kanban tasks.
    MVP: Converts suggestions (potentially filtering by type later) into basic tasks.
    Future: Implement more sophisticated planning, prioritization, consolidation.
    """
    tasks = []
    logging.debug(f"plan_followups_logic received {len(suggestions)} suggestions for patient ID: {patient_id}, trial ID: {trial_id}")
    for i, suggestion in enumerate(suggestions):
        # Currently converting ALL suggestions to tasks. 
        # Can add filtering later: if suggestion.action_type == 'TASK':
        
        # --- CHANGED: Use criterion text for task content --- 
        # Use the suggestion's criterion text for more specific task content
        # Prefix it for clarity on the Kanban board
        task_content = f"Clarify: {suggestion.criterion}" if suggestion.criterion else suggestion.suggestion
        # --- END CHANGE --- 
        
        # Generate a reasonably unique ID
        task_id = f"task_{patient_id or 'anon'}_{trial_id or 'notrial'}_{int(time.time())}_{i}" # Include trial_id in task_id for uniqueness

        new_task = {
            "id": task_id,
            "columnId": "followUpNeeded", # Default starting column
            "content": task_content, # Use the updated content
            "patientId": patient_id, # Use the passed patient_id
            "suggestion_type": suggestion.action_type,
            "related_criterion": suggestion.criterion, # Keep related criterion for context
            "trial_id": trial_id,          # Added: Store trial ID
            "trial_title": trial_title     # Added: Store trial title
        }
        tasks.append(new_task)
        KANBAN_TASKS_STORE.append(new_task) # Add to global store
            
    logging.info(f"Generated {len(tasks)} Kanban tasks from {len(suggestions)} suggestions for patient {patient_id}, trial {trial_id}. KANBAN_TASKS_STORE now has {len(KANBAN_TASKS_STORE)} tasks.")
    return tasks

# --- NEW: Endpoint for Planning Follow-ups (No longer a stub) --- 
@app.post("/api/plan-followups")
async def api_plan_followups(request: PlanFollowupsRequest):
    """ 
    Receives action suggestions and generates structured Kanban tasks.
    Calls the plan_followups_logic function to perform the task generation.
    Tasks are added to KANBAN_TASKS_STORE.
    """
    logging.info(f"Received request to plan follow-ups. Patient ID: {request.patient_id}, Trial ID: {request.trial_id}")
    logging.debug(f"Action Suggestions Received: {request.action_suggestions}")
    
    try:
        # Call the actual planning logic
        planned_tasks = await plan_followups_logic(
            suggestions=request.action_suggestions, 
            patient_id=request.patient_id, 
            trial_id=request.trial_id, 
            trial_title=request.trial_title
        )
        logging.info(f"Plan followups logic generated {len(planned_tasks)} tasks and added them to store.")
        
        return {"success": True, "planned_tasks": planned_tasks} # Return the newly planned tasks
        
    except Exception as e:
        logging.error(f"Error during plan_followups_logic execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate follow-up tasks: {e}")

# --- End Endpoint --- 

# --- Endpoint to get details and analysis for a SINGLE trial --- 
@app.get("/api/trial-details/{trial_id}")
async def get_trial_details(trial_id: str, patient_id: str):
    """
    Retrieves detailed information about a clinical trial, 
    including eligibility criteria, arms, and locations.
    Optionally provides patient-specific eligibility assessments.
    """
    # Fetch trial data using the trial_id
    trial_data = mock_trials_data.get(trial_id)
    
    if not trial_data:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found")
    
    # If patient_id provided, include initial eligibility assessment
    if patient_id:
        # Always uppercase patient IDs when accessing the mock data dictionary
        patient_data = mock_patient_data_dict.get(patient_id.upper())
        if patient_data:
            # Perform initial eligibility assessment for this patient
            # This is a simplified example - real implementation would be more sophisticated
            eligibility_summary = await clinical_trial_agent.assess_trial_eligibility(patient_data, trial_data)
            response = {
                "trial_data": trial_data,
                "eligibility_assessment": eligibility_summary
            }
        else:
            response = {
                "trial_data": trial_data,
                "error": f"Patient {patient_id} not found - cannot provide personalized eligibility"
            }
    else:
        # Just return trial data without patient-specific assessment
        response = {"trial_data": trial_data}
    
    return response

# --- Deep Dive Endpoint --- 

class DeepDiveRequest(BaseModel):
    """Request body for initiating an eligibility deep dive."""
    unmet_criteria: List[Dict[str, Any]] = Field(..., description="List of criteria initially marked as unmet.")
    unclear_criteria: List[Dict[str, Any]] = Field(..., description="List of criteria initially marked as unclear.")
    patient_data: Dict[str, Any] = Field(..., description="The full patient data object available at the time of request.")
    trial_data: Dict[str, Any] = Field(..., description="The trial data object (containing NCT ID, title, etc.).")

@app.post("/api/request-deep-dive")
async def request_deep_dive(request: DeepDiveRequest):
    """Triggers the EligibilityDeepDiveAgent to analyze specific criteria."""
    logging.info(f"Received request for deep dive analysis for trial: {request.trial_data.get('nct_id', 'N/A')}")
    
    try:
        agent = EligibilityDeepDiveAgent()
        
        # Pass the necessary data from the request to the agent's run method
        # Use request.dict() to pass keyword arguments matching the agent's expected kwargs
        report = await agent.run(**request.dict())
        
        logging.info(f"Deep dive analysis completed for trial: {request.trial_data.get('nct_id', 'N/A')}. Summary: {report.get('summary', 'No summary provided.')}")
        return report
        
    except Exception as e:
        trial_id = request.trial_data.get('nct_id', 'N/A')
        logging.error(f"Error during deep dive analysis for trial {trial_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during the deep dive analysis: {e}"
        )

# --- Pydantic Model for Agent Proxy Request ---
class AgentProxyRequest(BaseModel):
    patientId: str
    intent: str
    prompt: Optional[str] = None
    payload: Optional[Dict[str, Any]] = {}

# --- NEW HTTP Endpoint for Agent Proxy ---
@app.post("/api/agent-proxy")
async def http_agent_proxy(request: AgentProxyRequest):
    """Handles intent-driven agent commands via HTTP POST."""
    logger.info(f"Received HTTP POST request for /api/agent-proxy. Patient ID: {request.patientId}, Intent: {request.intent}")

    current_patient_data = {}
    if request.patientId:
        current_patient_data = mock_patient_data_dict.get(request.patientId.upper(), {})
        if not current_patient_data:
            logger.warning(f"Patient data not found for ID: {request.patientId} during agent-proxy request.")
            # Return a 404 if patient data is essential and not found
            # For some intents, patient data might be optional, adjust as needed.
            # For summarize_deep_dive, it's essential.
            raise HTTPException(status_code=404, detail=f"Patient {request.patientId} not found.")
    
    # Construct a message structure similar to what WebSocket's agent_command expects
    # This allows reusing the agents_orchestrator.handle_agent_command logic
    message_for_orchestrator = {
        "command": request.intent,  # map intent to command
        "patientId": request.patientId,
        "prompt": request.prompt,
        # Add other relevant fields from request.payload if your agents expect them
        # e.g., if payload contains entities, criteria etc.
        **(request.payload if request.payload else {}) 
    }

    try:
        command_result = await agents_orchestrator.handle_agent_command(
            message=message_for_orchestrator,
            patient_data_cache=current_patient_data
        )
        logger.debug(f"[/api/agent-proxy] Agent command '{request.intent}' result: {json.dumps(command_result, indent=2)}")
        
        # Check if the command_result itself indicates an error from the agent
        if command_result.get("status") == "failure" or command_result.get("type") == "error":
            error_detail = command_result.get("error", "Agent processing failed.")
            # Try to get a more specific error from agent's output if available
            if command_result.get("output") and isinstance(command_result.get("output"), dict):
                error_detail = command_result.get("output").get("error", error_detail)
            
            logger.error(f"Agent '{request.intent}' failed for patient {request.patientId}: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)
            
        return command_result
    except HTTPException as http_exc: # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing /api/agent-proxy for intent '{request.intent}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error processing agent request: {str(e)}")

# Add logic to run the app if this script is executed directly
# (e.g., for development/testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
