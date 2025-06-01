import json
import logging
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

# Use absolute import and alias for clarity and robustness
from backend.agents.agent_orchestrator import AgentOrchestrator as WebSocketAgentOrchestrator 
from ..config import constants # Assuming constants are here
from ..storage.patient_data_storage import PatientDataStorage # For fetching patient data

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Holds active connections: Dict[room_id, Set[WebSocket]]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Holds patient data if needed by agents called from consultation rooms
        self.patient_data_cache = PatientDataStorage() # Or your actual data access layer

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = set()
        self.active_connections[room_id].add(websocket)
        logger.info(f"WebSocket {websocket.client} connected to room {room_id}")

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]: # Remove room if empty
                del self.active_connections[room_id]
        logger.info(f"WebSocket {websocket.client} disconnected from room {room_id}")

    async def broadcast_to_room(self, room_id: str, message: str, exclude_sender: WebSocket = None):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                if connection != exclude_sender:
                    try:
                        await connection.send_text(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting to {connection.client} in room {room_id}: {e}", exc_info=True)
                        # Optionally handle disconnects here

    async def send_personal_message(self, websocket: WebSocket, message: str):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message to {websocket.client}: {e}", exc_info=True)

# Instantiate a single ConnectionManager and the SPECIFIC AgentOrchestrator for WebSockets
manager = ConnectionManager()
print("***** In websocket_handler.py: BEFORE instantiating WebSocketAgentOrchestrator *****") # DEBUG PRINT
websocket_orchestrator = WebSocketAgentOrchestrator() # Initialize the WS orchestrator
print("***** In websocket_handler.py: AFTER instantiating WebSocketAgentOrchestrator *****") # DEBUG PRINT

# This is the main WebSocket endpoint function, e.g., for FastAPI
async def websocket_endpoint(websocket: WebSocket, room_id: str, user_id: str, patient_id: str = None):
    """
    Handles WebSocket connections for patient-specific rooms and consultation rooms.
    - patient_id: Primary patient context for the main EHR view.
    - room_id: Can be patient_id for patient context, or a unique ID for consults.
    - user_id: Identifier for the connected user.
    """
    await manager.connect(websocket, room_id)
    logger.info(f"User {user_id} connected to room: {room_id}, primary patient context: {patient_id}")

    # Load patient data into cache if this is a patient room connection
    if patient_id and room_id == patient_id:
        # This assumes your PatientDataStorage has a method to load/cache data
        # manager.patient_data_cache.load_patient(patient_id) 
        pass # Placeholder for actual data loading if needed on connect

    try:
        while True:
            data = await websocket.receive_text()
            message_json = json.loads(data)
            message_type = message_json.get("type")
            client_info = f"client {websocket.client} in room {room_id}"
            logger.debug(f"Received message type '{message_type}' from {client_info}: {data}")

            # --- Message Handling Logic ---
            logger.critical(f"--- NAVIGATING IF/ELIF CHAIN --- message_type: '{message_type}'")

            if message_type is not None and message_type.strip() == "ping":
                logger.critical(f"--- EVALUATING 'ping': TRUE --- message_type: '{message_type}'")
                await manager.send_personal_message(websocket, json.dumps({"type": "pong"}))

            elif message_type is not None and message_type.strip() == "prompt": # From main EHR CoPilot panel
                logger.critical(f"--- EVALUATING 'prompt': TRUE --- message_type: '{message_type}'")
                prompt_text = message_json.get("prompt")
                if not patient_id:
                    logger.warning(f"Prompt received from {client_info} but no patient_id context for room.")
                    await manager.send_personal_message(websocket, json.dumps({
                        "type": "error", "message": "No patient context for this room to process prompt."
                    }))
                    continue
                
                # Fetch patient data using the primary patient_id for this connection
                current_patient_data = manager.patient_data_cache.get_patient_data(patient_id)
                if not current_patient_data:
                    logger.error(f"Patient data not found for {patient_id} to handle prompt from {client_info}")
                    await manager.send_personal_message(websocket, json.dumps({
                        "type": "error", "message": f"Patient data for {patient_id} not found."
                    }))
                    continue

                # Process prompt with orchestrator
                # The orchestrator.handle_prompt now returns a dict that should align with frontend's promptResult
                orchestrator_response = await websocket_orchestrator.handle_prompt(prompt_text, current_patient_data)
                response_msg = {
                    "type": "prompt_result", 
                    "originalPrompt": prompt_text, # For context on frontend
                    "roomId": room_id, # Ensure roomID is part of the response
                    **orchestrator_response # Spread the agent's result (status, output, summary, message)
                }
                await manager.send_personal_message(websocket, json.dumps(response_msg))

            elif message_type is not None and message_type.strip() == "agent_command": # From ConsultationPanel buttons
                logger.critical(f"--- EVALUATING 'agent_command': TRUE --- message_type: '{message_type}'")
                logger.info(f"Handling agent_command from {client_info}: {message_json.get('command')}")
                command_patient_id = message_json.get("patientId") # PatientId is in the command payload

                # Use the patient_data_cache passed to handle_agent_command
                # The orchestrator will fetch data if needed based on command_patient_id
                agent_response = await websocket_orchestrator.handle_agent_command(message_json, manager.patient_data_cache)
                
                response_type = "agent_output" # Default
                original_command = message_json.get("command")

                if original_command == constants.ANALYZE_INITIATOR_NOTE:
                    response_type = "initiator_note_analysis"
                elif original_command == constants.SYNTHESIZE_CONSULTATION:
                    response_type = "consultation_synthesis_result"

                response_message = {
                    "type": response_type,
                    "roomId": room_id,
                    "command": original_command,
                }
                if agent_response.get("status") == "success":
                    response_message["result"] = agent_response.get("result")
                    if agent_response.get("result", {}).get("agentName"):
                        response_message["agentName"] = agent_response["result"]["agentName"]
                else:
                    response_message["error"] = agent_response.get("error", f"Unknown error executing {original_command}")
                
                # Broadcast to the specific room (consultation panel)
                logger.info(f"Broadcasting {response_type} for command '{original_command}' to room {room_id}")
                await manager.broadcast_to_room(room_id, json.dumps(response_message))

            elif message_type is not None and message_type.strip() == "initiate_consult":
                logger.critical(f"--- EVALUATING 'initiate_consult': TRUE --- message_type: '{message_type}'")
                logger.info("%%%%%%% WEBSOCKET HANDLER: ENTERED INITIATE_CONSULT BLOCK (SUCCESS) %%%%%%%")
                target_user_id = message_json.get("targetUserId")
                initiator_info = message_json.get("initiator") # {id, name}
                consult_room_id = message_json.get("roomId")
                consult_context = message_json.get("context")
                logger.info(f"[WS Handler INITIATE_CONSULT] Received consult_context from initiator: {consult_context}")
                consult_patient_id = message_json.get("patientId")
                
                logger.info(f"User {user_id} initiating consult with {target_user_id} for patient {consult_patient_id} in room {consult_room_id}")

                # Construct the notification for the target user
                # This message will be sent to the *target user's main patient room* initially, 
                # or a general notification channel if you have one.
                # For this setup, we assume target user might be in *their own* version of the patient room.
                # A more robust system might use a dedicated user-channel for notifications.
                notification_payload = {
                    "type": "consult_request",
                    "roomId": consult_room_id, # The new room they should join if they accept
                    "patientId": consult_patient_id,
                    "initiator": initiator_info,
                    "context": consult_context
                }
                
                # Broadcast to the *target user's patient context room* or *all rooms*
                # This is a simplification. Ideally, you'd have a way to target a user directly.
                # For now, let's try broadcasting to a room matching the target_patient_id, 
                # hoping Dr. B is viewing that patient when Dr. A initiates.
                # A better way would be to map user_id to their active websocket(s).
                
                # Simplistic: broadcast to all connections in the target patient's room
                # If Dr. B is viewing PAT12345, they will get this.
                target_patient_room_for_notification = consult_patient_id 
                logger.info(f"Broadcasting consult_request to room: {target_patient_room_for_notification} for target user {target_user_id}")
                
                # We need to find the WebSocket(s) for target_user_id. 
                # This current manager setup doesn't directly map user_id to WebSocket.
                # We will broadcast to the target_patient_room_for_notification, and the frontend for target_user_id needs to pick it up.
                # This means if Dr. B is on patient XYZ's page, they won't get the notification for PAT12345.
                
                # For demo: broadcast to the room matching the patientId of the consult
                # This assumes the target colleague is likely viewing the same patient record. 
                # This is a limitation of not having user-specific notification channels.
                connections_in_target_patient_room = manager.active_connections.get(target_patient_room_for_notification, set())
                found_target_user_socket = False
                for ws_conn in connections_in_target_patient_room:
                    # This check is conceptual, as we don't store user_id per WebSocket in this simple manager
                    # In a real app, you'd have a mapping like Dict[user_id, WebSocket] or similar
                    # For now, we just broadcast to everyone in that patient's room.
                    # The frontend for Dr. B will decide if it's for them.
                    await manager.send_personal_message(ws_conn, json.dumps(notification_payload))
                    found_target_user_socket = True # Assume if we sent to the room, it might reach them.

                if found_target_user_socket:
                    await manager.send_personal_message(websocket, json.dumps({"type": "initiate_ok", "roomId": consult_room_id}))
                else:
                    logger.warning(f"Could not find active WebSocket for target user {target_user_id} in patient room {target_patient_room_for_notification} to send consult request.")
                    await manager.send_personal_message(websocket, json.dumps({
                        "type": "initiate_fail", 
                        "roomId": consult_room_id, 
                        "error": f"Colleague {target_user_id} does not seem to be active in the patient's context ({consult_patient_id})."
                    }))

            elif message_type is not None and message_type.strip() == "chat_message":
                logger.critical(f"--- EVALUATING 'chat_message': TRUE --- message_type: '{message_type}'")
                # Broadcast chat messages to everyone in the same (consultation) room
                logger.info(f"Broadcasting chat_message to room {room_id}: {message_json.get('content')[:30]}...")
                await manager.broadcast_to_room(room_id, data, exclude_sender=websocket)
            
            else:
                logger.critical(f"--- FALLING THROUGH TO ELSE --- message_type: '{message_type}'")
                logger.warning(f"Unknown message type received from {client_info}: {message_type}")
                await manager.send_personal_message(websocket, json.dumps({"type": "error", "message": f"Unknown message type: {message_type}"}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by {client_info}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for {client_info}: {e}", exc_info=True)
        try:
            await manager.send_personal_message(websocket, json.dumps({
                "type": "error", "message": f"Server error: {str(e)}"
            }))
        except Exception as send_err:
            logger.error(f"Failed to send error to {client_info} after previous error: {send_err}")
    finally:
        manager.disconnect(websocket, room_id)
        logger.info(f"Cleaned up connection for {client_info}")

# Note: You'd typically register this endpoint with your FastAPI app, e.g.:
# from fastapi import APIRouter
# router = APIRouter()
# router.add_api_websocket_route("/ws/{room_id}/{user_id}", websocket_endpoint)
# And then include this router in your main FastAPI app.
# If patient_id is optional for consult rooms, the route might be /ws/{room_id}/{user_id}
# and you'd handle patient_id being None. For this example, I made it part of the path for patient rooms.
# A common pattern is /ws/{room_id}?user_id=...&patient_id=... (query params)
# Or, more simply, the client sends these IDs in an initial "join" message after connecting.
# This example uses path parameters for simplicity. 