from fastapi import WebSocket
from typing import List, Dict, Set, Optional, Any
import asyncio
import json
from starlette.websockets import WebSocketState

class ConnectionManager:
    """Manages active WebSocket connections, rooms, and user mapping."""
    def __init__(self):
        # Stores connections per room
        self.room_connections: Dict[str, List[WebSocket]] = {}
        # Allows quick lookup of rooms a socket is in
        self.socket_to_rooms: Dict[WebSocket, Set[str]] = {}
        # Maps user ID to their active WebSocket connection(s)
        # Note: A user might have multiple connections (e.g., multiple tabs)
        self.user_connections: Dict[str, List[WebSocket]] = {}
        # Maps WebSocket to user ID for quick reverse lookup
        self.socket_to_user: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket):
        """Accepts a new WebSocket connection. Room joining happens separately."""
        await websocket.accept()
        self.socket_to_rooms[websocket] = set() # Initialize room set for this socket
        print(f"WebSocket connection accepted: {websocket.client.host}:{websocket.client.port}")
        # User association will happen upon successful authentication

    async def associate_user(self, user_id: str, websocket: WebSocket):
        """Associates an authenticated user ID with a WebSocket connection."""
        self.socket_to_user[websocket] = user_id
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        if websocket not in self.user_connections[user_id]:
            self.user_connections[user_id].append(websocket)
            print(f"Associated user '{user_id}' with socket {websocket.client.host}:{websocket.client.port}")
        else:
             print(f"User '{user_id}' already associated with socket {websocket.client.host}:{websocket.client.port}")

    async def get_user_sockets(self, user_id: str) -> List[WebSocket]:
        """Returns a list of active WebSocket connections for a given user ID."""
        return self.user_connections.get(user_id, [])

    async def join_room(self, room_id: str, websocket: WebSocket):
        """Adds a WebSocket connection to a specific room."""
        if room_id not in self.room_connections:
            self.room_connections[room_id] = []
        if websocket not in self.room_connections[room_id]:
            self.room_connections[room_id].append(websocket)
            self.socket_to_rooms[websocket].add(room_id)
            print(f"Socket {websocket.client.host}:{websocket.client.port} joined room '{room_id}'")
            # Optionally broadcast a join message to the room
            # await self.broadcast_to_room(room_id, json.dumps({"type": "join", "user": "anonymous"}), websocket)
        else:
             print(f"Socket {websocket.client.host}:{websocket.client.port} already in room '{room_id}'")

    async def leave_room(self, room_id: str, websocket: WebSocket):
        """Removes a WebSocket connection from a specific room."""
        if room_id in self.room_connections and websocket in self.room_connections[room_id]:
            self.room_connections[room_id].remove(websocket)
            if not self.room_connections[room_id]: # Delete room if empty
                 del self.room_connections[room_id]
            if websocket in self.socket_to_rooms:
                 self.socket_to_rooms[websocket].discard(room_id)
            print(f"Socket {websocket.client.host}:{websocket.client.port} left room '{room_id}'")
            # Optionally broadcast a leave message
            # await self.broadcast_to_room(room_id, json.dumps({"type": "leave", "user": "anonymous"}), websocket)
        else:
            print(f"Socket {websocket.client.host}:{websocket.client.port} not found in room '{room_id}' for removal")

    def disconnect(self, websocket: WebSocket):
        """Handles disconnection, removing socket from rooms and user mappings."""
        user_id = self.socket_to_user.get(websocket)
        client_info = f"{getattr(websocket, 'client', 'UnknownClient')}" # Get client info safely
        print(f"[Disconnect START] for socket {client_info} (User: {user_id or 'N/A'})")
        
        # Remove from rooms - MODIFIED LOGIC
        if websocket in self.socket_to_rooms:
            rooms_to_leave = list(self.socket_to_rooms[websocket]) # Get rooms socket was in
            print(f"[Disconnect] Socket {client_info} leaving rooms: {rooms_to_leave}")
            
            # --- Directly remove socket from room lists --- 
            for room_id in rooms_to_leave:
                print(f"[Disconnect] Attempting removal from room '{room_id}'")
                if room_id in self.room_connections:
                    print(f"[Disconnect] Room '{room_id}' connections BEFORE removal: {self.room_connections[room_id]}")
                    if websocket in self.room_connections[room_id]:
                        try:
                            self.room_connections[room_id].remove(websocket)
                            print(f"[Disconnect] Removed socket from room '{room_id}' list.")
                            if not self.room_connections[room_id]: # Clean up empty room
                                del self.room_connections[room_id]
                                print(f"[Disconnect] Deleted empty room: '{room_id}'")
                            print(f"[Disconnect] Room '{room_id}' connections AFTER removal: {self.room_connections.get(room_id)}")
                        except ValueError:
                            print(f"[Disconnect WARNING] Socket not found in room '{room_id}' during direct removal (ValueError).")
                    else:
                        print(f"[Disconnect WARNING] Socket not found in room '{room_id}' connection list (pre-check). Connections: {self.room_connections[room_id]}")
                else:
                    print(f"[Disconnect WARNING] Room '{room_id}' not found in room_connections during disconnect cleanup.")
            # --- End direct removal --- 

            # Now clear the socket's room tracking
            del self.socket_to_rooms[websocket]
        else:
             print(f"[Disconnect] Socket {client_info} not found in socket_to_rooms mapping.")
            
        # Remove from user mapping
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
                if not self.user_connections[user_id]: # Remove user if no connections left
                    del self.user_connections[user_id]
                print(f"Removed socket association for user '{user_id}'")
            else:
                print(f"Warning: Socket not found in user '{user_id}' connection list during disconnect.")

        # Remove reverse lookup
        if websocket in self.socket_to_user:
            del self.socket_to_user[websocket]
            
        print(f"[Disconnect END] for socket {client_info} (User: {user_id or 'N/A'})")

    async def send_personal_message(self, message_data: Any, websocket: WebSocket):
        """Sends a message (Python object) directly to a specific WebSocket connection, converting to JSON."""
        try:
            # Convert the Python object (dict, list, etc.) to a JSON string
            message_str = json.dumps(message_data)
            await websocket.send_text(message_str)
        except TypeError as e:
            # Handle cases where the data isn't JSON serializable
            print(f"Serialization Error sending personal message to {websocket.client.host}:{websocket.client.port}: {e}. Data: {message_data}")
            # Optionally, disconnect or send an error message back if appropriate
            # For now, just log and potentially disconnect
            self.disconnect(websocket)
        except Exception as e:
             # Catch other errors like connection closed abruptly
             print(f"Error sending personal message to {websocket.client.host}:{websocket.client.port}: {e}")
             self.disconnect(websocket)

    async def broadcast_to_room(self, room_id: str, message_data: Any, sender: Optional[WebSocket] = None):
        """Sends a message (Python object) to all connections in a room (optionally excluding sender), converting to JSON."""
        if room_id in self.room_connections:
            try:
                # Convert the Python object to a JSON string once for efficiency
                message_str = json.dumps(message_data)
                sender_info = f"{getattr(sender, 'client', 'N/A')}" if sender else "N/A"
                print(f"[Broadcast START] Room '{room_id}' (from {sender_info}): {message_str[:100]}...")
                
                # Log the connection list BEFORE iterating
                current_connections = self.room_connections.get(room_id, []) # Get safely
                print(f"[Broadcast] Connections in room '{room_id}' before loop: {current_connections}")

                # Prepare tasks, excluding the sender if provided
                tasks = []
                # Iterate over a COPY of the list to avoid issues if disconnect modifies it during iteration
                connections_in_room = list(current_connections) 
                for websocket in connections_in_room:
                    
                    # Log the specific websocket object being processed
                    print(f"[Broadcast Loop] Processing websocket: {websocket}, State: {getattr(websocket, 'client_state', 'UNKNOWN')}")

                    # --- Check --- 
                    if not websocket or websocket.client_state != WebSocketState.CONNECTED:
                        print(f"[Broadcast Loop] Skipping broadcast to non-existent or disconnected socket: {websocket}")
                        # Optionally try to clean up here if a None or disconnected socket is found
                        if room_id in self.room_connections and websocket in self.room_connections.get(room_id, []):
                             try:
                                 self.room_connections[room_id].remove(websocket)
                                 print(f"[Broadcast Loop] Removed non-connected socket {websocket} from room '{room_id}' list.")
                             except ValueError:
                                 print(f"[Broadcast Loop WARNING] Socket {websocket} not found for removal despite check.")
                        continue # Skip to the next socket
                    # --- END CHECK --- 
                    
                    if sender is None or websocket != sender:
                         # Use a wrapper coroutine to handle potential send errors individually
                         async def send_wrapper(ws, msg_str):
                            ws_info = f"{getattr(ws, 'client', '?')}"
                            try:
                                # Ensure the socket is still connected before sending
                                if ws and ws.client_state == WebSocketState.CONNECTED:
                                    print(f"[Broadcast Send] Sending to {ws_info} in room '{room_id}'")
                                    await ws.send_text(msg_str)
                                else:
                                     print(f"[Broadcast Send SKIPPED] Socket {ws_info} disconnected before send.")
                                     self.disconnect(ws) # Ensure cleanup if state check missed something
                            except Exception as send_ex:
                                print(f"[Broadcast Send ERROR] Error sending to {ws_info} in room '{room_id}': {send_ex}")
                                # Disconnect the problematic socket
                                self.disconnect(ws)
                         
                         tasks.append(send_wrapper(websocket, message_str))
                
                if tasks:
                    print(f"[Broadcast] Gathering {len(tasks)} send tasks for room '{room_id}'")
                    await asyncio.gather(*tasks) # Exceptions are handled within send_wrapper
                    print(f"[Broadcast END] Send tasks completed for room '{room_id}'")
                else:
                    print(f"[Broadcast END] No clients (excluding sender, if specified) in room '{room_id}' to broadcast to.")
                    
            except TypeError as e:
                print(f"[Broadcast ERROR] Serialization Error preparing broadcast for room '{room_id}': {e}. Data: {message_data}")
            except Exception as e:
                 print(f"[Broadcast ERROR] Unexpected error during broadcast preparation for room '{room_id}': {e}")
        else:
             print(f"[Broadcast WARNING] Attempted to broadcast to non-existent room '{room_id}'")

# Singleton instance
manager = ConnectionManager() 