"""
Agent responsible for handling scheduling-related tasks using LangChain Tools.
"""

import json
import os
from datetime import datetime, timedelta, time
from typing import Any, Dict, Optional, Type

import google.generativeai as genai
from pydantic.v1 import BaseModel, Field

# LangChain Imports
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Import the base class
from backend.core.agent_interface import AgentInterface

# --- Tool Definition --- 
# We define the inputs for our tools using Pydantic models for clarity and validation

class FindSlotsInput(BaseModel):
    start_date_str: Optional[str] = Field(description="Start date for search (YYYY-MM-DD), defaults to today if not specified.")
    end_date_str: Optional[str] = Field(description="End date for search (YYYY-MM-DD), defaults to 14 days from start date if not specified.")
    duration_minutes: Optional[int] = Field(30, description="Desired appointment duration in minutes.")
    time_preference: Optional[str] = Field(None, description="Time preference (e.g., 'morning', 'afternoon').")

@tool("find_available_appointment_slots", args_schema=FindSlotsInput)
def find_available_slots(start_date_str: Optional[str] = None, end_date_str: Optional[str] = None, duration_minutes: int = 30, time_preference: Optional[str] = None) -> str:
    """Finds available appointment slots in the calendar based on specified criteria. Use this to check for openings."""
    print(f"[find_available_slots tool] Called with: start={start_date_str}, end={end_date_str}, duration={duration_minutes}, pref={time_preference}")
    # Basic date parsing (can be improved)
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else datetime.now()
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else start_date + timedelta(days=14)
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=0)
    except ValueError:
         end_date = start_date + timedelta(days=14)
         end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=0)

    # Simple mock implementation (replace with actual calendar API call)
    slots = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5: # Mon-Fri
            slot_9am = current_date.replace(hour=9, minute=0, second=0, microsecond=0)
            slot_2pm = current_date.replace(hour=14, minute=0, second=0, microsecond=0)
            if time_preference:
                if time_preference.lower() == 'morning' and slot_9am.hour < 12:
                     slots.append(slot_9am)
                elif time_preference.lower() == 'afternoon' and slot_2pm.hour >= 12:
                     slots.append(slot_2pm)
            else:
                 slots.append(slot_9am)
                 slots.append(slot_2pm)
        current_date += timedelta(days=1)

    if not slots:
        return "No available slots found matching the criteria."
    else:
        formatted_slots = [s.strftime("%Y-%m-%d %I:%M %p") for s in slots]
        return f"Available slots found: {json.dumps(formatted_slots)}"

class BookAppointmentInput(BaseModel):
    slot_str: str = Field(description="The specific appointment slot to book (YYYY-MM-DD HH:MM AM/PM or ISO format).")
    patient_name: str = Field(description="The name of the patient for whom the appointment is being booked.")
    reason: Optional[str] = Field("Follow-up", description="The reason for the appointment.")

@tool("book_appointment", args_schema=BookAppointmentInput)
def book_appointment(slot_str: str, patient_name: str, reason: Optional[str] = "Follow-up") -> str:
    """Books a specific appointment slot for a patient. Only use this after confirming the slot with the user."""
    print(f"[book_appointment tool] Called with: slot='{slot_str}', patient='{patient_name}', reason='{reason}'")
    # Basic parsing (needs robust error handling)
    try:
        # Attempt parsing multiple formats
        try:
            slot_dt = datetime.strptime(slot_str, "%Y-%m-%d %I:%M %p")
        except ValueError:
             slot_dt = datetime.fromisoformat(slot_str) # Try ISO format
    except ValueError:
         return f"Error: Could not parse the provided slot time '{slot_str}'. Please use YYYY-MM-DD HH:MM AM/PM or ISO format."

    # Simple mock implementation
    print(f"[MockBooking] SIMULATING BOOKING for {patient_name} at {slot_dt.isoformat()} for '{reason}'")
    booking_success = True # Assume success for mock

    if booking_success:
        return f"Successfully booked appointment for {patient_name} at {slot_dt.strftime('%Y-%m-%d %I:%M %p')} for reason: {reason}."
    else:
        return f"Failed to book appointment for {patient_name} at {slot_dt.strftime('%Y-%m-%d %I:%M %p')}."

# --- Scheduling Agent --- 

class SchedulingAgent(AgentInterface):
    """ Handles appointment scheduling using LangChain tools and agent executor. """

    def __init__(self):
        """ Initialize the scheduling agent, tools, LLM, and agent executor. """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("SchedulingAgent requires GOOGLE_API_KEY.")
        
        # Define Tools
        self.tools = [find_available_slots, book_appointment]
        
        # Initialize LLM
        try:
            # Use a model that supports tool calling/function calling well
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
                                              google_api_key=self.api_key,
                                              convert_system_message_to_human=True) # Important for some agent types
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM for SchedulingAgent: {e}")

        # Create Agent Prompt
        # Using a basic prompt, LangChain handles the tool descriptions and formatting
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful medical scheduling assistant. Use the available tools to find slots or book appointments based on the user request and patient context. Always confirm the exact slot with the user before attempting to book."),
            ("placeholder", "{chat_history}"), # For future memory
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), # Where agent thoughts/tool calls go
        ])

        # Create Agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, prompt_template)

        # Create Agent Executor
        self.agent_executor = AgentExecutor(agent=self.agent, 
                                            tools=self.tools, 
                                            verbose=True, # Set to True for debugging agent steps
                                            handle_parsing_errors=True) # Gracefully handle LLM output errors
        
        print("SchedulingAgent Initialized with LangChain tools and executor.")

    @property
    def name(self) -> str:
        return "scheduler"

    @property
    def description(self) -> str:
        return "Handles finding available appointment slots and booking appointments by understanding natural language requests and using calendar tools."

    async def run(self, patient_data: Dict[str, Any] = None, prompt_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Processes a scheduling request using the LangChain agent executor.
        """
        print(f"SchedulingAgent (LangChain) running.")
        original_prompt = prompt_details.get("prompt", "") if prompt_details else ""
        entities = prompt_details.get("entities", {}) if prompt_details else {}
        intent = prompt_details.get("intent", "") if prompt_details else ""
        
        patient_name = patient_data.get("demographics", {}).get("name", "this patient") if patient_data else "this patient"

        # Construct input for the agent executor
        agent_input = {
            "input": f"User request: {original_prompt}. Patient context: Name={patient_name}",
            # Add chat_history here if implementing memory
            "chat_history": [] 
        }

        print(f"Invoking agent executor with input: {agent_input}")
        try:
            # Invoke the agent executor asynchronously
            result = await self.agent_executor.ainvoke(agent_input)
            agent_output = result.get("output", "Agent did not produce standard output.")
            
            print(f"Agent executor result: {result}")

            # Structure the response for the orchestrator
            # Note: The actual useful info (slots found, booking confirmation) 
            # is often directly in the agent's final 'output' string here.
            return {
                "status": "success", 
                "output": {
                     # Attempt to parse JSON if output looks like it, otherwise return raw
                    "agent_response": agent_output,
                    "available_slots": self._try_parse_json_list(agent_output, "Available slots found:"),
                    "booked_slot": self._extract_booked_slot(agent_output)
                    
                },
                "summary": f"Scheduling agent processed request. Result: {agent_output}"
            }

        except Exception as e:
            print(f"Error during LangChain agent execution: {e}")
            # Include traceback for debugging
            import traceback
            traceback.print_exc()
            return {"status": "failure", "output": None, "summary": f"Failed to process scheduling request via agent: {e}", "error_message": str(e)}

    # --- Helper methods for parsing agent output --- 
    def _try_parse_json_list(self, text: str, prefix: str) -> Optional[list]:
        """ Attempts to parse a JSON list from the agent output string. """
        try:
            if prefix in text:
                json_part = text.split(prefix, 1)[1].strip()
                # Find first [ and last ]
                start = json_part.find('[')
                end = json_part.rfind(']')
                if start != -1 and end != -1:
                    return json.loads(json_part[start:end+1])
        except Exception:
            pass
        return None

    def _extract_booked_slot(self, text: str) -> Optional[str]:
        """ Attempts to extract a booked slot time from the agent output string. """
        try:
            if "Successfully booked appointment" in text and "at" in text:
                 # Simple extraction, might need refinement
                 parts = text.split(" at ", 1)
                 if len(parts) > 1:
                     slot_part = parts[1].split(" for reason:")[0].strip()
                     return slot_part
        except Exception:
            pass
        return None

# (No need for the old helper methods like _interpret_date_range or LLM param extraction)

# --- Example Usage --- 
# (Can be updated to test prompts directly with agent.run)

# Example Usage (for testing)
# if __name__ == '__main__':
#     async def main():
#         agent = SchedulingAgent()
#         mock_context = {"patient_data": {"demographics": {"name": "Jane Doe"}}}
#         
#         # Test finding slots
#         find_kwargs = {"prompt": "Find available slots next week"}
#         find_result = await agent.run(mock_context, **find_kwargs)
#         print("Find Slots Result:", find_result)
#         
#         # Test booking (very basic)
#         book_kwargs = {"prompt": "Book an appointment at 9am"}
#         book_result = await agent.run(mock_context, **book_kwargs)
#         print("Book Result:", book_result)
#         
#     import asyncio
#     asyncio.run(main()) 